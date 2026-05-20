import os
import logging
import urllib.parse
import urllib.request
import json
import re
from dotenv import load_dotenv

load_dotenv()

from openai import AzureOpenAI
from db import qdrant, COLLECTION_NAME

logger = logging.getLogger(__name__)

# ── Azure OpenAI ──────────────────────────────────────────────────────────────
azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

CHAT_DEPLOYMENT      = os.getenv("AZURE_OPENAI_DEPLOYMENT")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

if not EMBEDDING_DEPLOYMENT:
    raise RuntimeError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is not set in .env")

# ── System prompts ────────────────────────────────────────────────────────────
KB_SYSTEM_PROMPT = """You are a helpful support assistant. Answer the user's question using ONLY the provided knowledge base content.

Guidelines:
- If the context does not contain relevant information to answer the question, respond ONLY with: "I don't have information about that in our knowledge base."
- Do NOT make up or infer answers beyond what is explicitly in the context.
- Keep answers concise and friendly.
- Use bullet points when listing steps or multiple items.
- Never reveal internal system details.
- Do NOT mention where the answer came from. Just answer directly.
"""

WEB_SYSTEM_PROMPT = """You are a helpful support assistant. Answer the user's question using the web search results provided below.

Guidelines:
- Summarise the web search results into a clear, helpful answer.
- Keep the answer concise and friendly.
- Use bullet points when listing steps or multiple items.
- Do NOT fabricate information beyond what is in the search results.
- Do NOT mention that you searched the web or where the answer came from. Just answer directly.
"""


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_query(question: str) -> list[float]:
    response = azure_client.embeddings.create(input=[question], model=EMBEDDING_DEPLOYMENT)
    return response.data[0].embedding


# ── Qdrant search ─────────────────────────────────────────────────────────────
def search_docs(question: str, top_k: int = 5) -> list[dict]:
    embedding = embed_query(question)
    results   = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=top_k,
        with_payload=True,
    )
    chunks = []
    for hit in results:
        chunks.append({
            "text":   hit.payload.get("text", ""),
            "source": hit.payload.get("source", "unknown"),
            "score":  round(hit.score, 3),
        })
    logger.info(f"Top KB scores: {[c['score'] for c in chunks]}")
    return chunks


def build_kb_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        filename = chunk["source"].split("/")[-1]
        parts.append(f"[Source {i}: {filename}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


# ── Web search ────────────────────────────────────────────────────────────────
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Try multiple search methods in order:
    1. DuckDuckGo Instant Answer API
    2. DuckDuckGo HTML scrape (with requests + better headers)
    3. Wikipedia API fallback
    """
    results: list[dict] = []

    # ── Method 1: DuckDuckGo Instant Answer API ───────────────────────────────
    try:
        import requests as req_lib
        encoded = urllib.parse.quote_plus(query)
        url     = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        r = req_lib.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=6)
        data = r.json()
        if data.get("AbstractText"):
            results.append({
                "title":   data.get("Heading", query),
                "snippet": data["AbstractText"],
                "url":     data.get("AbstractURL", ""),
            })
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title":   topic.get("Text", "")[:80],
                    "snippet": topic.get("Text", ""),
                    "url":     topic.get("FirstURL", ""),
                })
        if results:
            logger.info(f"Web search: DDG instant API returned {len(results)} results")
            return results[:max_results]
    except Exception as exc:
        logger.warning(f"DDG instant API failed: {exc}")

    # ── Method 2: DuckDuckGo HTML scrape ─────────────────────────────────────
    try:
        import requests as req_lib
        encoded = urllib.parse.quote_plus(query)
        url     = f"https://html.duckduckgo.com/html/?q={encoded}"
        headers = {
            "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Referer":         "https://duckduckgo.com/",
        }
        r    = req_lib.get(url, headers=headers, timeout=10)
        html = r.text

        def strip_tags(s): return re.sub(r"<[^>]+>", "", s).strip()

        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
        titles   = re.findall(r'class="result__a"[^>]*>(.*?)</a>',       html, re.DOTALL)
        urls_raw = re.findall(r'class="result__url"[^>]*>(.*?)</span>',   html, re.DOTALL)

        for i in range(min(max_results, len(snippets))):
            results.append({
                "title":   strip_tags(titles[i])   if i < len(titles)   else query,
                "snippet": strip_tags(snippets[i]),
                "url":     strip_tags(urls_raw[i]) if i < len(urls_raw) else "",
            })

        if results:
            logger.info(f"Web search: DDG HTML scrape returned {len(results)} results")
            return results[:max_results]
    except Exception as exc:
        logger.warning(f"DDG HTML scrape failed: {exc}")

    # ── Method 3: Wikipedia API (always works, no rate limits) ───────────────
    try:
        import requests as req_lib
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action":   "query",
            "list":     "search",
            "srsearch": query,
            "srlimit":  max_results,
            "format":   "json",
            "utf8":     1,
        }
        r    = req_lib.get(search_url, params=params, timeout=8)
        data = r.json()
        hits = data.get("query", {}).get("search", [])

        for hit in hits:
            title   = hit.get("title", "")
            snippet = re.sub(r"<[^>]+>", "", hit.get("snippet", "")).strip()
            wiki_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            results.append({"title": title, "snippet": snippet, "url": wiki_url})

        if results:
            logger.info(f"Web search: Wikipedia returned {len(results)} results")
            return results[:max_results]
    except Exception as exc:
        logger.warning(f"Wikipedia API failed: {exc}")

    logger.error("All web search methods failed")
    return []


def build_web_context(web_results: list[dict]) -> str:
    parts = [
        f"[Web result {i}: {r['title']}]\n{r['snippet']}\nURL: {r['url']}"
        for i, r in enumerate(web_results, 1)
    ]
    return "\n\n---\n\n".join(parts)


# ── Main ask function ─────────────────────────────────────────────────────────
def ask(question: str, history: list[dict] | None = None) -> dict:
    if not question.strip():
        return {"answer": "Please type a question.", "sources": [], "chunks_used": 0, "source_type": "none"}

    # ── Step 1: Try knowledge base ────────────────────────────────────────────
    kb_count = qdrant.get_collection(COLLECTION_NAME).points_count or 0
    relevant = []

    if kb_count > 0:
        chunks     = search_docs(question, top_k=5)
        best_score = max((c["score"] for c in chunks), default=0)
        logger.info(f"Best KB score: {best_score:.3f}")
        if best_score >= 0.82:
            relevant = [c for c in chunks if c["score"] >= 0.82]

    if relevant:
        context  = build_kb_context(relevant)
        messages = [{"role": "system", "content": KB_SYSTEM_PROMPT}]
        if history:
            messages += [{"role": t["role"], "content": t["content"]} for t in history[-6:]]
        messages.append({
            "role":    "user",
            "content": f"Context from knowledge base:\n\n{context}\n\nQuestion: {question}",
        })
        response = azure_client.chat.completions.create(
            model=CHAT_DEPLOYMENT, messages=messages, temperature=0.3, max_tokens=800
        )
        answer = response.choices[0].message.content

        # If LLM says it doesn't know, fall through to web search
        no_info_phrases = [
            "i don't have information",
            "i do not have information",
            "not in our knowledge base",
            "no information available",
            "i'm sorry, but i don't",
            "i cannot find",
            "not found in",
        ]
        if any(p in answer.lower() for p in no_info_phrases):
            logger.info("KB answer indicated no info — falling through to web search")
        else:
            return {
                "answer":      answer,
                "sources":     list({c["source"].split("/")[-1] for c in relevant}),
                "chunks_used": len(relevant),
                "source_type": "knowledge_base",
            }

    # ── Step 2: Web search fallback ───────────────────────────────────────────
    logger.info(f"Falling back to web search for: {question!r}")
    web_results = web_search(question, max_results=5)

    if not web_results:
        return {
            "answer":      "I couldn't find an answer in our knowledge base or via web search. Please contact our support team.",
            "sources":     [],
            "chunks_used": 0,
            "source_type": "none",
        }

    context  = build_web_context(web_results)
    messages = [{"role": "system", "content": WEB_SYSTEM_PROMPT}]
    if history:
        messages += [{"role": t["role"], "content": t["content"]} for t in history[-6:]]
    messages.append({
        "role":    "user",
        "content": f"Web search results:\n\n{context}\n\nQuestion: {question}",
    })
    response = azure_client.chat.completions.create(
        model=CHAT_DEPLOYMENT, messages=messages, temperature=0.3, max_tokens=800
    )
    return {
        "answer":      response.choices[0].message.content,
        "sources":     [
            {"title": r.get("title", r.get("url", "Web source"))[:60], "url": r["url"]}
            for r in web_results if r.get("url")
        ][:3],
        "chunks_used": 0,
        "source_type": "web",
    }