# Help Desk Chatbot — Setup Guide

## Project structure

```
ChatBot/
├── backend/
│   ├── main.py          ← FastAPI app
│   ├── ingest.py        ← S3 → ChromaDB ingestion pipeline
│   ├── query.py         ← RAG query + web search fallback
│   ├── requirements.txt
│   ├── .env             ← your secrets (copy from .env.example)
│   └── .env.example
└── frontend/
    └── index.html       ← single-file UI (open directly in browser)
```

---

## 1. Fill in your `.env`

Copy `.env.example` to `.env` and fill in every value.

**The most common mistake:** `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` left blank.
This MUST be the name of your Azure OpenAI **embedding** deployment
(e.g. `text-embedding-ada-002` or `text-embedding-3-small`), NOT your chat deployment.

---

## 2. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

> **Python 3.10+** is required (uses `list[str] | None` type hints).

---

## 3. Run the backend

```bash
cd backend
python main.py
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

---

## 4. Open the frontend

Just open `frontend/index.html` in your browser — no build step needed.
It connects to `http://localhost:8000` automatically.

---

## 5. Ingest your PDFs

Click **"↑ Sync documents"** in the sidebar, or call the API directly:

```bash
curl -X POST http://localhost:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"force": false}'
```

Use `"force": true` to re-ingest files already in ChromaDB.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT is not set` | Add it to `.env` |
| ChromaDB telemetry errors | Already silenced in this version — upgrade posthog if still seen |
| `Error code: 400 — Missing required parameter: messages` | Wrong deployment used for embeddings — check `.env` |
| S3 access denied | Check `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` |
| `collection.count()` returns 0 after ingest | Run ingest first; check for errors in console |
