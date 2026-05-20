import os
import logging
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from ingest import ingest_all, get_collection_stats
from query import ask

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Help Desk Chatbot API",
    description="RAG-powered help desk chatbot with web search fallback",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[Any]   # str for KB, {title,url} for web
    chunks_used: int
    source_type: str


class IngestRequest(BaseModel):
    force: bool = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    try:
        return get_collection_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest")
def trigger_ingest(req: IngestRequest = IngestRequest()):
    try:
        result = ingest_all(force=req.force)
        return result
    except Exception as exc:
        logger.error(f"Ingestion failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))



class DeleteRequest(BaseModel):
    source: str = ""   # specific filename to delete, empty = delete everything


@app.delete("/delete")
def delete_docs(req: DeleteRequest = DeleteRequest()):
    """
    Delete documents from the vector database.
    - Pass source="" (or omit) to wipe the entire collection.
    - Pass source="filename.pdf" to delete only that file's chunks.
    """
    try:
        from db import qdrant, COLLECTION_NAME
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        if req.source:
            # Delete only chunks that belong to this source file
            before = qdrant.get_collection(COLLECTION_NAME).points_count
            qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=req.source))]
                ),
            )
            after  = qdrant.get_collection(COLLECTION_NAME).points_count
            deleted = (before or 0) - (after or 0)
            logger.info(f"Deleted {deleted} chunks for source: {req.source}")
            return {"deleted": deleted, "source": req.source}
        else:
            # Wipe entire collection and recreate it empty
            from db import EMBEDDING_DIM
            from qdrant_client.models import Distance, VectorParams
            count = qdrant.get_collection(COLLECTION_NAME).points_count or 0
            qdrant.delete_collection(COLLECTION_NAME)
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info(f"Wiped entire collection ({count} chunks deleted)")
            return {"deleted": count, "source": "all"}
    except Exception as exc:
        logger.error(f"Delete failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        history = [{"role": m.role, "content": m.content} for m in req.history]
        result  = ask(req.question, history=history)
        return ChatResponse(**result)
    except Exception as exc:
        logger.error(f"Chat error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Serve frontend ────────────────────────────────────────────────────────────
frontend_dist = os.path.join(os.path.dirname(__file__), "../frontend/dist")
if os.path.isdir(frontend_dist):
    app.mount("/assets", StaticFiles(directory=f"{frontend_dist}/assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        return FileResponse(os.path.join(frontend_dist, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # ← reload=True spawns multiple processes that all try
                        #   to lock the same qdrant_db folder. Keep this False.
    )