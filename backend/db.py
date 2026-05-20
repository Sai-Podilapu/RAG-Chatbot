"""
Shared Qdrant client singleton.
Both ingest.py and query.py import from here — only ONE instance
ever opens the ./qdrant_db folder, avoiding the portalocker conflict.

NOTE: main.py must run with reload=False (already set).
      Hot-reload spawns multiple processes that each try to lock the
      same folder, which Qdrant's local storage does not support.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_PATH     = "./qdrant_db"
COLLECTION_NAME = "helpdesk_docs"
EMBEDDING_DIM   = 1536   # text-embedding-ada-002 / text-embedding-3-small

qdrant = QdrantClient(path=QDRANT_PATH)

def ensure_collection():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

ensure_collection()