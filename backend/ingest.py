import os
import uuid
import tempfile
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import boto3
import fitz  # PyMuPDF
from openai import AzureOpenAI
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

# ── Shared Qdrant singleton (no duplicate lock) ───────────────────────────────
from db import qdrant, COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Validate required env vars ────────────────────────────────────────────────
_REQUIRED = [
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION", "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY", "S3_BUCKET_NAME",
]
for _var in _REQUIRED:
    if not os.getenv(_var):
        logger.warning(f"Missing env var: {_var}")

# ── Azure OpenAI ──────────────────────────────────────────────────────────────
azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
if not EMBEDDING_DEPLOYMENT:
    raise RuntimeError(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT is not set in .env\n"
        "Example value: text-embedding-ada-002"
    )

# ── AWS S3 ────────────────────────────────────────────────────────────────────
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-east-1"),
)
BUCKET = os.getenv("S3_BUCKET_NAME")


# ── Helpers ───────────────────────────────────────────────────────────────────

def list_s3_pdfs() -> list[str]:
    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".pdf"):
                keys.append(obj["Key"])
    logger.info(f"Found {len(keys)} PDFs in S3 bucket '{BUCKET}'")
    return keys


def download_pdf(key: str) -> str:
    safe_name  = key.replace("/", "_").replace("\\", "_")
    local_path = Path(tempfile.gettempdir()) / safe_name
    s3_client.download_file(BUCKET, key, str(local_path))
    logger.info(f"Downloaded: {key}")
    return str(local_path)


def extract_text(pdf_path: str) -> str:
    doc   = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []
    step   = max(chunk_size - overlap, 1)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = azure_client.embeddings.create(input=texts, model=EMBEDDING_DEPLOYMENT)
    return [item.embedding for item in response.data]


def already_ingested(source_key: str) -> bool:
    results, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_key))]
        ),
        limit=1,
    )
    return len(results) > 0


def delete_by_source(source_key: str):
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_key))]
        ),
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def ingest_all(force: bool = False) -> dict:
    keys  = list_s3_pdfs()
    stats = {"total": len(keys), "ingested": 0, "skipped": 0, "errors": []}

    for key in keys:
        if not force and already_ingested(key):
            logger.info(f"Skipping (already ingested): {key}")
            stats["skipped"] += 1
            continue

        local_path: str | None = None
        try:
            local_path = download_pdf(key)
            text       = extract_text(local_path)

            if not text.strip():
                logger.warning(f"No text extracted from {key}")
                stats["errors"].append({"file": key, "error": "No text extracted"})
                continue

            chunks = chunk_text(text)
            logger.info(f"{key}: {len(chunks)} chunk(s)")

            if force and already_ingested(key):
                delete_by_source(key)

            # Embed in batches of 16
            all_embeddings: list[list[float]] = []
            for i in range(0, len(chunks), 16):
                all_embeddings.extend(embed_texts(chunks[i : i + 16]))

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=all_embeddings[i],
                    payload={"text": chunks[i], "source": key, "chunk_index": i},
                )
                for i in range(len(chunks))
            ]

            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            stats["ingested"] += 1
            logger.info(f"Ingested: {key}")

        except Exception as exc:
            logger.error(f"Error ingesting {key}: {exc}")
            stats["errors"].append({"file": key, "error": str(exc)})

        finally:
            if local_path:
                try:
                    os.remove(local_path)
                except OSError:
                    pass

    return stats


def get_collection_stats() -> dict:
    info = qdrant.get_collection(COLLECTION_NAME)
    return {"total_chunks": info.points_count, "collection": COLLECTION_NAME}