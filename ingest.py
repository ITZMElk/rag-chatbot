"""Document ingestion pipeline for the RAG chatbot.

This module teaches the first half of the RAG story: documents are loaded,
chunked into smaller pieces, embedded, and stored so later retrieval can find
relevant context quickly.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it for every ingestion step."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_collection():
    """Return the persistent Chroma collection used for document storage."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def extract_text_from_pdf(file_path: Path) -> list[tuple[int, str]]:
    """Read a PDF and return a list of (page_number, text) pairs."""
    reader = PdfReader(str(file_path))
    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((page_number, text))
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split a document into overlapping chunks.

    In a production RAG system this is usually token-based, but we use a simple
    word-based chunker here to keep the project lightweight and easy to learn.
    """
    if not text.strip():
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def is_already_indexed(collection, source_name: str) -> bool:
    """Check whether a document from the same source has already been indexed."""
    try:
        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        for metadata in metadatas:
            if isinstance(metadata, list):
                for item in metadata:
                    if isinstance(item, dict) and item.get("source") == source_name:
                        return True
            elif isinstance(metadata, dict) and metadata.get("source") == source_name:
                return True
    except Exception:
        return False
    return False


def ingest_pdfs(file_paths: Sequence[Path | str]) -> dict[str, object]:
    """Ingest one or more PDFs into the Chroma vector store.

    The pipeline keeps the UX simple: each uploaded PDF is saved locally, its
    text is extracted page by page, chunked, embedded, and then stored with
    metadata so later queries can point back to the source document and page.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    collection = get_collection()
    model = get_embedding_model()

    persisted_paths: list[Path] = []
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            continue
        destination = DATA_DIR / path.name
        destination.write_bytes(path.read_bytes()) if not destination.exists() else None
        persisted_paths.append(destination)

    documents_to_embed: list[str] = []
    metadatas: list[dict[str, object]] = []
    ids: list[str] = []
    ingested_documents: list[str] = []
    skipped_documents: list[str] = []

    for path in persisted_paths:
        if is_already_indexed(collection, path.name):
            skipped_documents.append(path.name)
            continue

        pages = extract_text_from_pdf(path)
        if not pages:
            skipped_documents.append(path.name)
            continue

        for page_number, page_text in pages:
            chunks = chunk_text(page_text)
            for chunk_index, chunk in enumerate(chunks):
                documents_to_embed.append(chunk)
                metadatas.append(
                    {
                        "source": path.name,
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "chunk_hash": _hash_text(chunk),
                    }
                )
                ids.append(f"{path.stem}-{page_number}-{chunk_index}-{_hash_text(chunk)}")

        ingested_documents.append(path.name)

    if documents_to_embed:
        embeddings = model.encode(documents_to_embed, convert_to_numpy=True)
        collection.add(
            ids=ids,
            documents=documents_to_embed,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )

    return {
        "ingested": ingested_documents,
        "skipped": skipped_documents,
        "chunk_count": len(documents_to_embed),
    }


def get_indexed_documents() -> list[str]:
    """Return the list of document names currently stored in Chroma."""
    collection = get_collection()
    result = collection.get(include=["metadatas"])
    seen: set[str] = set()
    metadatas = result.get("metadatas") or []
    for metadata in metadatas:
        if isinstance(metadata, list):
            for item in metadata:
                if isinstance(item, dict):
                    source = item.get("source")
                    if source:
                        seen.add(str(source))
        elif isinstance(metadata, dict):
            source = metadata.get("source")
            if source:
                seen.add(str(source))
    return sorted(seen)


def get_document_stats() -> dict[str, int]:
    """Return simple document and chunk statistics for the sidebar UI."""
    collection = get_collection()
    result = collection.get(include=["documents"])
    docs = result.get("documents") or []
    return {
        "document_count": len(get_indexed_documents()),
        "chunk_count": len(docs),
    }
