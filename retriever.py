"""Retrieval layer for the RAG chatbot.

This step is the bridge between stored knowledge and the language model. The
query is embedded with the same model used for the documents, and the most
similar chunks are fetched so the generator can answer from context rather than
from memory alone.
"""

from __future__ import annotations

from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME, SIMILARITY_THRESHOLD, TOP_K


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Reuse a single embedding model for both ingestion and retrieval."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_collection():
    """Return the persistent Chroma collection used for retrieval."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def retrieve_context(query: str, top_k: int = TOP_K, similarity_threshold: float = SIMILARITY_THRESHOLD) -> list[dict[str, object]]:
    """Embed a query and return the best matching chunks with their metadata."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    model = get_embedding_model()
    query_embedding = model.encode(query, convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []

    retrieved_chunks: list[dict[str, object]] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        similarity = max(0.0, 1.0 - float(distance))
        if similarity >= similarity_threshold:
            retrieved_chunks.append(
                {
                    "content": document,
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page", "unknown"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "similarity": round(similarity, 4),
                }
            )

    return retrieved_chunks
