"""Project-wide constants and environment configuration for the RAG chatbot."""

from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "uploads"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "rag_documents"

# The embedding model is intentionally lightweight so it can run locally.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Chunking controls the granularity of the retrieved context.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval settings control how much context is passed to the generator.
TOP_K = int(os.getenv("TOP_K", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.2"))

# Gemini generation settings.
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite")

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
