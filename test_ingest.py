"""Small smoke-test script for the ingestion pipeline.

Run this before the full Streamlit app to verify that a sample PDF is:
1. loaded from data/uploads/
2. chunked into smaller pieces
3. embedded and stored in ChromaDB

Usage:
    python test_ingest.py
"""

from pathlib import Path

from ingest import ingest_pdfs


def main() -> None:
    uploads_dir = Path(__file__).resolve().parent / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(uploads_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/uploads/.")
        print("Place a sample PDF there and run this script again.")
        return

    sample_pdf = pdf_files[0]
    print(f"Ingesting: {sample_pdf.name}")
    result = ingest_pdfs([sample_pdf])

    print("Ingestion result:")
    print(f"- ingested: {result.get('ingested', [])}")
    print(f"- skipped: {result.get('skipped', [])}")
    print(f"- chunk_count: {result.get('chunk_count', 0)}")

    # Re-read the collection directly to inspect the stored metadata.
    from ingest import get_collection

    collection = get_collection()
    stored = collection.get(include=["documents", "metadatas"])
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []

    print("\nStored chunks:")
    for index, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
        print(f"{index}. length={len(document.split())} words")
        print(f"   metadata={metadata}")


if __name__ == "__main__":
    main()
