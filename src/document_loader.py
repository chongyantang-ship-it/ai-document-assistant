
"""
Document loading utilities for the academic document assistant.

This module provides a lightweight general ingestion layer.
It reads text-based academic documents from data/raw/ and prepares them
for chunking and fact extraction.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_text_documents(raw_dir: Path = RAW_DIR):
    """
    Load all supported text documents from the raw data directory.

    Returns:
        list[dict]: Each item contains filename, path, suffix, and text.
    """
    documents = []

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw document directory does not exist: {raw_dir}")

    for path in sorted(raw_dir.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore").strip()

        if not text:
            continue

        documents.append(
            {
                "filename": path.name,
                "path": str(path),
                "suffix": path.suffix.lower(),
                "text": text,
            }
        )

    return documents


def combine_documents(documents):
    """
    Combine all loaded documents into one text block for global fact extraction.
    """
    parts = []

    for doc in documents:
        parts.append(f"\n\n===== SOURCE FILE: {doc['filename']} =====\n")
        parts.append(doc["text"])

    return "\n".join(parts).strip()


if __name__ == "__main__":
    docs = load_text_documents()
    print(f"Loaded {len(docs)} document(s).")
    for doc in docs:
        print(f"- {doc['filename']} ({len(doc['text'])} characters)")
