
"""
Prepare documents for the academic document assistant.

This script rebuilds:
1. data/processed/structured_facts.json
2. data/processed/chunks.json

It makes the prototype more general because the runtime knowledge base can be
rebuilt from whatever text documents are placed in data/raw/.
"""

from pathlib import Path
import json
import re

from document_loader import load_text_documents, combine_documents
from fact_extractor import extract_structured_facts, save_structured_facts


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
FACTS_PATH = PROCESSED_DIR / "structured_facts.json"


def guess_category(text):
    """
    Assign a rough category to a text chunk.
    """
    lowered = text.lower()

    if any(k in lowered for k in ["due", "deadline", "word limit", "submit", "submission", "pdf", "file format"]):
        return "factual_constraints"

    if any(k in lowered for k in ["presentation", "slides", "oral"]):
        return "presentation"

    if any(k in lowered for k in ["rubric", "criteria", "hd", "high distinction", "marking"]):
        return "rubric_interpretation"

    if any(k in lowered for k in ["github", "repository", "repo", "commit", "branch"]):
        return "submission_rules"

    if any(k in lowered for k in ["evaluation", "baseline", "accuracy", "hallucination", "retrieval", "metric"]):
        return "evaluation"

    if any(k in lowered for k in ["rag", "llm", "semantic", "embedding", "rule-based", "structured knowledge"]):
        return "ai_methods"

    return "general"


def split_into_sections(text):
    """
    Split text into rough sections using headings and blank lines.
    """
    lines = text.splitlines()
    sections = []
    current_title = "Document"
    current_lines = []

    heading_pattern = re.compile(r"^\s*(#{1,6}\s+.+|[A-Z][A-Za-z0-9 ,:/&()_-]{3,80})\s*$")

    for line in lines:
        stripped = line.strip()

        if heading_pattern.match(stripped) and len(stripped.split()) <= 12:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
                current_lines = []
            current_title = stripped.lstrip("#").strip()
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return [(title, body) for title, body in sections if body.strip()]


def chunk_text(text, max_chars=900, overlap=120):
    """
    Split long text into overlapping chunks.
    """
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def build_chunks(documents):
    """
    Build chunk dictionaries compatible with the retriever.
    """
    chunks = []

    for doc in documents:
        sections = split_into_sections(doc["text"])

        for section_index, (section_title, section_text) in enumerate(sections, start=1):
            small_chunks = chunk_text(section_text)

            for chunk_index, chunk in enumerate(small_chunks, start=1):
                chunk_id = (
                    f"{Path(doc['filename']).stem}"
                    f"__{section_title.lower().replace(' ', '_')[:40]}"
                    f"__{section_index:02d}_{chunk_index:02d}"
                )

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source_file": doc["filename"],
                        "section": section_title,
                        "category": guess_category(section_title + "\n" + chunk),
                        "text": chunk,
                    }
                )

    return chunks


def save_chunks(chunks, output_path: Path = CHUNKS_PATH):
    """
    Save chunks to JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    documents = load_text_documents()
    if not documents:
        raise RuntimeError("No supported text documents found in data/raw/.")

    print(f"Loaded {len(documents)} raw document(s):")
    for doc in documents:
        print(f"- {doc['filename']} ({len(doc['text'])} chars)")

    combined_text = combine_documents(documents)
    facts = extract_structured_facts(combined_text)
    save_structured_facts(facts, FACTS_PATH)

    chunks = build_chunks(documents)
    save_chunks(chunks, CHUNKS_PATH)

    print()
    print(f"Saved structured facts to: {FACTS_PATH}")
    print(f"Saved {len(chunks)} chunks to: {CHUNKS_PATH}")
    print()
    print("Detected facts:")
    for key, item in facts.items():
        if isinstance(item, dict):
            print(f"- {key}: {item.get('value')}")
        else:
            print(f"- {key}: {item}")


if __name__ == "__main__":
    main()
