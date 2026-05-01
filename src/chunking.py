
import json
import re
from pathlib import Path


def clean_text(text):
    """Basic text cleaning while preserving headings and important content."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def guess_category(filename, text):
    """Assign a simple category based on source filename and content."""
    lower_name = filename.lower()
    lower_text = text.lower()

    if "presentation" in lower_name:
        return "presentation"
    if "template" in lower_name:
        return "report_template"
    if "project_design" in lower_name:
        return "system_design"
    if any(word in lower_text for word in ["due", "word limit", "similarity", "submit", "format"]):
        return "factual_constraints"
    if any(word in lower_text for word in ["evaluation", "metrics", "baseline", "accuracy"]):
        return "evaluation"
    if any(word in lower_text for word in ["rag", "embedding", "rule-based", "structured knowledge"]):
        return "ai_methods"

    return "general"


def split_into_paragraphs(text):
    """Split text into meaningful paragraphs."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs


def create_chunks(raw_dir, output_path, max_words=180):
    """
    Create chunks from raw txt files.
    For this prototype, each chunk is built from paragraphs and capped by max_words.
    """
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)

    chunks = []
    chunk_counter = 1

    for file_path in sorted(raw_dir.glob("*.txt")):
        text = clean_text(file_path.read_text(encoding="utf-8"))
        paragraphs = split_into_paragraphs(text)

        current_parts = []
        current_word_count = 0

        for para in paragraphs:
            words = para.split()
            para_word_count = len(words)

            if current_parts and current_word_count + para_word_count > max_words:
                chunk_text = "\n\n".join(current_parts).strip()

                chunks.append({
                    "chunk_id": f"chunk_{chunk_counter:03d}",
                    "source_file": file_path.name,
                    "section": "auto_chunked_section",
                    "category": guess_category(file_path.name, chunk_text),
                    "text": chunk_text
                })

                chunk_counter += 1
                current_parts = []
                current_word_count = 0

            current_parts.append(para)
            current_word_count += para_word_count

        if current_parts:
            chunk_text = "\n\n".join(current_parts).strip()

            chunks.append({
                "chunk_id": f"chunk_{chunk_counter:03d}",
                "source_file": file_path.name,
                "section": "auto_chunked_section",
                "category": guess_category(file_path.name, chunk_text),
                "text": chunk_text
            })

            chunk_counter += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return chunks


if __name__ == "__main__":
    project_root = Path("/content/ai-document-assistant")
    raw_dir = project_root / "data" / "raw"
    output_path = project_root / "data" / "processed" / "chunks.json"

    chunks = create_chunks(raw_dir, output_path)

    print(f"Created {len(chunks)} chunks.")
    print(f"Saved to: {output_path}")

    for chunk in chunks[:3]:
        print("\n---")
        print("chunk_id:", chunk["chunk_id"])
        print("source_file:", chunk["source_file"])
        print("category:", chunk["category"])
        print("text preview:", chunk["text"][:300])
