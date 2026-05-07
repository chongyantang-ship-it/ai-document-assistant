import json
import re
from pathlib import Path


SECTION_HEADING_PATTERNS = [
    re.compile(r"^[A-Z][A-Za-z0-9 ,/&()'.-]{2,80}:$"),
    re.compile(r"^\d+(\.\d+)*\s+[A-Z].+"),
    re.compile(r"^Report Rubric: .+"),
    re.compile(r"^Presentation Rubric: .+"),
]


def clean_text(text):
    """Normalize line endings and collapse noisy whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def slugify(text):
    """Convert a label into a stable slug for chunk identifiers."""
    normalized = text.lower().strip()
    normalized = re.sub(r"^\d+(\.\d+)*\s*", "", normalized)
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_") or "section"


def normalize_heading(text):
    """Remove numbering and trailing punctuation from a heading."""
    normalized = text.strip().rstrip(":")
    normalized = re.sub(r"^\d+(\.\d+)*\s*", "", normalized)
    return normalized.strip()


def is_section_heading(line):
    """Return True when a line looks like a standalone heading."""
    compact = line.strip()
    if not compact:
        return False
    if len(compact.split()) > 14:
        return False
    return any(pattern.match(compact) for pattern in SECTION_HEADING_PATTERNS)


def guess_category(filename, section_title, text):
    """Assign a coarse category based on file name, section name, and text."""
    lower_name = filename.lower()
    lower_section = section_title.lower()
    lower_text = text.lower()

    if "presentation_rubric" in lower_name:
        return "presentation_rubric"
    if "rubric" in lower_name:
        return "rubric"
    if "presentation" in lower_name:
        return "presentation"
    if "template" in lower_name:
        return "report_template"
    if "project_design" in lower_name:
        return "system_design"

    if any(word in lower_section for word in ["submission", "due", "format", "similarity"]):
        return "factual_constraints"
    if any(word in lower_section for word in ["evaluation", "results", "metrics", "baseline"]):
        return "evaluation"
    if any(word in lower_section for word in ["theoretical", "workflow", "methodology"]):
        return "ai_methods"
    if any(word in lower_text for word in ["hd requires", "high distinction", "rubric"]):
        return "rubric"
    if any(word in lower_text for word in ["due", "word limit", "similarity", "submit", "format"]):
        return "factual_constraints"
    if any(word in lower_text for word in ["evaluation", "metrics", "baseline", "accuracy"]):
        return "evaluation"
    if any(word in lower_text for word in ["rag", "embedding", "rule-based", "structured knowledge"]):
        return "ai_methods"

    return "general"


def split_into_sections(text):
    """Split a raw text document into sections keyed by detected headings."""
    sections = []
    current_title = "Document Overview"
    current_lines = []

    for line in text.split("\n"):
        stripped = line.strip()
        if is_section_heading(stripped):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = normalize_heading(stripped)
            current_lines = []
            continue

        if stripped:
            current_lines.append(stripped)
        elif current_lines:
            current_lines.append("")

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return [(title, body) for title, body in sections if body]


def split_section_into_chunks(section_text, max_words):
    """Split a section into chunk-sized pieces while preserving paragraphs."""
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", section_text) if paragraph.strip()]
    current_parts = []
    current_word_count = 0
    chunks = []

    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.split())
        if current_parts and current_word_count + paragraph_word_count > max_words:
            chunks.append("\n\n".join(current_parts).strip())
            current_parts = []
            current_word_count = 0

        current_parts.append(paragraph)
        current_word_count += paragraph_word_count

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return chunks


def build_chunks_for_file(file_path, max_words):
    """Create deterministic chunk records for one source file."""
    text = clean_text(file_path.read_text(encoding="utf-8"))
    sections = split_into_sections(text)
    chunk_records = []

    for section_title, section_text in sections:
        section_chunks = split_section_into_chunks(section_text, max_words=max_words)
        for index, chunk_text in enumerate(section_chunks, start=1):
            chunk_records.append({
                "chunk_id": f"{file_path.stem}__{slugify(section_title)}__{index:02d}",
                "source_file": file_path.name,
                "section": section_title,
                "category": guess_category(file_path.name, section_title, chunk_text),
                "text": chunk_text,
            })

    return chunk_records


def create_chunks(raw_dir, output_path, max_words=130):
    """Build the processed chunk file from every raw text document."""
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    chunks = []

    for file_path in sorted(raw_dir.glob("*.txt")):
        chunks.extend(build_chunks_for_file(file_path, max_words=max_words))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(chunks, output_file, ensure_ascii=False, indent=2)

    return chunks


def print_chunk_preview(chunks, preview_count=3):
    """Print a small preview of generated chunks for quick inspection."""
    for chunk in chunks[:preview_count]:
        print("\n---")
        print("chunk_id:", chunk["chunk_id"])
        print("source_file:", chunk["source_file"])
        print("section:", chunk["section"])
        print("category:", chunk["category"])
        print("text preview:", chunk["text"][:300])


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    output_path = project_root / "data" / "processed" / "chunks.json"

    chunk_data = create_chunks(raw_dir, output_path)
    print(f"Created {len(chunk_data)} chunks.")
    print(f"Saved to: {output_path}")
    print_chunk_preview(chunk_data)
