import json
import re
from pathlib import Path

from brief_structure import slugify, split_into_sections
from document_loader import read_document_text
from fact_extractor import save_structured_facts
from runtime_config import iter_runtime_sources


def guess_category(filename, section_title, text):
    """Assign a coarse category based on file name, section name, and text."""
    lower_name = filename.lower()
    lower_section = section_title.lower()
    lower_text = text.lower()

    if "presentation_rubric" in lower_name:
        return "presentation_rubric"
    if "rubric" in lower_name:
        return "rubric"
    if "report rubric" in lower_section or "presentation rubric" in lower_section:
        return "rubric"
    if "presentation" in lower_section:
        return "presentation"
    if "presentation" in lower_name:
        return "presentation"
    if "template" in lower_name:
        return "report_template"
    if "project_design" in lower_name:
        return "system_design"

    if any(word in lower_section for word in ["task overview", "objective", "requirements", "extensions"]):
        return "workflow_planning"
    if any(word in lower_section for word in ["submission", "due", "format", "similarity"]):
        return "factual_constraints"
    if any(word in lower_section for word in ["evaluation", "results", "metrics", "baseline"]):
        return "evaluation"
    if any(word in lower_section for word in ["theoretical", "workflow", "methodology"]):
        return "ai_methods"
    if any(word in lower_section for word in ["ethics", "integrity", "policy"]):
        return "ethics_policy"
    if any(word in lower_text for word in ["hd requires", "high distinction", "rubric"]):
        return "rubric"
    if any(word in lower_text for word in ["due", "word limit", "similarity", "submit", "format"]):
        return "factual_constraints"
    if any(word in lower_text for word in ["evaluation", "metrics", "baseline", "accuracy"]):
        return "evaluation"
    if any(word in lower_text for word in ["rag", "embedding", "rule-based", "structured knowledge"]):
        return "ai_methods"
    if any(word in lower_text for word in ["ethical", "genai", "academic integrity", "official guidance"]):
        return "ethics_policy"
    if any(word in lower_text for word in ["role allocation", "divide the work", "presentation responsibility"]):
        return "workflow_planning"

    return "general"


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


def build_chunks_for_file(file_path, authority, role, max_words):
    """Create deterministic chunk records for one source file."""
    text = read_document_text(file_path)
    sections = split_into_sections(text)
    chunk_records = []
    source_id = slugify(file_path.stem)

    for section_title, section_text in sections:
        section_chunks = split_section_into_chunks(section_text, max_words=max_words)
        for index, chunk_text in enumerate(section_chunks, start=1):
            chunk_records.append({
                "chunk_id": f"{source_id}__{slugify(section_title)}__{index:02d}",
                "source_file": file_path.name,
                "authority": authority,
                "role": role,
                "section": section_title,
                "category": guess_category(file_path.name, section_title, chunk_text),
                "text": chunk_text,
            })

    return chunk_records


def create_chunks(output_path, max_words=130, config_path=None):
    """Build the processed chunk file from the configured runtime sources."""
    output_path = Path(output_path)
    chunks = []

    for source in iter_runtime_sources(config_path=config_path):
        file_path = source["path"]
        if not file_path.exists():
            raise FileNotFoundError(
                f"Configured runtime source does not exist: {file_path}"
            )
        chunks.extend(
            build_chunks_for_file(
                file_path=file_path,
                authority=source["authority"],
                role=source["role"],
                max_words=max_words,
            )
        )

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
    chunks_output_path = project_root / "data" / "processed" / "chunks.json"
    facts_output_path = project_root / "data" / "processed" / "structured_facts.json"

    chunk_data = create_chunks(chunks_output_path)
    facts_data = save_structured_facts(facts_output_path)
    print(f"Created {len(chunk_data)} chunks.")
    print(f"Saved chunks to: {chunks_output_path}")
    print(f"Saved structured facts to: {facts_output_path}")
    print(f"Active brief source: {facts_data.get('brief_source_file', 'unknown')}")
    print_chunk_preview(chunk_data)
