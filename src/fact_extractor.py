
"""
General structured fact extraction for academic assignment documents.

This module extracts common assessment facts from uploaded assignment documents,
such as due dates, word limits, file formats, presentation requirements,
similarity limits, and repository requirements.

The extraction is intentionally conservative:
- It uses deterministic patterns first.
- It stores evidence text for transparency.
- If a fact cannot be found, the value is left as None.
"""

import re
from pathlib import Path
import json

from document_loader import load_text_documents, combine_documents


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FACTS_PATH = PROCESSED_DIR / "structured_facts.json"


def find_first(patterns, text, flags=re.IGNORECASE | re.MULTILINE):
    """
    Find the first matching pattern in text.

    Returns:
        dict or None: value and evidence if matched.
    """
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            value = match.group(1).strip() if match.groups() else match.group(0).strip()
            evidence = match.group(0).strip()
            return {
                "value": value,
                "evidence": evidence,
            }
    return None


def extract_structured_facts(text):
    """
    Extract common academic assignment facts from combined document text.
    """
    facts = {}

    facts["report_due_date"] = find_first(
        [
            r"(?:report|final report|assignment|assessment)[^\n]{0,80}?(?:due|deadline|submit|submission)[^\n]{0,40}?([0-9]{1,2}\s+[A-Za-z]+\s+20[0-9]{2}(?:\s+[0-9]{1,2}[:.][0-9]{2})?)",
            r"(?:due|deadline|submission)[^\n]{0,40}?([0-9]{1,2}\s+[A-Za-z]+\s+20[0-9]{2}(?:\s+[0-9]{1,2}[:.][0-9]{2})?)",
            r"([0-9]{1,2}\s+[A-Za-z]+\s+20[0-9]{2}\s+[0-9]{1,2}[:.][0-9]{2})",
        ],
        text,
    )

    facts["word_limit"] = find_first(
        [
            r"(?:word limit|maximum words|up to|no more than)[^\n]{0,40}?([0-9,]{3,6}\s*words?)",
            r"([0-9,]{3,6}\s*words?)",
        ],
        text,
    )

    facts["report_format"] = find_first(
        [
            r"(?:report|submission)[^\n]{0,80}?(PDF)",
            r"(PDF file)",
        ],
        text,
    )

    facts["slide_format"] = find_first(
        [
            r"(?:slides?|presentation)[^\n]{0,80}?((?:PowerPoint|PPTX|PDF|Google Slides)[^\n]{0,40})",
            r"((?:PowerPoint|PPTX|Google Slides))",
        ],
        text,
    )

    facts["presentation_duration"] = find_first(
        [
            r"(?:presentation|oral)[^\n]{0,80}?([0-9]+(?:\.[0-9]+)?\s*(?:minutes?|mins?))",
            r"([0-9]+(?:\.[0-9]+)?\s*(?:minutes?|mins?))",
        ],
        text,
    )

    facts["similarity_limit"] = find_first(
        [
            r"(?:similarity|Turnitin)[^\n]{0,80}?([0-9]{1,2}\s*%)",
            r"([0-9]{1,2}\s*%)\s*(?:similarity|Turnitin)",
        ],
        text,
    )

    facts["github_required"] = find_first(
        [
            r"((?:GitHub|repository|repo)[^\n]{0,100}?(?:required|must|need|submit|include)[^\n]{0,80})",
            r"((?:required|must|need|submit|include)[^\n]{0,80}?(?:GitHub|repository|repo)[^\n]{0,80})",
        ],
        text,
    )

    facts["report_sections"] = find_first(
        [
            r"((?:executive summary|introduction|methodology|evaluation|discussion|conclusion|references)[^\n]{0,300})",
            r"((?:report should include|report must include)[^\n]{0,300})",
        ],
        text,
    )

    facts["required_ai_methods"] = {
        "value": (
            "The assistant may combine structured knowledge representation, "
            "rule-based reasoning, semantic retrieval, retrieval-augmented generation, "
            "evidence-grounded answer generation, and empirical evaluation."
        ),
        "evidence": "This is derived from the project design and implemented system architecture.",
    }

    facts["evaluation_summary"] = {
        "value": (
            "The system should be evaluated using representative test questions, "
            "baseline comparisons, answer accuracy, retrieval hit rate, citation support, "
            "hallucination rate, unsupported handling accuracy, and response time."
        ),
        "evidence": "This is derived from the evaluation design of the prototype.",
    }

    facts["official_guidance_warning"] = {
        "value": (
            "The assistant is a learning support tool and should not replace official course guidance, "
            "lecturers, tutors, or assessment instructions."
        ),
        "evidence": "This is an ethical limitation of the system.",
    }

    return facts


def save_structured_facts(facts, output_path: Path = FACTS_PATH):
    """
    Save extracted facts to JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    documents = load_text_documents()
    combined_text = combine_documents(documents)
    facts = extract_structured_facts(combined_text)
    save_structured_facts(facts)

    print(f"Saved structured facts to: {FACTS_PATH}")
    for key, item in facts.items():
        value = item.get("value") if isinstance(item, dict) else item
        print(f"- {key}: {value}")
