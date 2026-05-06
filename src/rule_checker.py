
import json
import re
from pathlib import Path


def load_facts(facts_path=None):
    """Load structured facts from JSON."""
    if facts_path is None:
        facts_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "structured_facts.json"
    else:
        facts_path = Path(facts_path)

    with open(facts_path, "r", encoding="utf-8") as f:
        return json.load(f)


def contains_whole_word(text, word):
    """Check whether a word appears as a whole word."""
    pattern = r"\b" + re.escape(word) + r"\b"
    return re.search(pattern, text) is not None


def check_rule_based_answer(question, facts):
    """
    Return a rule-based answer for exact assessment constraints.
    If the question is not covered by rules, return None.
    """
    q = question.lower()

    # Report sections should be checked before GitHub/repository
    if any(k in q for k in ["report section", "sections", "include in the report", "table of contents"]):
        sections = facts["report_sections"]
        return {
            "answer": "The report should include: " + ", ".join(sections) + ".",
            "evidence": [
                {
                    "source": "structured_facts.json",
                    "field": "report_sections",
                    "text": ", ".join(sections)
                }
            ],
            "confidence": "High",
            "route": "rule-based"
        }

    # Report deadline
    if any(k in q for k in ["due", "deadline", "when is the report", "report due"]):
        return {
            "answer": f"The report is due on {facts['report_due_date']}.",
            "evidence": [
                {
                    "source": "structured_facts.json",
                    "field": "report_due_date",
                    "text": facts["report_due_date"]
                }
            ],
            "confidence": "High",
            "route": "rule-based"
        }

    # Word limit
    if any(k in q for k in ["word limit", "how many words", "3500", "words"]):
        return {
            "answer": f"The report word limit is {facts['report_word_limit']}.",
            "evidence": [
                {
                    "source": "structured_facts.json",
                    "field": "report_word_limit",
                    "text": facts["report_word_limit"]
                }
            ],
            "confidence": "High",
            "route": "rule-based"
        }

    # Similarity / Turnitin
    if any(k in q for k in ["similarity", "turnitin", "plagiarism"]):
        return {
            "answer": f"The maximum allowed similarity score is {facts['similarity_limit']}.",
            "evidence": [
                {
                    "source": "structured_facts.json",
                    "field": "similarity_limit",
                    "text": facts["similarity_limit"]
                }
            ],
            "confidence": "High",
            "route": "rule-based"
        }

    # Report file format
    if ("report" in q and any(k in q for k in ["format", "file type", "submit", "submission", "pdf"])):
        return {
            "answer": f"The report should be submitted as a {facts['report_format']} file. The suggested filename is {facts['report_filename']}.",
            "evidence": [
                {
                    "source": "structured_facts.json",
                    "field": "report_format",
                    "text": facts["report_format"]
                },
                {
                    "source": "structured_facts.json",
                    "field": "report_filename",
                    "text": facts["report_filename"]
                }
            ],
            "confidence": "High",
            "route": "rule-based"
        }

    # Slide file format
    if any(k in q for k in ["slide", "slides", "powerpoint", "ppt", "pptx"]):
        return {
            "answer": f"The presentation slides should be submitted as a {facts['slide_format']} file. The suggested filename is {facts['slide_filename']}.",
            "evidence": [
                {
                    "source": "structured_facts.json",
                    "field": "slide_format",
                    "text": facts["slide_format"]
                },
                {
                    "source": "structured_facts.json",
                    "field": "slide_filename",
                    "text": facts["slide_filename"]
                }
            ],
            "confidence": "High",
            "route": "rule-based"
        }

    # Presentation duration
    if "presentation" in q and any(k in q for k in ["duration", "long", "time", "minutes"]):
        return {
            "answer": f"The presentation duration is {facts['presentation_duration']}.",
            "evidence": [
                {
                    "source": "structured_facts.json",
                    "field": "presentation_duration",
                    "text": facts["presentation_duration"]
                }
            ],
            "confidence": "High",
            "route": "rule-based"
        }

    # GitHub requirement
    # Important: use whole-word matching for repo to avoid matching "report".
    if "github" in q or "repository" in q or contains_whole_word(q, "repo"):
        if facts.get("github_required", False):
            answer = "Yes, a GitHub repository is required for the project prototype and evaluation files."
        else:
            answer = "The structured facts do not indicate that a GitHub repository is required."

        return {
            "answer": answer,
            "evidence": [
                {
                    "source": "structured_facts.json",
                    "field": "github_required",
                    "text": str(facts.get("github_required"))
                }
            ],
            "confidence": "High",
            "route": "rule-based"
        }

    # Individual contribution
    if "individual contribution" in q or "contribution" in q:
        return {
            "answer": "Yes, individual contributions must be clearly documented in the report.",
            "evidence": [
                {
                    "source": "structured_facts.json",
                    "field": "individual_contribution_required",
                    "text": str(facts.get("individual_contribution_required"))
                }
            ],
            "confidence": "High",
            "route": "rule-based"
        }

    return None
