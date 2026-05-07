
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FACTS_PATH = PROJECT_ROOT / "data" / "processed" / "structured_facts.json"


def load_facts(path=FACTS_PATH):
    """Load structured facts extracted from assignment documents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fact_value(fact):
    if isinstance(fact, dict):
        return str(fact.get("value", "")).strip()
    return str(fact).strip()


def _fact_evidence(fact):
    if isinstance(fact, dict):
        return str(fact.get("evidence", _fact_value(fact))).strip()
    return str(fact).strip()


def _answer(question, field, fact, answer_text, confidence="High"):
    return {
        "question": question,
        "route": "rule-based",
        "confidence": confidence,
        "answer": answer_text,
        "evidence": [
            {
                "source": "structured_facts.json",
                "field": field,
                "text": _fact_evidence(fact),
            }
        ],
    }


def _words(q):
    cleaned = (
        q.replace("/", " ")
        .replace("-", " ")
        .replace("_", " ")
        .replace("?", " ")
        .replace(".", " ")
        .replace(",", " ")
    )
    return cleaned.split()


def check_rule_based_answer(question, facts):
    """
    Deterministic rule-based answer checker.

    Specific rules are placed before broader rules.
    If no safe structured rule matches, return None so app.py can use RAG.
    """
    q = question.lower().strip()
    q_words = _words(q)

    # Group presentation dates
    if "group a" in q and "presentation" in q and any(x in q for x in ["due", "when", "date"]):
        fact = facts.get("presentation_due_dates", {}).get("group_a")
        if fact:
            return _answer(
                question,
                "presentation_due_dates.group_a",
                fact,
                "The Group A presentation is due on 7 May 2026 in Week 11."
            )

    if "group b" in q and "presentation" in q and any(x in q for x in ["due", "when", "date"]):
        fact = facts.get("presentation_due_dates", {}).get("group_b")
        if fact:
            return _answer(
                question,
                "presentation_due_dates.group_b",
                fact,
                "The Group B presentation is due on 14 May 2026 in Week 12."
            )

    # Report due date
    if "report" in q and any(x in q for x in ["due", "deadline", "submit date", "submission date", "when"]):
        fact = facts.get("report_due_date")
        if fact:
            return _answer(
                question,
                "report_due_date",
                fact,
                f"The report is due on {_fact_value(fact)}."
            )

    # Word limit
    if "word" in q and "limit" in q:
        fact = facts.get("word_limit")
        if fact:
            return _answer(
                question,
                "word_limit",
                fact,
                f"The report word limit is {_fact_value(fact)}."
            )

    # Similarity / Turnitin
    if "similarity" in q or "turnitin" in q:
        fact = facts.get("similarity_limit")
        if fact:
            return _answer(
                question,
                "similarity_limit",
                fact,
                f"The maximum allowed similarity score is {_fact_value(fact)}."
            )

    # Report font
    if "font" in q and "report" in q:
        fact = facts.get("report_font")
        if fact:
            return _answer(
                question,
                "report_font",
                fact,
                f"The report should use {_fact_value(fact)} fonts."
            )

    # Report format
    if "report" in q and "format" in q:
        fact = facts.get("report_format")
        if fact:
            return _answer(
                question,
                "report_format",
                fact,
                f"The report should be submitted in {_fact_value(fact)} format."
            )

    # Slide / presentation format
    if ("slide" in q or "presentation" in q) and "format" in q:
        fact = facts.get("slide_format")
        if fact:
            return _answer(
                question,
                "slide_format",
                fact,
                f"The presentation slides should follow this format requirement: {_fact_value(fact)}."
            )

    # Presentation duration
    if "presentation" in q and any(x in q for x in ["long", "duration", "time", "minutes"]):
        fact = facts.get("presentation_duration")
        if fact:
            return _answer(
                question,
                "presentation_duration",
                fact,
                f"The presentation duration is {_fact_value(fact)}."
            )

    # Report filename
    if "report" in q and any(x in q for x in ["filename", "file name", "name should", "named"]):
        fact = facts.get("report_filename")
        if fact:
            return _answer(
                question,
                "report_filename",
                fact,
                "The suggested report filename is AT3_Report_Group_xxxx.pdf."
            )

    # Slide filename
    if ("slide" in q or "presentation" in q) and any(x in q for x in ["filename", "file name", "name should", "named"]):
        fact = facts.get("slide_filename")
        if fact:
            return _answer(
                question,
                "slide_filename",
                fact,
                "The suggested slide filename is AT3_Slide_Group_xxxx.pptx."
            )

    # First presentation slide
    if "first" in q and ("presentation" in q or "slide" in q) and any(x in q for x in ["include", "included", "contain"]):
        fact = facts.get("first_presentation_slide")
        if fact:
            return _answer(
                question,
                "first_presentation_slide",
                fact,
                "The first presentation slide should include the group number, names, and student IDs."
            )

    # First page of the report
    if "first" in q and "page" in q and "report" in q:
        fact = facts.get("first_page_requirements")
        if fact:
            return _answer(
                question,
                "first_page_requirements",
                fact,
                "The first page of the report should include all group members' names and student IDs."
            )

    # Group submission
    if "who" in q and "submit" in q and ("group report" in q or "presentation" in q):
        fact = facts.get("group_submission")
        if fact:
            return _answer(
                question,
                "group_submission",
                fact,
                _fact_value(fact)
            )

    # HD individual contribution rule must come before general individual contribution
    if "hd" in q and "individual" in q and "contribution" in q:
        fact = facts.get("hd_individual_contribution")
        if fact:
            return _answer(
                question,
                "hd_individual_contribution",
                fact,
                _fact_value(fact)
            )

    # Individual contributions
    if "individual" in q and "contribution" in q:
        fact = facts.get("individual_contributions")
        if fact:
            return _answer(
                question,
                "individual_contributions",
                fact,
                "Yes. Individual contributions need to be documented."
            )

    # Report sections
    if "report" in q and any(x in q for x in ["section", "sections"]):
        fact = facts.get("report_sections")
        if fact:
            return _answer(
                question,
                "report_sections",
                fact,
                f"The report should include {_fact_value(fact)}."
            )

    # System pipeline
    if "pipeline" in q or ("system" in q and "workflow" in q):
        fact = facts.get("system_pipeline")
        if fact:
            return _answer(
                question,
                "system_pipeline",
                fact,
                f"The system pipeline includes {_fact_value(fact)}."
            )

    # Evaluation metrics — before general evaluation summary
    if "metric" in q and "evaluation" in q:
        fact = facts.get("evaluation_metrics")
        if fact:
            return _answer(
                question,
                "evaluation_metrics",
                fact,
                f"The evaluation should report {_fact_value(fact)}."
            )

    # Evaluation categories — before general evaluation summary
    if "categor" in q and "evaluation" in q:
        fact = facts.get("evaluation_categories")
        if fact:
            return _answer(
                question,
                "evaluation_categories",
                fact,
                f"The evaluation questions should cover {_fact_value(fact)}."
            )

    # Why unsupported questions are included
    if "unsupported" in q and ("why" in q or "include" in q or "included" in q):
        fact = facts.get("unsupported_questions_reason")
        if fact:
            return _answer(
                question,
                "unsupported_questions_reason",
                fact,
                _fact_value(fact)
            )

    # General evaluation design
    if "evaluated" in q or "evaluate" in q or "evaluation" in q:
        fact = facts.get("evaluation_summary")
        if fact:
            return _answer(
                question,
                "evaluation_summary",
                fact,
                f"The system should be evaluated using {_fact_value(fact)}."
            )

    # GitHub / repository requirement
    # Important: "repo" must be matched as a standalone word, not inside "report".
    if "github" in q or "repository" in q or "repo" in q_words:
        fact = facts.get("github_required")
        if fact:
            return _answer(
                question,
                "github_required",
                fact,
                _fact_value(fact)
            )

    # AI methods used
    if "ai method" in q or "ai methods" in q or "methods are used" in q:
        fact = facts.get("required_ai_methods")
        if fact:
            return _answer(
                question,
                "required_ai_methods",
                fact,
                f"The assistant may combine {_fact_value(fact)}."
            )

    # Ethical / official guidance limitation
    if "replace official" in q or "official course guidance" in q:
        fact = facts.get("official_guidance_warning")
        if fact:
            return _answer(
                question,
                "official_guidance_warning",
                fact,
                _fact_value(fact),
                confidence="Medium"
            )

    return None
