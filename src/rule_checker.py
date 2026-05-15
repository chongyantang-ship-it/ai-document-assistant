import json
import re
from pathlib import Path

from fact_extractor import save_structured_facts


RUBRIC_SECTION_KEYWORDS = {
    "executive_summary_and_introduction": [
        "executive summary",
        "introduction",
        "problem definition",
        "significance",
    ],
    "theoretical_justification": [
        "theoretical justification",
        "literature",
        "theory",
        "method justification",
    ],
    "workflow_and_methodology": [
        "workflow",
        "methodology",
        "model design",
        "algorithmic approach",
        "design decisions",
    ],
    "empirical_analysis_and_results": [
        "empirical",
        "results",
        "experimental",
        "experiment",
        "evaluation metrics",
    ],
    "critical_reflection": [
        "critical reflection",
        "limitations",
        "trade-offs",
        "scalability",
        "failure case",
    ],
    "conclusion_future_work_and_references": [
        "conclusion",
        "future work",
        "references",
    ],
    "individual_contribution": [
        "individual contribution",
        "contributions",
    ],
}

PRESENTATION_RUBRIC_SECTION_KEYWORDS = {
    "content_and_relevance": [
        "content and relevance",
        "relevance",
        "real-world problem",
    ],
    "ai_method_explanation_and_justification": [
        "ai method",
        "method explanation",
        "justification",
        "trade-offs",
    ],
    "findings_contributions_and_limitations": [
        "findings",
        "contributions",
        "limitations",
    ],
    "delivery_and_interaction": [
        "delivery",
        "interaction",
        "questions",
        "visuals",
    ],
}


def load_facts(facts_path=None):
    """Load the structured knowledge base from JSON."""
    if facts_path is None:
        facts_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "structured_facts.json"
    else:
        facts_path = Path(facts_path)

    if not facts_path.exists():
        save_structured_facts(facts_path)

    with open(facts_path, "r", encoding="utf-8") as facts_file:
        return json.load(facts_file)


def contains_whole_word(text, word):
    """Return True when a token appears as a standalone word."""
    pattern = r"\b" + re.escape(word) + r"\b"
    return re.search(pattern, text) is not None


def build_rule_answer(answer, field, text):
    """Create a standard response payload for rule-based answers."""
    return {
        "answer": answer,
        "evidence": [
            {
                "source": "structured_facts.json",
                "field": field,
                "text": text,
            }
        ],
        "confidence": "High",
        "route": "rule-based",
    }


def build_multi_evidence_rule_answer(answer, evidence_items):
    """Create a rule-based answer backed by several structured fields."""
    return {
        "answer": answer,
        "evidence": [
            {
                "source": "structured_facts.json",
                "field": field,
                "text": text,
            }
            for field, text in evidence_items
            if text
        ],
        "confidence": "High",
        "route": "rule-based",
    }


def find_matching_rubric_section(question, section_keywords):
    """Return the rubric section key that best matches a question."""
    lower_question = question.lower()
    for section_key, keywords in section_keywords.items():
        if any(keyword in lower_question for keyword in keywords):
            return section_key
    return None


def answer_rubric_question(question, facts):
    """Return a structured rubric answer when the section can be identified."""
    lower_question = question.lower()
    asks_for_hd = any(token in lower_question for token in ["hd", "high distinction", "criterion", "criteria"])
    asks_for_hd = asks_for_hd or (
        "rubric" in lower_question and any(token in lower_question for token in ["require", "requires"])
    )
    if not asks_for_hd:
        return None

    report_requirements = facts.get("rubric_hd_requirements", {})
    presentation_requirements = facts.get("presentation_rubric_hd_requirements", {})

    report_section = find_matching_rubric_section(question, RUBRIC_SECTION_KEYWORDS)
    if report_section is not None and report_section in report_requirements:
        answer = report_requirements[report_section]
        return build_rule_answer(answer, f"rubric_hd_requirements.{report_section}", answer)

    presentation_section = find_matching_rubric_section(question, PRESENTATION_RUBRIC_SECTION_KEYWORDS)
    if presentation_section is not None and presentation_section in presentation_requirements:
        answer = presentation_requirements[presentation_section]
        return build_rule_answer(
            answer,
            f"presentation_rubric_hd_requirements.{presentation_section}",
            answer,
        )

    if not report_requirements and not presentation_requirements:
        return None

    summary = (
        "Across the report, HD requires strong problem definition, rigorous theoretical justification, "
        "a clear methodology, comprehensive experiments, deep critical reflection, and realistic future work."
    )
    return build_rule_answer(summary, "rubric_hd_requirements", summary)


def answer_report_structure_question(question, facts):
    """Answer exact questions about report sections and formatting requirements."""
    lower_question = question.lower()
    report_word_limit = facts.get("report_word_limit", "")
    report_font = facts.get("report_font", "")
    report_sections = facts.get("report_sections", [])

    if (
        report_word_limit
        and report_font
        and "font" in lower_question
        and any(keyword in lower_question for keyword in ["word limit", "how many words", "maximum words"])
    ):
        answer = (
            f"The report word limit is {report_word_limit}. "
            f"The report should use {report_font}."
        )
        return build_multi_evidence_rule_answer(
            answer,
            [("report_word_limit", report_word_limit), ("report_font", report_font)],
        )

    if report_word_limit and any(keyword in lower_question for keyword in ["word limit", "how many words", "maximum words"]):
        return build_rule_answer(
            f"The report word limit is {report_word_limit}.",
            "report_word_limit",
            report_word_limit,
        )

    if report_sections and any(keyword in lower_question for keyword in ["report section", "sections", "include in the report", "table of contents"]):
        sections = report_sections
        return build_rule_answer(
            "The report should include: " + ", ".join(sections) + ".",
            "report_sections",
            ", ".join(sections),
        )

    if report_font and "font" in lower_question:
        return build_rule_answer(
            f"The report should use {report_font}.",
            "report_font",
            report_font,
        )

    return None


def answer_deadline_question(question, facts):
    """Answer exact questions about due dates and presentation timing."""
    lower_question = question.lower()
    presentation_due_dates = facts.get("presentation_due_dates", {})
    report_due_date = facts.get("report_due_date", "")
    presentation_duration = facts.get("presentation_duration", "")

    if "group a" in lower_question and "presentation" in lower_question and "due" in lower_question and presentation_due_dates.get("group_a"):
        due_date = presentation_due_dates["group_a"]
        return build_rule_answer(
            f"The Group A presentation is due on {due_date}.",
            "presentation_due_dates.group_a",
            due_date,
        )

    if "group b" in lower_question and "presentation" in lower_question and "due" in lower_question and presentation_due_dates.get("group_b"):
        due_date = presentation_due_dates["group_b"]
        return build_rule_answer(
            f"The Group B presentation is due on {due_date}.",
            "presentation_due_dates.group_b",
            due_date,
        )

    if report_due_date and any(keyword in lower_question for keyword in ["due", "deadline", "when is the report", "report due"]):
        return build_rule_answer(
            f"The report is due on {report_due_date}.",
            "report_due_date",
            report_due_date,
        )

    if presentation_duration and "presentation" in lower_question and any(keyword in lower_question for keyword in ["duration", "long", "time", "minutes"]):
        return build_rule_answer(
            f"The presentation duration is {presentation_duration}.",
            "presentation_duration",
            presentation_duration,
        )

    return None


def answer_submission_question(question, facts):
    """Answer exact questions about file formats, filenames, and submission rules."""
    lower_question = question.lower()
    first_slide_requirements = facts.get("first_slide_requirements", [])
    first_page_requirements = facts.get("first_page_requirements", [])
    group_submission = facts.get("group_submission", "")
    presentation_highlights = facts.get("presentation_highlights", [])
    report_filename = facts.get("report_filename", "")
    slide_filename = facts.get("slide_filename", "")
    report_format = facts.get("report_format", "")
    slide_format = facts.get("slide_format", "")

    if (
        "first slide" in lower_question
        or "first presentation slide" in lower_question
        or ("presentation slide" in lower_question and "include" in lower_question)
    ) and first_slide_requirements:
        items = ", ".join(first_slide_requirements)
        return build_rule_answer(
            "The first presentation slide should include: " + items + ".",
            "first_slide_requirements",
            items,
        )

    if ("first page" in lower_question or ("report" in lower_question and "student id" in lower_question)) and first_page_requirements:
        items = ", ".join(first_page_requirements)
        return build_rule_answer(
            "The first page of the report should include: " + items + ".",
            "first_page_requirements",
            items,
        )

    if group_submission and "who" in lower_question and "submit" in lower_question:
        return build_rule_answer(
            group_submission + ".",
            "group_submission",
            group_submission,
        )

    if presentation_highlights and "presentation" in lower_question and any(keyword in lower_question for keyword in ["highlight", "include", "should show"]):
        items = ", ".join(presentation_highlights)
        return build_rule_answer(
            "The presentation should highlight: " + items + ".",
            "presentation_highlights",
            items,
        )

    if report_filename and "report" in lower_question and "filename" in lower_question:
        return build_rule_answer(
            f"The suggested report filename is {report_filename}.",
            "report_filename",
            report_filename,
        )

    if slide_filename and "filename" in lower_question and any(keyword in lower_question for keyword in ["slide", "slides", "pptx", "powerpoint"]):
        return build_rule_answer(
            f"The suggested slide filename is {slide_filename}.",
            "slide_filename",
            slide_filename,
        )

    if report_format and report_filename and "report" in lower_question and any(keyword in lower_question for keyword in ["format", "file type", "submit", "submission", "pdf"]):
        answer = (
            f"The report should be submitted as a {report_format} file. "
            f"The suggested filename is {report_filename}."
        )
        return build_multi_evidence_rule_answer(
            answer,
            [("report_format", report_format), ("report_filename", report_filename)],
        )

    if slide_format and slide_filename and any(keyword in lower_question for keyword in ["slide", "slides", "powerpoint", "ppt", "pptx"]):
        answer = (
            f"The presentation slides should be submitted as a {slide_format} file. "
            f"The suggested filename is {slide_filename}."
        )
        return build_multi_evidence_rule_answer(
            answer,
            [("slide_format", slide_format), ("slide_filename", slide_filename)],
        )

    return None


def answer_policy_question(question, facts):
    """Answer exact policy questions about similarity, GitHub, and contributions."""
    lower_question = question.lower()
    similarity_limit = facts.get("similarity_limit", "")
    maximum_group_size = facts.get("maximum_group_size", "")

    if similarity_limit and any(keyword in lower_question for keyword in ["similarity", "turnitin", "plagiarism"]):
        return build_rule_answer(
            f"The maximum allowed similarity score is {similarity_limit}.",
            "similarity_limit",
            similarity_limit,
        )

    if maximum_group_size and any(keyword in lower_question for keyword in ["group size", "maximum group size", "how many students per group"]):
        return build_rule_answer(
            f"The brief states a maximum group size of {maximum_group_size}.",
            "maximum_group_size",
            maximum_group_size,
        )

    if "github" in lower_question or "repository" in lower_question or contains_whole_word(lower_question, "repo"):
        if facts.get("github_required", False):
            answer = "Yes, the brief indicates that a GitHub repository is required."
        else:
            answer = "The structured facts do not indicate that a GitHub repository is required."
        return build_rule_answer(answer, "github_required", str(facts.get("github_required")))

    if "individual contribution" in lower_question or "contribution" in lower_question:
        if not facts.get("individual_contribution_required"):
            return None
        return build_rule_answer(
            "Yes, individual contributions must be clearly documented in the report.",
            "individual_contribution_required",
            str(facts["individual_contribution_required"]),
        )

    return None


def answer_system_design_question(question, facts):
    """Answer exact questions about the prototype architecture and reasoning flow."""
    lower_question = question.lower()

    if ("what ai methods" in lower_question or ("ai method" in lower_question and "used" in lower_question)) and facts.get("required_ai_methods"):
        methods = ", ".join(facts["required_ai_methods"])
        return build_rule_answer(
            "The project uses: " + methods + ".",
            "required_ai_methods",
            methods,
        )

    if ("system pipeline" in lower_question or "pipeline" in lower_question) and facts.get("system_pipeline_steps"):
        steps = ", ".join(facts["system_pipeline_steps"])
        return build_rule_answer(
            "The system pipeline includes: " + steps + ".",
            "system_pipeline_steps",
            steps,
        )

    if "simple chatbot" in lower_question and facts.get("system_positioning"):
        return build_rule_answer(
            facts["system_positioning"],
            "system_positioning",
            facts["system_positioning"],
        )

    if "exact factual" in lower_question and facts.get("exact_question_strategy"):
        return build_rule_answer(
            facts["exact_question_strategy"],
            "exact_question_strategy",
            facts["exact_question_strategy"],
        )

    if ("open-ended rubric" in lower_question or "open ended rubric" in lower_question) and facts.get("open_ended_question_strategy"):
        return build_rule_answer(
            facts["open_ended_question_strategy"],
            "open_ended_question_strategy",
            facts["open_ended_question_strategy"],
        )

    return None


def answer_evaluation_design_question(question, facts):
    """Answer exact questions about the evaluation design used in the report."""
    lower_question = question.lower()

    if "how should the system be evaluated" in lower_question and facts.get("evaluation_summary"):
        return build_rule_answer(
            facts["evaluation_summary"],
            "evaluation_summary",
            facts["evaluation_summary"],
        )

    if "baseline" in lower_question and facts.get("evaluation_baselines"):
        baselines = ", ".join(facts["evaluation_baselines"])
        return build_rule_answer(
            "Appropriate baseline methods are: " + baselines + ".",
            "evaluation_baselines",
            baselines,
        )

    if "metric" in lower_question and facts.get("evaluation_metrics"):
        metrics = ", ".join(facts["evaluation_metrics"])
        return build_rule_answer(
            "The evaluation should report: " + metrics + ".",
            "evaluation_metrics",
            metrics,
        )

    if (
        ("evaluation" in lower_question and "category" in lower_question)
        or ("evaluation" in lower_question and "cover" in lower_question)
        or ("question" in lower_question and "cover" in lower_question)
    ) and facts.get("evaluation_question_categories"):
        categories = ", ".join(facts["evaluation_question_categories"])
        return build_rule_answer(
            "The evaluation questions should cover: " + categories + ".",
            "evaluation_question_categories",
            categories,
        )

    if "unsupported" in lower_question and "evaluation" in lower_question and facts.get("unsupported_question_rationale"):
        return build_rule_answer(
            facts["unsupported_question_rationale"],
            "unsupported_question_rationale",
            facts["unsupported_question_rationale"],
        )

    return None


def check_rule_based_answer(question, facts):
    """
    Return a rule-based answer for exact assessment constraints.

    If the question is not fully covered by rules, return None so the caller can
    fall back to retrieval and evidence-grounded generation.
    """
    for resolver in [
        answer_report_structure_question,
        answer_deadline_question,
        answer_submission_question,
        answer_policy_question,
    ]:
        result = resolver(question, facts)
        if result is not None:
            return result

    return None
