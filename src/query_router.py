EXACT_RULE_KEYWORDS = [
    "due",
    "deadline",
    "word limit",
    "how many words",
    "font",
    "duration",
    "minutes",
    "how long",
    "similarity",
    "turnitin",
    "plagiarism score",
    "file format",
    "file type",
    "pdf",
    "pptx",
    "filename",
    "report sections",
    "sections should be included in the report",
    "what sections should be included in the report",
    "first slide",
    "first presentation slide",
    "first page",
    "student id",
    "group number",
    "who needs to submit",
    "who should submit",
    "submit the group report",
    "submit the presentation",
    "group size",
    "maximum group size",
    "how many students per group",
]

OPEN_ENDED_RAG_KEYWORDS = [
    "rubric",
    "high distinction",
    "hd",
    "criterion",
    "criteria",
    "ai methods",
    "system pipeline",
    "workflow",
    "methodology",
    "evaluation",
    "baseline",
    "metric",
    "how should",
    "how do we",
    "why should",
    "why is",
    "plan",
    "roadmap",
    "timeline",
    "divide the work",
    "role allocation",
    "genai",
    "ethical",
    "ethics",
    "risk",
    "limitation",
    "story",
    "scenario",
]

UNSUPPORTED_PATTERNS = [
    "can i submit late without approval",
    "can we submit late without approval",
    "can i ignore",
    "can we ignore",
    "exact six-person role split",
    "exact six person role split",
    "exact role split",
    "exact role allocation",
    "exactly how should six group members divide the work",
    "day-by-day schedule",
    "day by day schedule",
    "guarantee a high distinction",
    "guarantee high distinction",
    "guarantee an hd",
    "guarantee hd",
    "write the whole assignment",
    "write my whole assignment",
    "cheat",
    "bypass turnitin",
    "replace tutors",
    "replace official course guidance",
    "replace official guidance",
    "ignore official instructions",
]


def is_unsupported_question(question):
    """Return True when the question requests unsafe or unsupported behaviour."""
    lower_question = question.lower()
    return any(pattern in lower_question for pattern in UNSUPPORTED_PATTERNS)


def is_open_ended_rag_question(question):
    """Return True when the question asks for interpretation, planning, or synthesis."""
    lower_question = question.lower()
    return any(keyword in lower_question for keyword in OPEN_ENDED_RAG_KEYWORDS)


def is_rule_question(question):
    """Return True when the question is best served by deterministic structured facts."""
    lower_question = question.lower()
    if is_open_ended_rag_question(question):
        return False
    return any(keyword in lower_question for keyword in EXACT_RULE_KEYWORDS)


def route_query(question):
    """
    Decide which component should answer the question.

    Routes:
    - rule: exact assessment constraints stored in facts
    - rag: open-ended interpretation, planning, and policy questions
    - unsupported: questions that should not be answered confidently
    """
    if is_unsupported_question(question):
        return "unsupported"

    if is_rule_question(question):
        return "rule"

    return "rag"
