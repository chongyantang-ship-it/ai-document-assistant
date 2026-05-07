RULE_KEYWORDS = [
    "due",
    "deadline",
    "word limit",
    "how many words",
    "duration",
    "minutes",
    "how long",
    "similarity",
    "turnitin",
    "plagiarism",
    "file format",
    "file type",
    "pdf",
    "pptx",
    "slides",
    "powerpoint",
    "github",
    "repository",
    "submit",
    "submission",
    "filename",
    "sections",
    "table of contents",
    "individual contribution",
    "first slide",
    "first presentation slide",
    "presentation slide",
    "first page",
    "student id",
    "group number",
    "group member",
    "font",
    "highlight",
    "rubric",
    "high distinction",
    "hd",
    "criterion",
    "ai methods",
    "system pipeline",
    "simple chatbot",
    "exact factual",
    "open-ended rubric",
    "open ended rubric",
    "baseline",
    "metrics",
    "evaluation",
    "evaluated",
    "evaluation questions",
    "unsupported handling",
]

UNSUPPORTED_PATTERNS = [
    "can i submit late without approval",
    "can we submit late without approval",
    "can i ignore",
    "can we ignore",
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


def is_rule_question(question):
    """Return True when the question is best served by structured facts and rules."""
    lower_question = question.lower()
    return any(keyword in lower_question for keyword in RULE_KEYWORDS)


def route_query(question):
    """
    Decide which component should answer the question.

    Routes:
    - rule: exact assessment constraints or rubric criteria stored in facts
    - rag: open-ended document interpretation
    - unsupported: questions that should not be answered confidently
    """
    if is_unsupported_question(question):
        return "unsupported"

    if is_rule_question(question):
        return "rule"

    return "rag"
