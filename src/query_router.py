
def route_query(question):
    """
    Decide which component should answer the question.
    Routes:
    - rule: exact assessment constraints
    - rag: open-ended document interpretation
    - unsupported: questions that should not be answered confidently
    """
    q = question.lower()

    unsupported_patterns = [
        "can i submit late without approval",
        "can we submit late without approval",
        "can i ignore",
        "can we ignore",
        "write the whole assignment",
        "write my whole assignment",
        "cheat",
        "bypass turnitin"
    ]

    if any(pattern in q for pattern in unsupported_patterns):
        return "unsupported"

    rule_keywords = [
        "due",
        "deadline",
        "word limit",
        "how many words",
        "similarity",
        "turnitin",
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
        "sections",
        "table of contents",
        "individual contribution"
    ]

    if any(keyword in q for keyword in rule_keywords):
        return "rule"

    return "rag"
