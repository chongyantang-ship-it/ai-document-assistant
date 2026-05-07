import re


SECTION_HEADING_PATTERNS = [
    re.compile(r"^[A-Z][A-Za-z0-9 ,/&()'.-]{2,80}:$"),
    re.compile(r"^\d+(\.\d+)*\s+[A-Z].+"),
    re.compile(r"^Report Rubric: .+"),
    re.compile(r"^Presentation Rubric: .+"),
]


def slugify(text):
    """Convert a label into a stable slug."""
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


def split_into_sections(text):
    """Split a raw document into sections keyed by detected headings."""
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
