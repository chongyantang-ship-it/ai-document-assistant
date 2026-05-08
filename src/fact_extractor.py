import json
import os
import re
from pathlib import Path

from answer_generator import (
    _call_chat_completion,
    _call_responses_api,
    _create_llm_client,
    _get_llm_settings,
)
from brief_structure import slugify, split_into_sections
from document_loader import read_document_text
from runtime_config import get_active_brief_path


TRUTHY_VALUES = {"1", "true", "yes", "on"}
FACT_EXTRACTION_MODEL_ENV = "FACT_EXTRACTION_MODEL"
FACT_EXTRACTION_USE_LLM_ENV = "FACT_EXTRACTION_USE_LLM"
FACT_SCHEMA_VERSION = "2026-05-general-single-brief-v2"
MISSING_VALUES = (None, "", [], {})

GENERAL_FACT_FIELDS = [
    "assessment_name",
    "course",
    "report_due_date",
    "report_word_limit",
    "report_font",
    "report_format",
    "report_filename",
    "slide_format",
    "slide_filename",
    "presentation_duration",
    "presentation_submission",
    "presentation_due_dates",
    "similarity_limit",
    "group_submission",
    "first_page_requirements",
    "first_slide_requirements",
    "presentation_highlights",
    "report_sections",
    "github_required",
    "individual_contribution_required",
    "maximum_group_size",
    "extension_policy",
    "academic_integrity_guidance",
    "rubric_sections",
]


def _normalize_whitespace(text):
    """Collapse whitespace so evidence snippets are compact and consistent."""
    return " ".join(str(text or "").split())


def _split_sentences(text):
    """Split free text into short sentence-like units for evidence lookup."""
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", str(text or "")) if segment.strip()]


def _build_section_lookup(section_pairs):
    """Map lowercase section titles to their body text."""
    return {title.lower(): body for title, body in section_pairs}


def _llm_fact_extraction_enabled():
    """Return True when the live extraction mode is explicitly enabled."""
    return os.getenv(FACT_EXTRACTION_USE_LLM_ENV, "0").strip().lower() in TRUTHY_VALUES


def _find_section(section_pairs, keywords):
    """Return the first section title and body whose heading contains a keyword."""
    for section_title, section_body in section_pairs:
        lower_title = section_title.lower()
        if any(keyword in lower_title for keyword in keywords):
            return section_title, section_body
    return "", ""


def _find_section_text(section_lookup, keywords):
    """Return the first section body whose title contains any keyword."""
    for section_title, section_body in section_lookup.items():
        if any(keyword in section_title for keyword in keywords):
            return section_body
    return ""


def _first_non_empty_line(text):
    """Return the first non-empty line from the brief."""
    for line in str(text or "").splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


def _first_matching_sentence(text, required_tokens):
    """Return the first sentence that contains every required token."""
    for sentence in _split_sentences(text):
        lower_sentence = sentence.lower()
        if all(token in lower_sentence for token in required_tokens):
            return sentence
    return ""


def _first_sentence_matching_any(text, keyword_sets):
    """Return the first sentence matching one of several token combinations."""
    for required_tokens in keyword_sets:
        sentence = _first_matching_sentence(text, required_tokens)
        if sentence:
            return sentence
    return ""


def _extract_first_match(text, patterns):
    """Return the first regex group match across several patterns."""
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return " ".join(part.strip() for part in match.groups() if part and part.strip()).strip()
    return ""


def _extract_first_match_with_sentence(text, patterns):
    """Return a regex match together with the sentence that supports it."""
    sentences = _split_sentences(text)
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue

        value = " ".join(part.strip() for part in match.groups() if part and part.strip()).strip()
        matched_text = match.group(0).strip()
        supporting_sentence = ""
        for sentence in sentences:
            if matched_text.lower() in sentence.lower():
                supporting_sentence = sentence
                break
        return value, supporting_sentence
    return "", ""


def _extract_date_phrase(sentence):
    """Extract a date phrase, plus a time if available, from a sentence."""
    if not sentence:
        return ""

    date_match = re.search(
        r"(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+20\d{2})",
        sentence,
        flags=re.IGNORECASE,
    )
    if not date_match:
        return ""

    time_match = re.search(r"(\d{1,2}:\d{2})", sentence)
    if time_match:
        return f"{date_match.group(1)} {time_match.group(1)}"

    return date_match.group(1)


def _extract_report_due_date(full_text, section_pairs):
    """Extract the main report due date and its evidence sentence."""
    best_without_time = ("", "", "")
    candidate_sections = [
        _find_section(section_pairs, ["submission due", "deadline"]),
        _find_section(section_pairs, ["overview"]),
    ]
    candidate_sections.append(("Full Document", full_text))

    for section_title, section_text in candidate_sections:
        for sentence in _split_sentences(section_text):
            lower_sentence = sentence.lower()
            if "report" not in lower_sentence or "due" not in lower_sentence:
                continue

            extracted = _extract_date_phrase(sentence)
            if not extracted:
                continue

            if re.search(r"\d{1,2}:\d{2}", sentence):
                return extracted, sentence, section_title
            if not best_without_time[0]:
                best_without_time = (extracted, sentence, section_title)

    return best_without_time


def _extract_presentation_due_dates(full_text, section_pairs):
    """Extract separate presentation dates for Group A and Group B when present."""
    candidate_blocks = []
    for keywords in [["submission due"], ["presentation sessions"], ["presentation session"]]:
        section_title, section_body = _find_section(section_pairs, keywords)
        if section_body:
            candidate_blocks.append((section_title, section_body))
    candidate_blocks.append(("Full Document", full_text))

    due_dates = {}
    evidence_fragments = []
    source_section = ""

    for section_title, candidate_text in candidate_blocks:
        for group_label in ["group a", "group b"]:
            match = re.search(
                rf"{group_label}[^.\n]*?(?:on|by)\s+(\d{{1,2}}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+20\d{{2}})",
                candidate_text,
                flags=re.IGNORECASE,
            )
            if not match:
                continue

            key = group_label.replace(" ", "_")
            if key in due_dates:
                continue

            due_dates[key] = match.group(1)
            source_section = source_section or section_title
            for sentence in _split_sentences(candidate_text):
                if group_label in sentence.lower() and match.group(1).lower() in sentence.lower():
                    evidence_fragments.append(sentence)
                    break

    return due_dates, " | ".join(evidence_fragments), source_section


def _extract_list_from_sentence(sentence):
    """Parse a short comma-separated requirement list from a sentence."""
    if not sentence:
        return []

    lower_sentence = sentence.lower()
    start_index = -1
    for marker in [
        "include",
        "list",
        "highlight",
        "state that",
        "cc field to confirm their agreement",
    ]:
        marker_index = lower_sentence.find(marker)
        if marker_index >= 0:
            start_index = marker_index + len(marker)
            break

    candidate_text = sentence[start_index:] if start_index >= 0 else sentence
    candidate_text = candidate_text.strip(" :.")
    candidate_text = candidate_text.replace(" and ", ", ")
    return [item.strip(" .") for item in candidate_text.split(",") if item.strip(" .")]


def _extract_report_sections(report_sections_text):
    """Extract the named report sections from the brief."""
    if not report_sections_text:
        return []

    section_names = re.findall(r"([A-Z][A-Za-z ]{2,80})\.", report_sections_text)
    allowed_section_keywords = {
        "summary",
        "introduction",
        "justification",
        "workflow",
        "methodology",
        "analysis",
        "results",
        "reflection",
        "conclusion",
        "future work",
        "references",
        "contribution",
        "contributions",
    }

    cleaned_sections = []
    for section_name in section_names:
        cleaned = section_name.strip()
        lowered = cleaned.lower()
        if (
            len(cleaned.split()) <= 8
            and lowered not in {"additional expectations"}
            and any(keyword in lowered for keyword in allowed_section_keywords)
        ):
            cleaned_sections.append(cleaned)

    deduplicated_sections = []
    seen = set()
    for section_name in cleaned_sections:
        lowered = section_name.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduplicated_sections.append(section_name)

    return deduplicated_sections


def _extract_hd_requirement(section_body):
    """Extract the HD line from a rubric section body."""
    match = re.search(r"HD:\s*(.+?)(?:\n[DCPF]:|$)", section_body, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return " ".join(match.group(1).split())


def _extract_rubric_facts(section_pairs):
    """Build generic rubric metadata from the active brief."""
    rubric_sections = []
    report_rubric_hd_requirements = {}
    presentation_rubric_hd_requirements = {}

    for title, body in section_pairs:
        lower_title = title.lower()
        if not ("report rubric:" in lower_title or "presentation rubric:" in lower_title):
            continue

        hd_requirement = _extract_hd_requirement(body)
        if not hd_requirement:
            continue

        scope = "presentation" if "presentation rubric:" in lower_title else "report"
        section_name = title.split(":", 1)[1].strip() if ":" in title else title
        section_slug = slugify(section_name)
        rubric_sections.append({
            "scope": scope,
            "title": section_name,
            "slug": section_slug,
            "hd_requirement": hd_requirement,
        })

        if scope == "report":
            report_rubric_hd_requirements[section_slug] = hd_requirement
        else:
            presentation_rubric_hd_requirements[section_slug] = hd_requirement

    return rubric_sections, report_rubric_hd_requirements, presentation_rubric_hd_requirements


def _extract_presentation_highlights(presentation_text):
    """Extract the key points the presentation should highlight."""
    match = re.search(
        r"should highlight\s+(.+?)(?:\.\s|$)",
        presentation_text or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []

    candidate_text = match.group(1).replace(" and ", ", ")
    return [item.strip(" .") for item in candidate_text.split(",") if item.strip(" .")]


def _extract_group_submission_sentence(full_text):
    """Build a concise summary of who submits the required files."""
    report_sentence = ""
    presentation_sentence = ""

    for sentence in _split_sentences(full_text):
        lower_sentence = sentence.lower()
        if "only one member" not in lower_sentence:
            continue
        if "report" in lower_sentence and not report_sentence:
            report_sentence = sentence
        if ("presentation" in lower_sentence or "powerpoint" in lower_sentence) and not presentation_sentence:
            presentation_sentence = sentence

    if report_sentence and presentation_sentence:
        combined = "Only one member from the group needs to submit the report and presentation files."
        evidence = f"{report_sentence} | {presentation_sentence}"
        return combined, evidence
    if report_sentence:
        return report_sentence.rstrip(".") + ".", report_sentence
    if presentation_sentence:
        return presentation_sentence.rstrip(".") + ".", presentation_sentence
    return "", ""


def _normalize_bool(value):
    """Convert a loose boolean-like value into True, False, or None."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None

    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "required", "1"}:
        return True
    if normalized in {"false", "no", "not required", "0"}:
        return False
    return None


def _normalize_list(value):
    """Convert a value into a cleaned list representation."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    candidate_text = str(value).strip()
    if not candidate_text:
        return []
    candidate_text = candidate_text.replace(" and ", ", ")
    return [item.strip(" .") for item in candidate_text.split(",") if item.strip(" .")]


def _normalize_rubric_sections(value):
    """Normalize rubric section records into a consistent list of dictionaries."""
    normalized_sections = []
    if not isinstance(value, list):
        return normalized_sections

    for item in value:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        scope = str(item.get("scope", "")).strip().lower()
        hd_requirement = _normalize_whitespace(item.get("hd_requirement", ""))
        if not title or scope not in {"report", "presentation"}:
            continue
        normalized_sections.append({
            "scope": scope,
            "title": title,
            "slug": slugify(title),
            "hd_requirement": hd_requirement,
        })

    return normalized_sections


def _normalize_field_value(field_name, value):
    """Normalize a field value based on the general fact schema."""
    if field_name in {"github_required", "individual_contribution_required"}:
        return _normalize_bool(value)
    if field_name in {
        "first_page_requirements",
        "first_slide_requirements",
        "presentation_highlights",
        "report_sections",
    }:
        return _normalize_list(value)
    if field_name == "presentation_due_dates":
        if isinstance(value, dict):
            return {
                str(key).strip().lower(): str(item).strip()
                for key, item in value.items()
                if str(item).strip()
            }
        return {}
    if field_name == "rubric_sections":
        return _normalize_rubric_sections(value)
    if value is None:
        return None
    return _normalize_whitespace(value)


def _extract_first_int(text):
    """Extract the first integer value from text when present."""
    match = re.search(r"(\d+)", str(text or ""))
    return int(match.group(1)) if match else None


def _extract_percentage(text):
    """Extract the first percentage value from text when present."""
    match = re.search(r"(\d+)\s*%", str(text or ""))
    return int(match.group(1)) if match else None


def _extract_due_date_parts(text):
    """Build a small normalized date payload from a due-date string."""
    if not text:
        return None
    date_match = re.search(r"(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+20\d{2})", text, flags=re.IGNORECASE)
    time_match = re.search(r"(\d{1,2}:\d{2})", text)
    if not date_match and not time_match:
        return None
    payload = {}
    if date_match:
        payload["date"] = date_match.group(1)
    if time_match:
        payload["time"] = time_match.group(1)
    return payload


def _normalize_machine_value(field_name, normalized_value):
    """Create a machine-friendly normalized value for reportable extraction outputs."""
    if normalized_value in MISSING_VALUES:
        return None

    if field_name == "report_due_date":
        return _extract_due_date_parts(normalized_value)
    if field_name == "report_word_limit":
        return {
            "max_words": _extract_first_int(normalized_value),
            "includes_references": "reference" in str(normalized_value).lower(),
        }
    if field_name == "report_font":
        allowed_fonts = []
        lower_value = str(normalized_value).lower()
        if "times" in lower_value:
            allowed_fonts.append("Times")
        if "arial" in lower_value:
            allowed_fonts.append("Arial")
        return {
            "font_size": _extract_first_int(normalized_value),
            "allowed_fonts": allowed_fonts,
        }
    if field_name == "presentation_duration":
        return {
            "total_minutes": _extract_first_int(normalized_value),
            "presentation_minutes": _extract_first_int(
                _extract_first_match(str(normalized_value), [r"(\d+)-minute presentation"])
            ),
            "qa_minutes": _extract_first_int(
                _extract_first_match(str(normalized_value), [r"(\d+)-minute q&a"])
            ),
        }
    if field_name == "similarity_limit":
        return {"max_percent": _extract_percentage(normalized_value)}
    if field_name == "maximum_group_size":
        return {"max_students": _extract_first_int(normalized_value)}
    if field_name in {"report_format", "slide_format"}:
        return str(normalized_value).upper()
    return normalized_value


def _is_valid_field_value(field_name, value):
    """Return True when a normalized value looks plausible for its field."""
    if value in MISSING_VALUES:
        return False

    if field_name in {"assessment_name", "course", "extension_policy", "academic_integrity_guidance"}:
        return len(str(value)) >= 5
    if field_name in {"report_due_date", "presentation_duration"}:
        return bool(re.search(r"\d", str(value)))
    if field_name == "report_word_limit":
        return "word" in str(value).lower() and bool(re.search(r"\d", str(value)))
    if field_name == "report_font":
        return any(token in str(value).lower() for token in ["font", "times", "arial"])
    if field_name in {"report_format", "slide_format"}:
        return str(value).upper() in {"PDF", "PPTX", "DOCX", "ZIP"}
    if field_name in {"report_filename", "slide_filename"}:
        return "." in str(value)
    if field_name == "similarity_limit":
        return "%" in str(value) or bool(re.search(r"\d", str(value)))
    if field_name == "maximum_group_size":
        return "student" in str(value).lower() or bool(re.search(r"\d", str(value)))
    if field_name in {"group_submission", "presentation_submission"}:
        return len(str(value)) >= 8
    if field_name in {"github_required", "individual_contribution_required"}:
        return isinstance(value, bool)
    if field_name in {
        "first_page_requirements",
        "first_slide_requirements",
        "presentation_highlights",
        "report_sections",
    }:
        return isinstance(value, list) and len(value) > 0
    if field_name == "presentation_due_dates":
        return isinstance(value, dict) and len(value) > 0
    if field_name == "rubric_sections":
        return isinstance(value, list) and len(value) > 0

    return True


def _build_field_record(field_name, value, evidence, source_section, source_file, extraction_method):
    """Create a detailed fact record with validation and confidence metadata."""
    normalized_value = _normalize_field_value(field_name, value)
    machine_value = _normalize_machine_value(field_name, normalized_value)
    evidence_text = _normalize_whitespace(evidence)
    validation_passed = _is_valid_field_value(field_name, normalized_value)

    confidence = 0.0
    if normalized_value not in MISSING_VALUES:
        confidence += 0.45
    if evidence_text:
        confidence += 0.30
    if validation_passed:
        confidence += 0.25
    confidence = round(min(confidence, 0.99), 2)

    if normalized_value in MISSING_VALUES:
        validation_status = "missing"
    elif validation_passed:
        validation_status = "passed"
    else:
        validation_status = "failed"

    return {
        "value": normalized_value,
        "normalized_value": machine_value,
        "evidence": evidence_text,
        "source_section": str(source_section or "").strip(),
        "source_file": str(source_file or "").strip(),
        "validation_status": validation_status,
        "confidence": confidence,
        "extraction_method": extraction_method,
    }


def _build_heuristic_fact_records(extracted_values, evidence_map, source_file):
    """Convert heuristic outputs into detailed fact records."""
    records = {}
    for field_name in GENERAL_FACT_FIELDS:
        evidence_item = evidence_map.get(field_name, {})
        records[field_name] = _build_field_record(
            field_name=field_name,
            value=extracted_values.get(field_name),
            evidence=evidence_item.get("evidence", ""),
            source_section=evidence_item.get("source_section", ""),
            source_file=source_file,
            extraction_method="heuristic",
        )
    return records


def _build_llm_fact_prompt(section_pairs):
    """Create a compact extraction prompt for the active brief."""
    section_blocks = [f"[Section] {title}\n{body}" for title, body in section_pairs]
    schema_hint = {
        "assessment_name": {"value": None, "evidence": None, "source_section": None},
        "course": {"value": None, "evidence": None, "source_section": None},
        "report_due_date": {"value": None, "evidence": None, "source_section": None},
        "report_word_limit": {"value": None, "evidence": None, "source_section": None},
        "report_font": {"value": None, "evidence": None, "source_section": None},
        "report_format": {"value": None, "evidence": None, "source_section": None},
        "report_filename": {"value": None, "evidence": None, "source_section": None},
        "slide_format": {"value": None, "evidence": None, "source_section": None},
        "slide_filename": {"value": None, "evidence": None, "source_section": None},
        "presentation_duration": {"value": None, "evidence": None, "source_section": None},
        "presentation_submission": {"value": None, "evidence": None, "source_section": None},
        "presentation_due_dates": {
            "value": {"group_a": None, "group_b": None},
            "evidence": None,
            "source_section": None,
        },
        "similarity_limit": {"value": None, "evidence": None, "source_section": None},
        "group_submission": {"value": None, "evidence": None, "source_section": None},
        "first_page_requirements": {"value": [], "evidence": None, "source_section": None},
        "first_slide_requirements": {"value": [], "evidence": None, "source_section": None},
        "presentation_highlights": {"value": [], "evidence": None, "source_section": None},
        "report_sections": {"value": [], "evidence": None, "source_section": None},
        "github_required": {"value": None, "evidence": None, "source_section": None},
        "individual_contribution_required": {"value": None, "evidence": None, "source_section": None},
        "maximum_group_size": {"value": None, "evidence": None, "source_section": None},
        "extension_policy": {"value": None, "evidence": None, "source_section": None},
        "academic_integrity_guidance": {"value": None, "evidence": None, "source_section": None},
        "rubric_sections": {
            "value": [
                {
                    "scope": "report or presentation",
                    "title": "rubric section title",
                    "hd_requirement": "HD requirement summary",
                }
            ],
            "evidence": None,
            "source_section": None,
        },
    }

    system_prompt = (
        "You extract structured assessment facts from one uploaded assignment brief. "
        "Return only valid JSON. Do not invent information. If a field is not supported by the document, "
        "return null for scalar values or an empty list or object where appropriate. "
        "Each field must include a short supporting evidence quote copied from the document and the section name when possible."
    )
    user_prompt = (
        "Extract the key assignment facts from this brief.\n\n"
        "Return JSON in the following schema shape exactly:\n"
        f"{json.dumps(schema_hint, ensure_ascii=False, indent=2)}\n\n"
        "Document sections:\n"
        f"{chr(10).join(section_blocks)}"
    )
    return system_prompt, user_prompt


def _parse_llm_json_response(raw_text):
    """Parse a JSON object from an LLM extraction response."""
    if not raw_text:
        return {}

    response_text = raw_text.strip()
    if response_text.startswith("```"):
        response_text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", response_text)
        response_text = re.sub(r"\s*```$", "", response_text)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _extract_facts_with_llm(section_pairs):
    """Use one LLM call to extract a general fact record set when configured."""
    client = _create_llm_client()
    if client is None:
        return {}, ""

    _, default_model, base_url = _get_llm_settings()
    model = os.getenv(FACT_EXTRACTION_MODEL_ENV, "").strip() or default_model
    system_prompt, user_prompt = _build_llm_fact_prompt(section_pairs)
    prefers_chat_completion = base_url.rstrip("/") != "https://api.openai.com/v1"

    try:
        if prefers_chat_completion:
            response_text = _call_chat_completion(client, model, system_prompt, user_prompt)
        else:
            response_text = _call_responses_api(client, model, system_prompt, user_prompt)
    except Exception as exc:
        print("LLM fact extraction failed:")
        print(type(exc).__name__)
        print(exc)
        return {}, model

    parsed_payload = _parse_llm_json_response(response_text)
    if parsed_payload:
        print("LLM fact extraction succeeded.")
        print("Parsed LLM fields:", list(parsed_payload.keys()))
    else:
        print("LLM fact extraction returned no parseable fields.")
    return parsed_payload, model


def _build_records_from_llm_payload(llm_payload, source_file):
    """Normalize LLM extraction JSON into field records."""
    if not isinstance(llm_payload, dict):
        return {}

    llm_records = {}
    for field_name in GENERAL_FACT_FIELDS:
        raw_field = llm_payload.get(field_name)
        if isinstance(raw_field, dict):
            raw_value = raw_field.get("value")
            raw_evidence = raw_field.get("evidence", "")
            raw_source_section = raw_field.get("source_section", "")
        else:
            raw_value = raw_field
            raw_evidence = ""
            raw_source_section = ""

        llm_records[field_name] = _build_field_record(
            field_name=field_name,
            value=raw_value,
            evidence=raw_evidence,
            source_section=raw_source_section,
            source_file=source_file,
            extraction_method="llm",
        )

    return llm_records


def _choose_better_record(field_name, heuristic_record, llm_record):
    """Select the stronger record between heuristic and LLM extraction."""
    if not llm_record:
        return heuristic_record
    if not heuristic_record:
        return llm_record

    heuristic_value = heuristic_record.get("value")
    llm_value = llm_record.get("value")

    if heuristic_record.get("validation_status") != "passed" and llm_record.get("validation_status") == "passed":
        return {**llm_record, "extraction_method": "llm+validated"}
    if llm_record.get("validation_status") != "passed":
        return heuristic_record

    if heuristic_value in MISSING_VALUES and llm_value not in MISSING_VALUES:
        return {**llm_record, "extraction_method": "llm+filled"}

    if field_name in {
        "report_due_date",
        "report_word_limit",
        "report_font",
        "report_format",
        "report_filename",
        "slide_format",
        "slide_filename",
        "presentation_duration",
        "similarity_limit",
        "maximum_group_size",
    }:
        if isinstance(llm_value, str) and isinstance(heuristic_value, str) and len(llm_value) > len(heuristic_value):
            return {**llm_record, "extraction_method": "hybrid_merge"}
        return heuristic_record

    if field_name in {
        "first_page_requirements",
        "first_slide_requirements",
        "presentation_highlights",
        "report_sections",
        "rubric_sections",
    }:
        if isinstance(llm_value, list) and len(llm_value) > len(heuristic_value or []):
            return {**llm_record, "extraction_method": "hybrid_merge"}
        return heuristic_record

    if field_name == "presentation_due_dates":
        if isinstance(llm_value, dict) and len(llm_value) > len(heuristic_value or {}):
            return {**llm_record, "extraction_method": "hybrid_merge"}
        return heuristic_record

    if field_name in {"github_required", "individual_contribution_required"}:
        if isinstance(llm_value, bool):
            return {**llm_record, "extraction_method": "hybrid_merge"}
        return heuristic_record

    if isinstance(llm_value, str) and len(llm_value) > len(str(heuristic_value or "")):
        return {**llm_record, "extraction_method": "hybrid_merge"}

    return heuristic_record


def _merge_fact_records(heuristic_records, llm_records):
    """Merge heuristic and LLM field records into one final record set."""
    merged_records = {}
    for field_name in GENERAL_FACT_FIELDS:
        merged_records[field_name] = _choose_better_record(
            field_name,
            heuristic_records.get(field_name),
            llm_records.get(field_name),
        )
    return merged_records


def _records_to_flat_values(field_records):
    """Convert detailed field records back into the flat structure used at runtime."""
    flat_values = {}
    for field_name, record in field_records.items():
        flat_values[field_name] = record.get("value")
    return flat_values


def _build_rubric_maps_from_sections(rubric_sections):
    """Derive report and presentation rubric maps from normalized rubric section records."""
    report_map = {}
    presentation_map = {}
    for item in rubric_sections:
        if item.get("scope") == "report":
            report_map[item["slug"]] = item.get("hd_requirement", "")
        elif item.get("scope") == "presentation":
            presentation_map[item["slug"]] = item.get("hd_requirement", "")
    return report_map, presentation_map


def _build_fact_extraction_metadata(field_records, use_llm, llm_model):
    """Summarize how the structured facts were produced for evaluation and reporting."""
    passed_fields = sum(1 for record in field_records.values() if record.get("validation_status") == "passed")
    missing_fields = sum(1 for record in field_records.values() if record.get("validation_status") == "missing")
    llm_used = any("llm" in str(record.get("extraction_method", "")) for record in field_records.values())
    return {
        "schema_version": FACT_SCHEMA_VERSION,
        "llm_requested": bool(use_llm),
        "llm_used": llm_used,
        "llm_model": llm_model if llm_used else "",
        "field_count": len(field_records),
        "validated_field_count": passed_fields,
        "missing_field_count": missing_fields,
    }


def _extract_heuristic_facts(active_brief_path, brief_text, section_pairs):
    """Extract heuristic facts together with source evidence for each field."""
    section_lookup = _build_section_lookup(section_pairs)
    rubric_sections, report_rubric_hd_requirements, presentation_rubric_hd_requirements = _extract_rubric_facts(section_pairs)

    overview_title, overview_text = _find_section(section_pairs, ["overview"])
    length_title, length_text = _find_section(section_pairs, ["length"])
    similarity_title, similarity_text = _find_section(section_pairs, ["similarity"])
    submission_title, submission_text = _find_section(section_pairs, ["submission format"])
    presentation_session_title, presentation_session_text = _find_section(section_pairs, ["presentation session"])
    report_sections_title, report_sections_text = _find_section(
        section_pairs,
        ["the report must include the following sections", "report sections"],
    )
    task_overview_title, task_overview_text = _find_section(section_pairs, ["task overview"])
    groupwork_title, groupwork_text = _find_section(section_pairs, ["groupwork"])
    requirements_title, requirements_text = _find_section(section_pairs, ["requirements"])
    extension_title, extension_text = _find_section(section_pairs, ["extensions"])
    academic_integrity_title, academic_integrity_text = _find_section(section_pairs, ["academic integrity"])
    part_2_presentation_title, part_2_presentation_text = _find_section(section_pairs, ["part 2 presentation"])
    presentation_title, presentation_text = _find_section(section_pairs, ["presentation"])
    oral_presentation_title, oral_presentation_text = _find_section(section_pairs, ["oral presentation"])
    presentation_focus_title = part_2_presentation_title or presentation_title
    presentation_focus_text = part_2_presentation_text or presentation_text

    report_due_date, report_due_evidence, report_due_section = _extract_report_due_date(brief_text, section_pairs)
    presentation_due_dates, presentation_due_evidence, presentation_due_section = _extract_presentation_due_dates(
        brief_text,
        section_pairs,
    )
    report_filename, report_filename_evidence = _extract_first_match_with_sentence(
        brief_text,
        [r"(AT\d+_Report_Group_[A-Za-z0-9xX-]+\.pdf)"],
    )
    slide_filename, slide_filename_evidence = _extract_first_match_with_sentence(
        brief_text,
        [r"(AT\d+_Slide_Group_[A-Za-z0-9xX-]+\.pptx)"],
    )
    report_word_limit, report_word_evidence = _extract_first_match_with_sentence(
        length_text or brief_text,
        [r"(up to\s+[\d,]+\s+words[^.]*)"],
    )
    report_font, report_font_evidence = _extract_first_match_with_sentence(
        length_text or brief_text,
        [r"(11-point\s+[^.]*?(?:times|arial)[^.]*fonts?)"],
    )
    similarity_limit, similarity_evidence = _extract_first_match_with_sentence(
        similarity_text or brief_text,
        [r"(20%)", r"(maximum allowed similarity score is\s+\d+%)"],
    )
    presentation_duration, presentation_duration_evidence = _extract_first_match_with_sentence(
        presentation_session_text or brief_text,
        [r"(\d+\s+minutes?[^.]*q&a)", r"(\d+-minute presentation plus \d+-minute q&a)"],
    )
    maximum_group_size, maximum_group_size_evidence = _extract_first_match_with_sentence(
        groupwork_text or task_overview_text or brief_text,
        [
            r"(maximum group size is\s+[A-Za-z0-9-]+\s+students?)",
            r"(maximum\s+[A-Za-z0-9-]+\s+students?\s+per group)",
        ],
    )

    first_page_sentence = _first_matching_sentence(brief_text, ["first page"])
    first_slide_sentence = _first_matching_sentence(brief_text, ["first slide"])
    group_submission_value, group_submission_evidence = _extract_group_submission_sentence(brief_text)
    presentation_submission_sentence = _first_sentence_matching_any(
        oral_presentation_text or presentation_session_text or brief_text,
        [
            ["powerpoint", "submitted", "presentation"],
            ["powerpoint", "submitted", "class"],
            ["presentation", "submitted", "before"],
        ],
    )
    github_sentence = _first_sentence_matching_any(
        report_sections_text or requirements_text or brief_text,
        [
            ["github", "repository"],
            ["publicly accessible", "github"],
        ],
    )
    contributions_sentence = _first_sentence_matching_any(
        task_overview_text or brief_text,
        [
            ["document", "contributions"],
            ["individual", "contributions"],
        ],
    )
    highlights_sentence = _first_sentence_matching_any(
        presentation_focus_text or brief_text,
        [["highlight"]],
    )

    report_format = "PDF" if "pdf" in (submission_text or requirements_text or brief_text).lower() else ""
    slide_format = "PPTX" if "pptx" in brief_text.lower() or "powerpoint" in brief_text.lower() else ""
    report_format_evidence = _first_sentence_matching_any(
        submission_text or requirements_text or brief_text,
        [["report", "pdf"], ["report submission", "pdf"]],
    )
    slide_format_evidence = _first_sentence_matching_any(
        requirements_text or oral_presentation_text or brief_text,
        [["powerpoint", "filename"], ["pptx"], ["presentation", "filename"]],
    )

    presentation_highlights = _extract_presentation_highlights(presentation_focus_text or "")
    report_sections = _extract_report_sections(report_sections_text)

    extracted_values = {
        "brief_source_file": active_brief_path.name,
        "brief_source_path": str(active_brief_path),
        "assessment_name": _first_non_empty_line(brief_text),
        "course": _extract_first_match(brief_text, [r"(\b\d{5}\s+[A-Za-z][^\n.]*)"]),
        "report_due_date": report_due_date,
        "report_word_limit": report_word_limit,
        "report_font": report_font,
        "report_format": report_format,
        "report_filename": report_filename,
        "slide_format": slide_format,
        "slide_filename": slide_filename,
        "presentation_duration": presentation_duration,
        "presentation_submission": presentation_submission_sentence,
        "presentation_due_dates": presentation_due_dates,
        "similarity_limit": similarity_limit,
        "group_submission": group_submission_value,
        "first_page_requirements": _extract_list_from_sentence(first_page_sentence),
        "first_slide_requirements": _extract_list_from_sentence(first_slide_sentence),
        "presentation_highlights": presentation_highlights,
        "report_sections": report_sections,
        "github_required": "github" in github_sentence.lower() if github_sentence else False,
        "individual_contribution_required": bool(contributions_sentence),
        "maximum_group_size": maximum_group_size,
        "extension_policy": extension_text,
        "academic_integrity_guidance": academic_integrity_text,
        "rubric_sections": rubric_sections,
    }

    evidence_map = {
        "assessment_name": {
            "evidence": _first_non_empty_line(brief_text),
            "source_section": "Document Overview",
        },
        "course": {
            "evidence": _extract_first_match(brief_text, [r"(\b\d{5}\s+[A-Za-z][^\n.]*)"]),
            "source_section": overview_title or "Document Overview",
        },
        "report_due_date": {
            "evidence": report_due_evidence,
            "source_section": report_due_section,
        },
        "report_word_limit": {
            "evidence": report_word_evidence,
            "source_section": length_title,
        },
        "report_font": {
            "evidence": report_font_evidence or report_word_evidence,
            "source_section": length_title,
        },
        "report_format": {
            "evidence": report_format_evidence,
            "source_section": submission_title or requirements_title,
        },
        "report_filename": {
            "evidence": report_filename_evidence,
            "source_section": requirements_title,
        },
        "slide_format": {
            "evidence": slide_format_evidence,
            "source_section": requirements_title or oral_presentation_title,
        },
        "slide_filename": {
            "evidence": slide_filename_evidence,
            "source_section": requirements_title or oral_presentation_title,
        },
        "presentation_duration": {
            "evidence": presentation_duration_evidence,
            "source_section": presentation_session_title,
        },
        "presentation_submission": {
            "evidence": presentation_submission_sentence,
            "source_section": oral_presentation_title or presentation_session_title,
        },
        "presentation_due_dates": {
            "evidence": presentation_due_evidence,
            "source_section": presentation_due_section,
        },
        "similarity_limit": {
            "evidence": similarity_evidence,
            "source_section": similarity_title,
        },
        "group_submission": {
            "evidence": group_submission_evidence,
            "source_section": submission_title or requirements_title,
        },
        "first_page_requirements": {
            "evidence": first_page_sentence,
            "source_section": submission_title or requirements_title,
        },
        "first_slide_requirements": {
            "evidence": first_slide_sentence,
            "source_section": oral_presentation_title or requirements_title,
        },
        "presentation_highlights": {
            "evidence": highlights_sentence,
            "source_section": presentation_focus_title or "Presentation",
        },
        "report_sections": {
            "evidence": report_sections_text,
            "source_section": report_sections_title,
        },
        "github_required": {
            "evidence": github_sentence,
            "source_section": report_sections_title or requirements_title,
        },
        "individual_contribution_required": {
            "evidence": contributions_sentence,
            "source_section": task_overview_title or groupwork_title,
        },
        "maximum_group_size": {
            "evidence": maximum_group_size_evidence,
            "source_section": groupwork_title or task_overview_title,
        },
        "extension_policy": {
            "evidence": extension_text,
            "source_section": extension_title,
        },
        "academic_integrity_guidance": {
            "evidence": academic_integrity_text,
            "source_section": academic_integrity_title,
        },
        "rubric_sections": {
            "evidence": " | ".join(item["title"] for item in rubric_sections),
            "source_section": "Rubric Sections",
        },
    }

    return (
        extracted_values,
        evidence_map,
        report_rubric_hd_requirements,
        presentation_rubric_hd_requirements,
    )


def extract_structured_facts(brief_path=None, use_llm=None):
    """Extract a generic structured fact set from the active assignment brief."""
    active_brief_path = Path(brief_path) if brief_path is not None else get_active_brief_path()
    brief_text = read_document_text(active_brief_path)
    section_pairs = split_into_sections(brief_text)
    source_file = active_brief_path.name

    (
        heuristic_values,
        evidence_map,
        heuristic_report_rubric_map,
        heuristic_presentation_rubric_map,
    ) = _extract_heuristic_facts(active_brief_path, brief_text, section_pairs)
    heuristic_records = _build_heuristic_fact_records(heuristic_values, evidence_map, source_file)

    should_use_llm = _llm_fact_extraction_enabled() if use_llm is None else bool(use_llm)
    llm_payload = {}
    llm_model = ""
    llm_records = {}
    if should_use_llm:
        llm_payload, llm_model = _extract_facts_with_llm(section_pairs)
        llm_records = _build_records_from_llm_payload(llm_payload, source_file)

    merged_records = _merge_fact_records(heuristic_records, llm_records)

    # If an LLM payload was successfully parsed, mark fields that were validated
    # by both heuristic extraction and LLM extraction. This preserves the more
    # stable heuristic value when appropriate, while correctly recording that
    # LLM-assisted extraction was used in the pipeline.
    if should_use_llm and llm_records:
        for field_name, llm_record in llm_records.items():
            if not llm_record:
                continue

            llm_value = llm_record.get("value")
            llm_status = llm_record.get("validation_status")
            merged_record = merged_records.get(field_name)

            if (
                merged_record
                and llm_value not in MISSING_VALUES
                and llm_status == "passed"
                and "llm" not in str(merged_record.get("extraction_method", "")).lower()
            ):
                merged_record["extraction_method"] = (
                    str(merged_record.get("extraction_method", "heuristic"))
                    + "+llm_verified"
                )
                merged_record["llm_evidence"] = llm_record.get("evidence", "")
                merged_record["llm_value"] = llm_value

    flat_values = _records_to_flat_values(merged_records)

    rubric_sections = flat_values.get("rubric_sections") or heuristic_values.get("rubric_sections", [])
    report_rubric_hd_requirements, presentation_rubric_hd_requirements = _build_rubric_maps_from_sections(rubric_sections)
    if not report_rubric_hd_requirements:
        report_rubric_hd_requirements = heuristic_report_rubric_map
    if not presentation_rubric_hd_requirements:
        presentation_rubric_hd_requirements = heuristic_presentation_rubric_map

    facts = {
        "schema_version": FACT_SCHEMA_VERSION,
        "brief_source_file": active_brief_path.name,
        "brief_source_path": str(active_brief_path),
        "fact_extraction": _build_fact_extraction_metadata(merged_records, should_use_llm, llm_model),
        "field_records": merged_records,
        **flat_values,
        "rubric_hd_requirements": report_rubric_hd_requirements,
        "presentation_rubric_hd_requirements": presentation_rubric_hd_requirements,
    }

    return facts


def save_structured_facts(output_path, brief_path=None, use_llm=None):
    """Extract structured facts and save them to the processed data directory."""
    output_file = Path(output_path)
    facts = extract_structured_facts(brief_path=brief_path, use_llm=use_llm)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as output_stream:
        json.dump(facts, output_stream, ensure_ascii=False, indent=2)

    return facts
