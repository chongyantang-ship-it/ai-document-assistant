import os
import re
import time
from datetime import datetime
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://api.openai.com/v1"
GEMINI_BASE_URL_HOST = "generativelanguage.googleapis.com"
GEMINI_CONSERVATIVE_FREE_MODE_RPM = {
    "gemini-2.5-flash-lite": 8,
    "gemini-2.5-flash": 4,
    "gemini-3.1-flash-lite": 12,
    "gemini-3-flash": 4,
}
_LAST_REQUEST_AT_BY_MODEL = {}

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


def _confidence_from_score(score):
    """Convert a numeric retrieval score into a user-facing confidence label."""
    if score >= 0.55:
        return "High"
    if score >= 0.30:
        return "Medium"
    return "Low"


def _get_llm_settings():
    """Read the configured LLM API settings from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or DEFAULT_BASE_URL
    return api_key, model, base_url


def _create_llm_client():
    """Create an OpenAI-compatible client when the runtime is configured."""
    if OpenAI is None:
        return None

    api_key, _, base_url = _get_llm_settings()
    if not api_key:
        return None

    return OpenAI(api_key=api_key, base_url=base_url)


def llm_is_configured():
    """Return True when an OpenAI-compatible client can be created."""
    return _create_llm_client() is not None


def _normalize_model_name(model):
    """Normalize a model identifier so provider-specific variants map consistently."""
    return model.strip().lower()


def _is_gemini_request(model, base_url):
    """Return True when the configured request targets Gemini-compatible endpoints."""
    normalized_model = _normalize_model_name(model)
    normalized_base_url = base_url.strip().lower()
    return normalized_model.startswith("gemini") or GEMINI_BASE_URL_HOST in normalized_base_url


def _llm_free_mode_is_enabled():
    """Return True when conservative free-tier throttling should be applied."""
    return os.getenv("LLM_FREE_MODE", "1").strip().lower() in {"1", "true", "yes", "on"}


def _get_gemini_free_mode_rpm(model):
    """Return the conservative RPM cap for a Gemini model in free mode."""
    normalized_model = _normalize_model_name(model)

    if normalized_model in GEMINI_CONSERVATIVE_FREE_MODE_RPM:
        return GEMINI_CONSERVATIVE_FREE_MODE_RPM[normalized_model]

    for known_model, conservative_rpm in GEMINI_CONSERVATIVE_FREE_MODE_RPM.items():
        if normalized_model.startswith(known_model):
            return conservative_rpm

    return 4


def _get_request_spacing_seconds(model, base_url):
    """Return the minimum spacing between live requests for the active provider settings."""
    if not _llm_free_mode_is_enabled():
        return 0.0

    if not _is_gemini_request(model, base_url):
        return 0.0

    conservative_rpm = _get_gemini_free_mode_rpm(model)
    if conservative_rpm <= 0:
        return 0.0

    return 60.0 / conservative_rpm


def _wait_for_rate_limit_window(model, base_url):
    """Throttle Gemini free-mode requests so calls stay under a conservative RPM cap."""
    spacing_seconds = _get_request_spacing_seconds(model, base_url)
    if spacing_seconds <= 0:
        return

    normalized_model = _normalize_model_name(model)
    now = time.time()
    last_request_at = _LAST_REQUEST_AT_BY_MODEL.get(normalized_model)
    if last_request_at is None:
        return

    elapsed_seconds = now - last_request_at
    remaining_wait_seconds = spacing_seconds - elapsed_seconds
    if remaining_wait_seconds <= 0:
        return

    if os.getenv("LLM_DEBUG", "0").strip() == "1":
        print(
            f"[LLM_DEBUG] Free-mode rate limiting for {normalized_model}. "
            f"Sleeping {round(remaining_wait_seconds, 2)}s before the next request."
        )
    time.sleep(remaining_wait_seconds)


def _record_request_timestamp(model):
    """Record when the latest request for a model was sent."""
    _LAST_REQUEST_AT_BY_MODEL[_normalize_model_name(model)] = time.time()


def _question_tokens(question):
    """Extract non-trivial lexical tokens from a question."""
    tokens = re.findall(r"[a-zA-Z0-9]+", question.lower())
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "is", "are",
        "was", "were", "be", "been", "being", "it", "this", "that", "with", "as", "by",
        "from", "can", "could", "should", "would", "will", "what", "which", "when", "where",
        "who", "how", "why", "do", "does", "did", "i", "we", "you", "they", "he", "she",
        "my", "our", "your", "their", "me", "us", "kind", "kinds", "use", "using",
    }
    return [token for token in tokens if len(token) >= 3 and token not in stopwords]


def _detect_answer_intent(question):
    """Infer whether the response should emphasise planning, evaluation, policy, or explanation."""
    lower_question = question.lower()

    if any(
        token in lower_question
        for token in [
            "plan",
            "roadmap",
            "divide work",
            "finish project",
            "timeline",
            "role allocation",
            "group members",
            "six group members",
        ]
    ):
        return "plan"
    if any(token in lower_question for token in ["ethic", "ethical", "genai", "allowed", "policy", "integrity"]):
        return "policy"
    if any(token in lower_question for token in ["evaluate", "evaluation", "metric", "baseline", "experiment"]):
        return "evaluation"
    if any(token in lower_question for token in ["rubric", "hd", "high distinction", "criterion"]):
        return "rubric"

    return "general"


def _generate_template_answer(retrieved_chunks):
    """Return a simple fallback answer based on the highest ranked chunk."""
    best_chunk = retrieved_chunks[0]
    return (
        "Based on the retrieved assignment documents, the most relevant information is: "
        + best_chunk["text"].replace("\n", " ")[:700]
    )


def _extract_relevant_points(question, retrieved_chunks, max_points=3):
    """Extract a few answer-ready bullet points from the retrieved evidence."""
    question_tokens = _question_tokens(question)
    candidates = []

    for chunk in retrieved_chunks[:4]:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", chunk.get("text", ""))
        for sentence in sentences:
            compact_sentence = sentence.strip(" -\t")
            if len(compact_sentence) < 20:
                continue

            lower_sentence = compact_sentence.lower()
            overlap = sum(1 for token in question_tokens if token in lower_sentence) if question_tokens else 0
            score = overlap + chunk.get("score", 0)
            candidates.append((score, overlap, compact_sentence))

    candidates.sort(key=lambda item: item[0], reverse=True)
    points = []
    seen = set()

    for _, overlap, sentence in candidates:
        if question_tokens and overlap == 0:
            continue
        normalized_sentence = sentence.lower()
        if normalized_sentence in seen:
            continue
        seen.add(normalized_sentence)
        points.append(sentence[:220])
        if len(points) >= max_points:
            break

    if not points:
        for _, _, sentence in candidates[:max_points]:
            normalized_sentence = sentence.lower()
            if normalized_sentence in seen:
                continue
            seen.add(normalized_sentence)
            points.append(sentence[:220])

    return points


def _generate_structured_template_answer(question, retrieved_chunks):
    """Generate a deterministic structured fallback answer from retrieved chunks."""
    points = _extract_relevant_points(question, retrieved_chunks, max_points=3)
    if not points:
        return _generate_template_answer(retrieved_chunks)

    direct_answer = points[0]
    bullet_lines = "\n".join(f"- {point}" for point in points)
    return f"Direct Answer: {direct_answer}\nKey Points:\n{bullet_lines}"


def _generate_template_plan_answer(question, retrieved_chunks):
    """Generate a deterministic planning-oriented fallback answer."""
    points = _extract_relevant_points(question, retrieved_chunks, max_points=3)
    direct_answer = (
        "Use the official brief to prioritise problem definition, theory, workflow, evaluation, "
        "reflection, and presentation preparation, and leave time for integration and rehearsal."
    )
    evidence_lines = "\n".join(f"- {point}" for point in points) if points else "- No strong planning evidence was retrieved."
    return (
        f"Direct Answer: {direct_answer}\n"
        "Action Plan:\n"
        "1. Confirm deadlines, submission rules, required report sections, and presentation constraints.\n"
        "2. Divide ownership across implementation, evaluation, report writing, and presentation preparation.\n"
        "3. Finish integration, final checks, and rehearsal before the final submission window.\n"
        "Evidence Used:\n"
        f"{evidence_lines}"
    )


def _generate_template_evaluation_answer(question, retrieved_chunks):
    """Generate a deterministic evaluation-oriented fallback answer."""
    points = _extract_relevant_points(question, retrieved_chunks, max_points=3)
    direct_answer = (
        "Evaluate the system with representative question sets, compare it with baseline methods, "
        "and report accuracy, evidence support, hallucination behaviour, and latency."
    )
    evidence_lines = "\n".join(f"- {point}" for point in points) if points else "- No strong evaluation evidence was retrieved."
    return (
        f"Direct Answer: {direct_answer}\n"
        "Evaluation Steps:\n"
        "1. Build a question set covering factual, rubric, workflow, and unsupported cases.\n"
        "2. Compare the hybrid system with keyword, RAG-only, and LLM-only baselines.\n"
        "3. Report answer accuracy, retrieval hit rate, citation support, hallucination rate, and response time.\n"
        "Evidence Used:\n"
        f"{evidence_lines}"
    )


def _generate_template_policy_answer(question, retrieved_chunks):
    """Generate a deterministic policy-oriented fallback answer."""
    points = _extract_relevant_points(question, retrieved_chunks, max_points=3)
    direct_answer = (
        "Use the documents cautiously: they support responsible AI use, but they do not grant blanket permission "
        "to use GenAI without checking subject rules."
    )
    evidence_lines = "\n".join(f"- {point}" for point in points) if points else "- No strong policy evidence was retrieved."
    return (
        f"Direct Answer: {direct_answer}\n"
        "Policy Interpretation:\n"
        "- Follow the explicit assignment and subject guidance before relying on GenAI.\n"
        "- Use AI as support rather than as a replacement for your own work or official teaching advice.\n"
        "- Acknowledge and reference GenAI use when required.\n"
        "Evidence Used:\n"
        f"{evidence_lines}"
    )


def _extract_response_text(response):
    """Extract plain text from an OpenAI-compatible responses API payload."""
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()

    output_items = getattr(response, "output", None) or []
    parts = []
    for item in output_items:
        for content_item in getattr(item, "content", []) or []:
            if getattr(content_item, "type", "") == "output_text":
                value = getattr(content_item, "text", "")
                if value:
                    parts.append(value)
    joined = "\n".join(parts).strip()
    return joined if joined else None


def _extract_chat_completion_text(completion):
    """Extract plain text from a chat completion payload across provider variants."""
    message = completion.choices[0].message
    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip() or None

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if text:
                parts.append(str(text).strip())

        joined = "\n".join(part for part in parts if part).strip()
        if joined:
            return joined

    refusal = getattr(message, "refusal", None)
    if refusal:
        return str(refusal).strip()

    return None


def _extract_retry_delay_seconds(error):
    """Infer a retry delay from an API exception when the provider supplies one."""
    body = getattr(error, "body", None)
    serialized_body = str(body) if body is not None else ""
    serialized_error = f"{serialized_body}\n{error}"

    retry_patterns = [
        r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+)s",
        r"retry in\s+(\d+(?:\.\d+)?)",
    ]
    for pattern in retry_patterns:
        match = re.search(pattern, serialized_error, flags=re.IGNORECASE)
        if match:
            try:
                return max(1, int(float(match.group(1))))
            except ValueError:
                continue

    return None


def _should_retry_llm_error(error):
    """Return True when an API exception looks transient enough to retry."""
    status_code = getattr(error, "status_code", None)
    return status_code in {408, 429, 500, 502, 503, 504}


def _call_with_retry(request_fn, debug_label, model, base_url):
    """Execute an LLM request with light retry support for transient provider errors."""
    max_attempts = max(1, int(os.getenv("LLM_MAX_RETRIES", "3")))
    default_delay_seconds = max(1, int(os.getenv("LLM_RETRY_DELAY_SECONDS", "8")))

    for attempt in range(1, max_attempts + 1):
        try:
            _wait_for_rate_limit_window(model, base_url)
            _record_request_timestamp(model)
            return request_fn()
        except Exception as error:
            if attempt >= max_attempts or not _should_retry_llm_error(error):
                raise

            retry_delay_seconds = _extract_retry_delay_seconds(error) or default_delay_seconds
            if os.getenv("LLM_DEBUG", "0").strip() == "1":
                print(
                    f"[LLM_DEBUG] {debug_label} failed on attempt {attempt}/{max_attempts}. "
                    f"Retrying in {retry_delay_seconds}s."
                )
            time.sleep(retry_delay_seconds)


def _build_evidence_block(retrieved_chunks):
    """Format retrieved chunks into a prompt-friendly evidence block."""
    evidence_lines = []
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        evidence_lines.append(
            f"[{rank}] authority={chunk.get('authority', 'unknown')} "
            f"source={chunk.get('source_file', 'unknown')} "
            f"section={chunk.get('section', 'unknown')} "
            f"chunk_id={chunk.get('chunk_id', 'unknown')} "
            f"score={round(chunk.get('score', 0), 4)}\n{chunk.get('text', '')}"
        )
    return "\n\n".join(evidence_lines)


def _build_grounded_prompts(question, retrieved_chunks):
    """Create the system and user prompts for evidence-grounded generation."""
    evidence_block = _build_evidence_block(retrieved_chunks)
    intent = _detect_answer_intent(question)
    current_date = datetime.now().strftime("%Y-%m-%d")

    system_prompt = (
        "You are an academic document assistant. "
        "Answer only from the supplied evidence. "
        "If the evidence is insufficient, say so explicitly. "
        "Do not invent requirements, dates, policies, or performance claims. "
        "Treat official course documents as authoritative for deadlines, submission rules, academic integrity, "
        "and GenAI guidance. Treat project planning documents as suggestions for workflow and team coordination."
    )

    if intent == "evaluation":
        format_instruction = (
            "Return the response in this exact structure:\n"
            "Direct Answer: <1-2 lines>\n"
            "Evaluation Steps:\n"
            "1. <step>\n"
            "2. <step>\n"
            "3. <step>\n"
            "Evidence Used:\n"
            "- <short evidence point>\n"
            "- <short evidence point>\n"
            "Keep the answer precise and report-friendly."
        )
    elif intent == "policy":
        format_instruction = (
            "Return the response in this exact structure:\n"
            "Direct Answer: <1-2 lines>\n"
            "Policy Interpretation:\n"
            "- <policy point>\n"
            "- <policy point>\n"
            "Evidence Used:\n"
            "- <short evidence point>\n"
            "- <short evidence point>\n"
            "Be cautious when the evidence says to check official subject rules."
        )
    elif intent == "plan":
        format_instruction = (
            "Return the response in this exact structure:\n"
            "Direct Answer: <1-2 lines>\n"
            "Action Plan:\n"
            "1. <step>\n"
            "2. <step>\n"
            "3. <step>\n"
            "Evidence Used:\n"
            "- <short evidence point>\n"
            "- <short evidence point>\n"
            "Keep the answer practical, grounded, and deadline-aware."
        )
    else:
        format_instruction = (
            "Return the response in this exact structure:\n"
            "Direct Answer: <1-3 lines>\n"
            "Key Points:\n"
            "- <point>\n"
            "- <point>\n"
            "Evidence Used:\n"
            "- <short evidence point>\n"
            "- <short evidence point>\n"
            "Keep the answer concise and academically precise."
        )

    prompt_prefix = f"Current Date: {current_date}\n\n" if intent == "plan" else ""
    user_prompt = f"{prompt_prefix}Question:\n{question}\n\nEvidence:\n{evidence_block}\n\n{format_instruction}"
    return intent, system_prompt, user_prompt


def _call_chat_completion(client, model, system_prompt, user_prompt):
    """Call the chat completions API and return plain text when available."""
    _, _, base_url = _get_llm_settings()
    completion = _call_with_retry(
        lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        ),
        "Chat completion request",
        model,
        base_url,
    )
    return _extract_chat_completion_text(completion)


def _call_responses_api(client, model, system_prompt, user_prompt):
    """Call the responses API and return plain text when available."""
    _, _, base_url = _get_llm_settings()
    response = _call_with_retry(
        lambda: client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        ),
        "Responses API request",
        model,
        base_url,
    )
    answer = _extract_response_text(response)
    return answer if answer else None


def _generate_llm_answer(question, retrieved_chunks):
    """Generate an evidence-grounded answer with an OpenAI-compatible client."""
    client = _create_llm_client()
    if client is None:
        return None

    _, model, base_url = _get_llm_settings()
    _, system_prompt, user_prompt = _build_grounded_prompts(question, retrieved_chunks)
    prefers_chat_completion = base_url.rstrip("/") != DEFAULT_BASE_URL.rstrip("/")

    try:
        if prefers_chat_completion:
            return _call_chat_completion(client, model, system_prompt, user_prompt)
        return _call_responses_api(client, model, system_prompt, user_prompt)
    except Exception as primary_error:
        if prefers_chat_completion:
            if os.getenv("LLM_DEBUG", "0").strip() == "1":
                print(f"[LLM_DEBUG] Grounded generation failed: {primary_error}")
            return None
        try:
            return _call_chat_completion(client, model, system_prompt, user_prompt)
        except Exception as fallback_error:
            if os.getenv("LLM_DEBUG", "0").strip() == "1":
                print(f"[LLM_DEBUG] Grounded generation failed: {fallback_error or primary_error}")
            return None


def generate_llm_only_answer(question):
    """Generate an ungrounded LLM-only baseline answer for evaluation."""
    client = _create_llm_client()
    if client is None:
        return None

    _, model, base_url = _get_llm_settings()
    system_prompt = (
        "You are answering an academic assignment question without retrieval support. "
        "Respond directly and concisely."
    )
    user_prompt = question.strip()
    prefers_chat_completion = base_url.rstrip("/") != DEFAULT_BASE_URL.rstrip("/")

    try:
        if prefers_chat_completion:
            return _call_chat_completion(client, model, system_prompt, user_prompt)
        return _call_responses_api(client, model, system_prompt, user_prompt)
    except Exception as primary_error:
        if prefers_chat_completion:
            if os.getenv("LLM_DEBUG", "0").strip() == "1":
                print(f"[LLM_DEBUG] LLM-only generation failed: {primary_error}")
            return None
        try:
            return _call_chat_completion(client, model, system_prompt, user_prompt)
        except Exception as fallback_error:
            if os.getenv("LLM_DEBUG", "0").strip() == "1":
                print(f"[LLM_DEBUG] LLM-only generation failed: {fallback_error or primary_error}")
            return None


def generate_rag_answer(question, retrieved_chunks, min_confidence_score=0.23):
    """
    Generate an evidence-grounded answer from retrieved chunks.

    When the configured LLM is unavailable, the function falls back to
    deterministic template generation so the prototype remains runnable.
    """
    if not retrieved_chunks:
        return {
            "answer": "The provided documents do not contain enough information to answer this question confidently.",
            "evidence": [],
            "confidence": "Low",
            "route": "rag",
        }

    intent = _detect_answer_intent(question)
    minimum_score_by_intent = {
        "plan": 0.16,
        "policy": 0.16,
        "evaluation": 0.18,
        "rubric": 0.18,
        "general": min_confidence_score,
    }
    best_chunk = retrieved_chunks[0]
    best_score = best_chunk.get("score", 0)
    if best_score < minimum_score_by_intent.get(intent, min_confidence_score):
        return {
            "answer": "The provided documents do not contain enough evidence to answer this question confidently.",
            "evidence": retrieved_chunks,
            "confidence": "Low",
            "route": "rag",
        }

    answer = _generate_llm_answer(question, retrieved_chunks)
    generation_mode = "llm"

    if answer is None:
        if intent == "evaluation":
            answer = _generate_template_evaluation_answer(question, retrieved_chunks)
        elif intent == "plan":
            answer = _generate_template_plan_answer(question, retrieved_chunks)
        elif intent == "policy":
            answer = _generate_template_policy_answer(question, retrieved_chunks)
        else:
            answer = _generate_structured_template_answer(question, retrieved_chunks)
        generation_mode = "template"

    return {
        "answer": answer,
        "evidence": retrieved_chunks,
        "confidence": _confidence_from_score(best_score),
        "route": "rag",
        "generation_mode": generation_mode,
    }


def generate_unsupported_answer():
    """Return a cautious response for unsafe or unsupported questions."""
    return {
        "answer": (
            "The provided documents do not contain enough evidence to answer this confidently, "
            "or the request may require official approval. Please check the official course instructions "
            "or contact the subject coordinator."
        ),
        "evidence": [],
        "confidence": "Low",
        "route": "unsupported",
    }
