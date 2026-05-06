
import os
import re
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
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


def _confidence_from_score(score):
    if score >= 0.50:
        return "High"
    if score >= 0.30:
        return "Medium"
    return "Low"


def _generate_template_answer(retrieved_chunks):
    best_chunk = retrieved_chunks[0]
    return (
        "Based on the retrieved assignment documents, the most relevant information is: "
        + best_chunk["text"].replace("\n", " ")[:700]
    )


def _question_tokens(question):
    tokens = re.findall(r"[a-zA-Z0-9]+", question.lower())
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "is", "are",
        "was", "were", "be", "been", "being", "it", "this", "that", "with", "as", "by",
        "from", "can", "could", "should", "would", "will", "what", "which", "when", "where",
        "who", "how", "why", "do", "does", "did", "i", "we", "you", "they", "he", "she",
        "my", "our", "your", "their", "me", "us", "kind", "kinds", "use", "using"
    }
    return [t for t in tokens if len(t) >= 3 and t not in stop]


def _extract_relevant_points(question, retrieved_chunks, max_points=3):
    q_tokens = _question_tokens(question)
    candidates = []
    for chunk in retrieved_chunks[:3]:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", chunk.get("text", ""))
        for sentence in sentences:
            sent = sentence.strip(" -\t")
            if len(sent) < 20:
                continue
            lower_sent = sent.lower()
            if "table of contents" in lower_sent:
                continue
            if re.match(r"^\d+(\.\d+)*\s+[a-z]", lower_sent):
                continue
            overlap = 0
            if q_tokens:
                overlap = sum(1 for t in q_tokens if t in lower_sent)

            penalty = 0.0
            if any(x in lower_sent for x in ["overview of", "introduction", "workflow and methodology"]):
                penalty = 0.3

            score = overlap + chunk.get("score", 0) - penalty
            candidates.append((score, overlap, sent))

    candidates.sort(key=lambda x: x[0], reverse=True)
    points = []
    seen = set()
    strict_overlap = bool(q_tokens)
    for _, overlap, sent in candidates:
        if strict_overlap and overlap == 0:
            continue
        compact = sent.strip()
        key = compact.lower()
        if key in seen:
            continue
        seen.add(key)
        points.append(compact[:180])
        if len(points) >= max_points:
            break
    if not points:
        for _, _, sent in candidates:
            compact = sent.strip()
            key = compact.lower()
            if key in seen:
                continue
            seen.add(key)
            points.append(compact[:180])
            if len(points) >= max_points:
                break
    return points


def _generate_structured_template_answer(question, retrieved_chunks):
    points = _extract_relevant_points(question, retrieved_chunks, max_points=3)
    if not points:
        return _generate_template_answer(retrieved_chunks)

    direct = points[0]
    bullet_lines = "\n".join([f"- {p}" for p in points[:3]])
    return f"Direct Answer: {direct}\nKey Points:\n{bullet_lines}"


def _generate_template_plan_answer(question, retrieved_chunks):
    top_chunks = retrieved_chunks[:3]
    direct_answer = (
        "Use the assignment instructions to finish in a structured order: "
        "confirm deliverables, split roles, complete implementation/evaluation, then finalize report and slides."
    )

    action_steps = [
        "Review required deliverables (report PDF, PPTX, GitHub) and confirm deadlines and formats.",
        "Split team tasks by role and complete the missing technical parts (evaluation quality, evidence-grounded answers, final checks).",
        "Prepare final presentation and report sections, then rehearse Q&A and submission checklist.",
    ]

    evidence_points = []
    for chunk in top_chunks:
        snippet = chunk.get("text", "").replace("\n", " ")[:120].strip()
        if snippet:
            evidence_points.append(f"- {snippet}...")
    if not evidence_points:
        evidence_points = ["- Retrieved assignment documents indicate submission and presentation requirements."]

    return (
        f"Direct Answer: {direct_answer}\n"
        "Action Plan:\n"
        f"1. {action_steps[0]}\n"
        f"2. {action_steps[1]}\n"
        f"3. {action_steps[2]}\n"
        "Evidence Used:\n"
        + "\n".join(evidence_points[:2])
    )


def _extract_response_text(response):
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()

    output = getattr(response, "output", None) or []
    parts = []
    for item in output:
        for content_item in getattr(item, "content", []) or []:
            if getattr(content_item, "type", "") == "output_text":
                value = getattr(content_item, "text", "")
                if value:
                    parts.append(value)
    joined = "\n".join(parts).strip()
    return joined if joined else None


def _generate_llm_answer(question, retrieved_chunks):
    if OpenAI is None:
        return None

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    client = OpenAI(api_key=api_key)

    evidence_lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        evidence_lines.append(
            f"[{i}] source={chunk.get('source_file', 'unknown')} "
            f"chunk_id={chunk.get('chunk_id', 'unknown')} "
            f"score={round(chunk.get('score', 0), 4)}\n{chunk.get('text', '')}"
        )
    evidence_block = "\n\n".join(evidence_lines)

    q_lower = question.lower()
    wants_plan = any(
        token in q_lower
        for token in ["plan", "steps", "how do i", "how should", "roadmap", "finish"]
    )

    system_prompt = (
        "You are an academic document assistant. "
        "Answer only using the provided evidence. "
        "If evidence is insufficient, say so clearly. "
        "Do not invent requirements, dates, or policies."
    )
    if wants_plan:
        format_instruction = (
            "Return the response in this exact structure:\n"
            "Direct Answer: <1-2 lines>\n"
            "Action Plan:\n"
            "1. <step>\n"
            "2. <step>\n"
            "3. <step>\n"
            "Evidence Used:\n"
            "- <short bullet from evidence>\n"
            "- <short bullet from evidence>\n"
            "Keep it concise and practical."
        )
    else:
        format_instruction = (
            "Return the response in this exact structure:\n"
            "Direct Answer: <1-3 lines>\n"
            "Key Points:\n"
            "- <point>\n"
            "- <point>\n"
            "Evidence Used:\n"
            "- <short bullet from evidence>\n"
            "- <short bullet from evidence>\n"
            "Keep it concise."
        )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"{format_instruction}"
    )

    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        answer = _extract_response_text(response)
        return answer if answer else None
    except Exception as exc:
        # Fallback for accounts/models where chat.completions is more compatible.
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            answer = completion.choices[0].message.content.strip()
            return answer if answer else None
        except Exception as fallback_exc:
            exc = fallback_exc
        if os.getenv("LLM_DEBUG", "0").strip() == "1":
            print(f"[LLM_DEBUG] OpenAI call failed: {exc}")
        return None


def generate_rag_answer(question, retrieved_chunks, min_confidence_score=0.25):
    """
    Generate evidence-grounded answers from retrieved chunks.
    Uses OpenAI when OPENAI_API_KEY is set; otherwise falls back to template mode.
    """

    if not retrieved_chunks:
        return {
            "answer": "The provided documents do not contain enough information to answer this question confidently.",
            "evidence": [],
            "confidence": "Low",
            "route": "rag"
        }

    best_chunk = retrieved_chunks[0]
    best_score = best_chunk.get("score", 0)

    if best_score < min_confidence_score:
        return {
            "answer": "The provided documents do not contain enough evidence to answer this question confidently.",
            "evidence": retrieved_chunks,
            "confidence": "Low",
            "route": "rag"
        }

    q_lower = question.lower()
    wants_plan = any(
        token in q_lower
        for token in ["plan", "steps", "how do i", "how should", "roadmap", "finish"]
    )

    answer = _generate_llm_answer(question, retrieved_chunks)
    generation_mode = "llm"
    if answer is None:
        if wants_plan:
            answer = _generate_template_plan_answer(question, retrieved_chunks)
        else:
            answer = _generate_structured_template_answer(question, retrieved_chunks)
        generation_mode = "template"

    confidence = _confidence_from_score(best_score)

    return {
        "answer": answer,
        "evidence": retrieved_chunks,
        "confidence": confidence,
        "route": "rag",
        "generation_mode": generation_mode
    }


def generate_unsupported_answer():
    """
    Generate a cautious answer when the documents do not support the request
    or when the question asks for inappropriate academic behaviour.
    """
    return {
        "answer": (
            "The provided documents do not contain enough evidence to answer this confidently, "
            "or the request may require official approval. Please check the official course instructions "
            "or contact the subject coordinator."
        ),
        "evidence": [],
        "confidence": "Low",
        "route": "unsupported"
    }
