
def generate_rag_answer(question, retrieved_chunks, min_confidence_score=0.25):
    """
    Generate a simple evidence-grounded answer from retrieved chunks.
    This prototype uses a template-based generator instead of an external LLM API.
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

    answer = (
        "Based on the retrieved assignment documents, the most relevant information is: "
        + best_chunk["text"].replace("\n", " ")[:700]
    )

    if best_score >= 0.50:
        confidence = "High"
    elif best_score >= 0.30:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "answer": answer,
        "evidence": retrieved_chunks,
        "confidence": confidence,
        "route": "rag"
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
