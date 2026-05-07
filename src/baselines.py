from answer_generator import generate_llm_only_answer, generate_rag_answer
from retrieval import SemanticRetriever, tokenize_query


class KeywordSearchBaseline:
    """A lightweight lexical baseline over the processed chunk collection."""

    def __init__(self, chunks):
        """Store chunk metadata for overlap-based ranking."""
        self.chunks = chunks

    def answer(self, question, top_k=3):
        """Return a keyword-based answer using simple lexical overlap."""
        query_tokens = tokenize_query(question)
        if not query_tokens:
            return {
                "answer": "Keyword baseline could not confidently answer this question.",
                "route": "keyword",
                "evidence": [],
                "confidence": "Low",
            }

        ranked_chunks = []
        for chunk in self.chunks:
            haystack = f"{chunk.get('section', '')} {chunk.get('text', '')}".lower()
            match_count = sum(1 for token in query_tokens if token in haystack)
            if match_count > 0:
                ranked_chunks.append((match_count / len(query_tokens), chunk))

        ranked_chunks.sort(key=lambda item: item[0], reverse=True)
        top_chunks = [chunk for _, chunk in ranked_chunks[:top_k]]

        if not top_chunks:
            return {
                "answer": "Keyword baseline could not confidently answer this question.",
                "route": "keyword",
                "evidence": [],
                "confidence": "Low",
            }

        answer = "Keyword baseline retrieved this evidence: " + top_chunks[0]["text"].replace("\n", " ")[:260]
        return {
            "answer": answer,
            "route": "keyword",
            "evidence": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "source_file": chunk["source_file"],
                    "section": chunk.get("section", ""),
                    "category": chunk.get("category", ""),
                    "score": float(score),
                    "text": chunk["text"],
                }
                for score, chunk in ranked_chunks[:top_k]
            ],
            "confidence": "Medium",
        }


class RagOnlyBaseline:
    """A retrieval-plus-generation baseline without structured rule routing."""

    def __init__(self, retriever):
        """Reuse the shared retriever so comparisons are computationally fair."""
        self.retriever = retriever

    def answer(self, question, top_k=4):
        """Answer every question through retrieval and grounded generation only."""
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        return generate_rag_answer(question, retrieved_chunks)


class LlmOnlyBaseline:
    """An ungrounded LLM baseline used for empirical comparison."""

    def answer(self, question):
        """Return a direct LLM answer without retrieval context."""
        answer = generate_llm_only_answer(question)
        if answer is None:
            answer = "LLM-only baseline could not answer because the configured provider was unavailable or returned no text."
        return {
            "answer": answer,
            "route": "llm-only",
            "evidence": [],
            "confidence": "Unknown",
            "generation_mode": "llm-only",
        }
