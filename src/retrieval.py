import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "is", "are",
    "was", "were", "be", "been", "being", "it", "this", "that", "with", "as", "by",
    "from", "can", "could", "should", "would", "will", "what", "which", "when", "where",
    "who", "how", "why", "do", "does", "did", "i", "we", "you", "they", "he", "she",
    "my", "our", "your", "their", "me", "us", "about", "kind", "kinds", "use", "using",
}


def tokenize_query(text):
    """Extract lightweight lexical tokens for reranking and section matching."""
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) >= 3 and token not in STOPWORDS
    }


def detect_query_focus(question):
    """Infer a coarse focus label that can boost matching chunk categories."""
    lower_question = question.lower()

    if any(token in lower_question for token in ["rubric", "hd", "high distinction", "criterion"]):
        return "rubric"
    if any(token in lower_question for token in ["presentation", "slide", "powerpoint"]):
        return "presentation"
    if any(token in lower_question for token in ["evaluate", "evaluation", "metric", "baseline", "experiment"]):
        return "evaluation"
    if any(token in lower_question for token in ["workflow", "methodology", "pipeline", "system"]):
        return "workflow"

    return "general"


def category_bonus(query_focus, chunk):
    """Return a small score bonus when a chunk matches the inferred query focus."""
    category = chunk.get("category", "")
    section = chunk.get("section", "").lower()

    if query_focus == "rubric" and "rubric" in category:
        return 0.12
    if query_focus == "presentation" and ("presentation" in category or "presentation" in section):
        return 0.10
    if query_focus == "evaluation" and ("evaluation" in category or "results" in section):
        return 0.10
    if query_focus == "workflow" and (
        category in {"system_design", "ai_methods"} or "workflow" in section or "methodology" in section
    ):
        return 0.10

    return 0.0


class SemanticRetriever:
    """
    Retrieve semantically relevant chunks with light lexical and category reranking.

    The retriever encodes every chunk once at startup and combines cosine
    similarity with simple overlap and category heuristics at query time.
    """

    def __init__(self, chunks_path=None, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Load the chunk store and build normalized embeddings for retrieval."""
        if chunks_path is None:
            chunks_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "chunks.json"
        else:
            chunks_path = Path(chunks_path)

        self.chunks_path = chunks_path
        self.model_name = model_name

        with open(self.chunks_path, "r", encoding="utf-8") as chunks_file:
            self.chunks = json.load(chunks_file)

        self.texts = [chunk["text"] for chunk in self.chunks]

        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        print(f"Encoding {len(self.texts)} chunks...")
        self.chunk_embeddings = self.model.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def retrieve(self, question, top_k=4):
        """Return the top ranked evidence chunks for a user question."""
        question_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        cosine_scores = cosine_similarity(question_embedding, self.chunk_embeddings)[0]
        candidate_count = min(len(self.chunks), max(top_k * 4, top_k))
        candidate_indices = np.argsort(cosine_scores)[::-1][:candidate_count]
        query_tokens = tokenize_query(question)
        query_focus = detect_query_focus(question)

        ranked_candidates = []
        for index in candidate_indices:
            chunk = self.chunks[index]
            chunk_text = chunk["text"].lower()
            chunk_section = chunk.get("section", "").lower()

            if not query_tokens:
                lexical_overlap = 0.0
            else:
                overlap_matches = sum(
                    1
                    for token in query_tokens
                    if token in chunk_text or token in chunk_section
                )
                lexical_overlap = overlap_matches / len(query_tokens)

            combined_score = (
                (0.76 * float(cosine_scores[index]))
                + (0.16 * lexical_overlap)
                + category_bonus(query_focus, chunk)
            )
            ranked_candidates.append((index, combined_score))

        ranked_candidates.sort(key=lambda item: item[1], reverse=True)
        top_indices = [index for index, _ in ranked_candidates[:top_k]]

        results = []
        for rank, index in enumerate(top_indices, start=1):
            chunk = self.chunks[index]
            results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "source_file": chunk["source_file"],
                "section": chunk.get("section", ""),
                "category": chunk.get("category", ""),
                "score": float(cosine_scores[index]),
                "text": chunk["text"],
            })

        return results
