
import json
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "is", "are",
    "was", "were", "be", "been", "being", "it", "this", "that", "with", "as", "by",
    "from", "can", "could", "should", "would", "will", "what", "which", "when", "where",
    "who", "how", "why", "do", "does", "did", "i", "we", "you", "they", "he", "she",
    "my", "our", "your", "their", "me", "us", "about", "kind", "kinds", "use", "using"
}


class SemanticRetriever:
    """
    Semantic embedding retriever.
    It converts document chunks and user questions into embedding vectors,
    then retrieves the most relevant chunks using cosine similarity.
    """

    def __init__(
        self,
        chunks_path=None,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        if chunks_path is None:
            chunks_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "chunks.json"
        else:
            chunks_path = Path(chunks_path)

        self.chunks_path = chunks_path
        self.model_name = model_name

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.texts = [chunk["text"] for chunk in self.chunks]

        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        print(f"Encoding {len(self.texts)} chunks...")
        self.chunk_embeddings = self.model.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def retrieve(self, question, top_k=3):
        """
        Retrieve top_k most relevant chunks for a user question.
        """
        question_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores = cosine_similarity(question_embedding, self.chunk_embeddings)[0]

        # Hybrid reranking: semantic similarity + lightweight lexical overlap.
        candidate_k = min(len(self.chunks), max(top_k * 3, top_k))
        candidate_indices = np.argsort(scores)[::-1][:candidate_k]
        query_tokens = {
            token for token in re.findall(r"[a-zA-Z0-9]+", question.lower())
            if len(token) >= 3 and token not in STOPWORDS
        }

        ranked = []
        for idx in candidate_indices:
            text = self.chunks[idx]["text"].lower()
            if not query_tokens:
                overlap = 0.0
            else:
                match_count = sum(1 for token in query_tokens if token in text)
                overlap = match_count / len(query_tokens)

            combined_score = (0.8 * float(scores[idx])) + (0.2 * overlap)
            ranked.append((idx, combined_score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        top_indices = [idx for idx, _ in ranked[:top_k]]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "source_file": chunk["source_file"],
                "section": chunk.get("section", ""),
                "category": chunk.get("category", ""),
                "score": float(scores[idx]),
                "text": chunk["text"]
            })

        return results
