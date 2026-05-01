
import sys
from pathlib import Path

SRC_DIR = Path("/content/ai-document-assistant/src")
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from rule_checker import load_facts, check_rule_based_answer
from retrieval import SemanticRetriever
from query_router import route_query
from answer_generator import generate_rag_answer, generate_unsupported_answer


class AcademicDocumentAssistant:
    """
    Hybrid AI academic document assistant.
    It combines:
    - structured knowledge representation
    - rule-based reasoning
    - semantic embedding retrieval
    - evidence-grounded answer generation
    """

    def __init__(self):
        self.facts = load_facts()
        self.retriever = SemanticRetriever()

    def answer(self, question, top_k=3):
        route = route_query(question)

        if route == "unsupported":
            return generate_unsupported_answer()

        if route == "rule":
            rule_result = check_rule_based_answer(question, self.facts)
            if rule_result is not None:
                return rule_result

            # fallback to RAG if no rule matched
            retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
            return generate_rag_answer(question, retrieved_chunks)

        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        return generate_rag_answer(question, retrieved_chunks)


def print_answer(question, result):
    print("\n" + "=" * 80)
    print("Question:", question)
    print("Route:", result.get("route"))
    print("Confidence:", result.get("confidence"))

    print("\nAnswer:")
    print(result.get("answer"))

    print("\nEvidence:")
    evidence = result.get("evidence", [])

    if not evidence:
        print("No evidence available.")
    else:
        for i, ev in enumerate(evidence, start=1):
            if "chunk_id" in ev:
                print(f"{i}. {ev.get('chunk_id')} | {ev.get('source_file')} | score={round(ev.get('score', 0), 4)}")
                print("   " + ev.get("text", "")[:300].replace("\n", " "))
            else:
                print(f"{i}. {ev.get('source')} | {ev.get('field')}")
                print("   " + ev.get("text", ""))


if __name__ == "__main__":
    assistant = AcademicDocumentAssistant()

    print("AI Academic Document Assistant")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nAsk a question: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        result = assistant.answer(question)
        print_answer(question, result)
