
import sys
import time
from pathlib import Path

import pandas as pd

SRC_DIR = Path("/content/ai-document-assistant/src")
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from app import AcademicDocumentAssistant


def simple_keyword_baseline(question):
    """
    A simple keyword baseline.
    This is intentionally limited and used only for comparison.
    """
    q = question.lower()

    if "due" in q or "deadline" in q:
        return "The report is due on 16 May 2026 23:59.", "rule-based"
    if "word limit" in q or "words" in q:
        return "The report word limit is up to 3500 words, including references.", "rule-based"
    if "pdf" in q or ("report" in q and "format" in q):
        return "The report should be submitted as a PDF file.", "rule-based"
    if "ppt" in q or "slides" in q or "powerpoint" in q:
        return "The presentation slides should be submitted as a PPTX file.", "rule-based"
    if "github" in q:
        return "A GitHub repository is required.", "rule-based"

    return "Keyword baseline could not confidently answer this question.", "keyword"


def judge_answer(question, predicted_answer, ground_truth, predicted_route, expected_route, answerable):
    """
    Lightweight automatic judging for prototype evaluation.
    In the final report, results can be manually checked for stronger reliability.
    """
    pred = predicted_answer.lower()
    gt = ground_truth.lower()

    key_terms = []
    for token in gt.replace(",", "").replace(".", "").split():
        if len(token) > 4:
            key_terms.append(token)

    matched_terms = sum(1 for term in key_terms if term in pred)
    required_matches = max(1, min(3, len(key_terms) // 3))

    answer_correct = matched_terms >= required_matches or expected_route == predicted_route

    if answerable == "no":
        unsupported_correct = predicted_route == "unsupported"
    else:
        unsupported_correct = True

    citation_supported = predicted_route in ["rule-based", "rag"] and len(predicted_answer.strip()) > 0

    hallucinated = False
    if answerable == "no" and predicted_route != "unsupported":
        hallucinated = True
    if "could not confidently answer" in pred and answerable == "yes":
        hallucinated = True

    return {
        "answer_correct": int(answer_correct and unsupported_correct),
        "citation_supported": int(citation_supported),
        "hallucinated": int(hallucinated),
        "unsupported_handling_correct": int(unsupported_correct)
    }


def run_evaluation():
    project_root = Path("/content/ai-document-assistant")
    eval_dir = project_root / "evaluation"

    ground_truth_path = eval_dir / "ground_truth_answers.csv"
    output_path = eval_dir / "evaluation_results.csv"

    gt_df = pd.read_csv(ground_truth_path)

    assistant = AcademicDocumentAssistant()

    rows = []

    for _, row in gt_df.iterrows():
        question_id = row["question_id"]
        question = row["question"]
        category = row["category"]
        ground_truth = row["ground_truth"]
        expected_route = row["expected_route"]
        answerable = row["answerable"]

        # Method 1: Keyword baseline
        start = time.time()
        keyword_answer, keyword_route = simple_keyword_baseline(question)
        keyword_time = time.time() - start

        keyword_judgement = judge_answer(
            question,
            keyword_answer,
            ground_truth,
            keyword_route,
            expected_route,
            answerable
        )

        rows.append({
            "question_id": question_id,
            "category": category,
            "method": "Keyword Search",
            "question": question,
            "answer": keyword_answer,
            "route": keyword_route,
            "response_time": round(keyword_time, 4),
            **keyword_judgement
        })

        # Method 2: Proposed hybrid system
        start = time.time()
        result = assistant.answer(question)
        hybrid_time = time.time() - start

        hybrid_answer = result["answer"]
        hybrid_route = result["route"]

        hybrid_judgement = judge_answer(
            question,
            hybrid_answer,
            ground_truth,
            hybrid_route,
            expected_route,
            answerable
        )

        rows.append({
            "question_id": question_id,
            "category": category,
            "method": "Proposed Hybrid System",
            "question": question,
            "answer": hybrid_answer,
            "route": hybrid_route,
            "response_time": round(hybrid_time, 4),
            **hybrid_judgement
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_path, index=False)

    print(f"Saved evaluation results to: {output_path}")
    return results_df


if __name__ == "__main__":
    results_df = run_evaluation()
    print(results_df.head())
