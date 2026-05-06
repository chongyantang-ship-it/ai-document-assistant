import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from answer_generator import llm_is_configured
from app import AcademicDocumentAssistant
from baselines import KeywordSearchBaseline, LlmOnlyBaseline, RagOnlyBaseline


SUMMARY_COLUMNS = [
    "answer_accuracy",
    "retrieval_hit_at_3",
    "citation_support_rate",
    "hallucination_rate",
    "unsupported_handling_accuracy",
    "average_response_time",
]


def normalize_text(text):
    """Normalize text for lenient keyword-based automatic judging."""
    return " ".join(str(text).lower().replace("\n", " ").split())


def parse_multi_value_field(raw_value):
    """Split a delimiter-separated CSV field into a clean list."""
    if pd.isna(raw_value):
        return []
    return [part.strip().lower() for part in str(raw_value).split("||") if part.strip()]


def extract_evidence_ids(result):
    """Extract chunk or fact identifiers from an answer record."""
    evidence_ids = []
    for evidence_item in result.get("evidence", []):
        if "chunk_id" in evidence_item:
            evidence_ids.append(evidence_item["chunk_id"])
        elif "field" in evidence_item:
            evidence_ids.append(evidence_item["field"])
    return evidence_ids


def looks_like_cautious_refusal(answer_text):
    """Return True when an answer explicitly signals uncertainty or refusal."""
    normalized_answer = normalize_text(answer_text)
    caution_markers = [
        "not enough evidence",
        "not enough information",
        "check the official course instructions",
        "contact the subject coordinator",
        "should not be used",
        "should not replace",
    ]
    return any(marker in normalized_answer for marker in caution_markers)


def answer_contains_required_keywords(answer_text, required_keywords):
    """Return True when every required keyword phrase appears in the answer."""
    normalized_answer = normalize_text(answer_text)
    return all(keyword in normalized_answer for keyword in required_keywords)


def judge_answer(result, ground_truth_row):
    """Score one predicted answer against lightweight but stricter heuristics."""
    answer_text = result.get("answer", "")
    predicted_route = result.get("route", "")
    expected_evidence_ids = parse_multi_value_field(ground_truth_row["expected_evidence_ids"])
    required_keywords = parse_multi_value_field(ground_truth_row["required_keywords"])
    evidence_ids = extract_evidence_ids(result)
    answerable = str(ground_truth_row["answerable"]).strip().lower() == "yes"

    retrieval_hit = int(bool(expected_evidence_ids) and any(evidence_id in evidence_ids[:3] for evidence_id in expected_evidence_ids))

    if answerable:
        answer_correct = int(answer_contains_required_keywords(answer_text, required_keywords))
        unsupported_handling_correct = 1
    else:
        unsupported_handling_correct = int(predicted_route == "unsupported" or looks_like_cautious_refusal(answer_text))
        answer_correct = unsupported_handling_correct

    has_grounded_evidence = bool(evidence_ids) and (retrieval_hit == 1 or predicted_route == "rule-based")
    citation_supported = int(answerable and answer_correct and has_grounded_evidence)

    if not answerable:
        hallucinated = int(not unsupported_handling_correct)
    else:
        hallucinated = int((not answer_correct) and (not looks_like_cautious_refusal(answer_text)))

    return {
        "answer_correct": answer_correct,
        "retrieval_hit_at_3": retrieval_hit,
        "citation_supported": citation_supported,
        "hallucinated": hallucinated,
        "unsupported_handling_correct": unsupported_handling_correct,
    }


def build_method_runners():
    """Construct the method runners used in the empirical comparison."""
    assistant = AcademicDocumentAssistant()
    keyword_baseline = KeywordSearchBaseline(assistant.retriever.chunks)
    rag_only_baseline = RagOnlyBaseline(assistant.retriever)

    runners = {
        "Keyword Search": keyword_baseline.answer,
        "RAG-only": rag_only_baseline.answer,
        "Proposed Hybrid System": assistant.answer,
    }

    enable_llm_only = os.getenv("ENABLE_LLM_ONLY_BASELINE", "0").strip() == "1"
    if enable_llm_only and llm_is_configured():
        llm_only_baseline = LlmOnlyBaseline()
        runners["LLM-only"] = llm_only_baseline.answer
    elif enable_llm_only:
        print("Skipping LLM-only baseline because no compatible LLM client is configured.")

    return runners


def write_test_questions(ground_truth_df, output_path):
    """Write the simplified test-question list used in the report appendix."""
    question_df = ground_truth_df[["question_id", "question", "category"]].copy()
    question_df.to_csv(output_path, index=False)


def summarize_results(results_df):
    """Aggregate evaluation rows into method-level and category-level summaries."""
    method_summary = (
        results_df.groupby("method", as_index=False)
        .agg(
            answer_accuracy=("answer_correct", "mean"),
            retrieval_hit_at_3=("retrieval_hit_at_3", "mean"),
            citation_support_rate=("citation_supported", "mean"),
            hallucination_rate=("hallucinated", "mean"),
            unsupported_handling_accuracy=("unsupported_handling_correct", "mean"),
            average_response_time=("response_time", "mean"),
        )
    )

    category_summary = (
        results_df.groupby(["method", "category"], as_index=False)
        .agg(
            answer_accuracy=("answer_correct", "mean"),
            retrieval_hit_at_3=("retrieval_hit_at_3", "mean"),
            citation_support_rate=("citation_supported", "mean"),
            hallucination_rate=("hallucinated", "mean"),
        )
    )

    return method_summary, category_summary


def build_failure_cases(results_df):
    """Collect incorrect or hallucinated answers for qualitative analysis."""
    failure_df = results_df[(results_df["answer_correct"] == 0) | (results_df["hallucinated"] == 1)].copy()
    return failure_df[
        [
            "question_id",
            "category",
            "method",
            "question",
            "answer",
            "route",
            "retrieval_hit_at_3",
            "hallucinated",
        ]
    ]


def save_method_comparison_chart(summary_df, output_path):
    """Save a bar chart comparing the main quantitative metrics by method."""
    metric_columns = [
        "answer_accuracy",
        "retrieval_hit_at_3",
        "citation_support_rate",
        "unsupported_handling_accuracy",
    ]
    plot_df = summary_df.set_index("method")[metric_columns]
    ax = plot_df.plot(kind="bar", figsize=(10, 5))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Method Comparison Across Core Metrics")
    ax.legend(loc="lower right")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_hallucination_chart(summary_df, output_path):
    """Save a chart focusing on hallucination rate by method."""
    ax = summary_df.plot(
        x="method",
        y="hallucination_rate",
        kind="bar",
        figsize=(8, 4),
        legend=False,
        color="#c44e52",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Hallucination Rate")
    ax.set_title("Hallucination Rate by Method")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_category_accuracy_chart(category_summary_df, output_path):
    """Save a chart showing per-category accuracy for the hybrid system."""
    hybrid_df = category_summary_df[category_summary_df["method"] == "Proposed Hybrid System"]
    ax = hybrid_df.plot(
        x="category",
        y="answer_accuracy",
        kind="bar",
        figsize=(10, 4),
        legend=False,
        color="#4c72b0",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Answer Accuracy")
    ax.set_title("Hybrid System Accuracy by Question Category")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_evaluation():
    """Run the full evaluation suite and save reusable report artefacts."""
    evaluation_dir = PROJECT_ROOT / "evaluation"
    figures_dir = PROJECT_ROOT / "report_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_path = evaluation_dir / "ground_truth_answers.csv"
    test_questions_path = evaluation_dir / "test_questions.csv"
    results_path = evaluation_dir / "evaluation_results.csv"
    summary_path = evaluation_dir / "summary_results.csv"
    category_summary_path = evaluation_dir / "category_summary_results.csv"
    failure_cases_path = evaluation_dir / "failure_cases.csv"

    ground_truth_df = pd.read_csv(ground_truth_path)
    write_test_questions(ground_truth_df, test_questions_path)

    method_runners = build_method_runners()
    rows = []
    total_questions = len(ground_truth_df)
    total_methods = len(method_runners)
    completed_runs = 0
    total_runs = total_questions * total_methods

    for question_index, (_, row) in enumerate(ground_truth_df.iterrows(), start=1):
        print(
            f"[Evaluation] Question {question_index}/{total_questions}: "
            f"{row['question_id']} | {row['category']} | {row['question']}"
        )
        for method_name, runner in method_runners.items():
            print(
                f"[Evaluation] Running {method_name} "
                f"({completed_runs + 1}/{total_runs})..."
            )
            start_time = time.time()
            result = runner(row["question"])
            response_time = time.time() - start_time

            judgement = judge_answer(result, row)
            rows.append({
                "question_id": row["question_id"],
                "category": row["category"],
                "method": method_name,
                "question": row["question"],
                "answer": result.get("answer", ""),
                "route": result.get("route", ""),
                "response_time": round(response_time, 4),
                **judgement,
            })
            completed_runs += 1

    results_df = pd.DataFrame(rows)
    results_df.to_csv(results_path, index=False)

    summary_df, category_summary_df = summarize_results(results_df)
    summary_df.to_csv(summary_path, index=False)
    category_summary_df.to_csv(category_summary_path, index=False)
    build_failure_cases(results_df).to_csv(failure_cases_path, index=False)

    save_method_comparison_chart(summary_df, figures_dir / "method_comparison_chart.png")
    save_hallucination_chart(summary_df, figures_dir / "hallucination_rate_chart.png")
    save_category_accuracy_chart(category_summary_df, figures_dir / "category_accuracy_chart.png")

    print(f"Saved evaluation results to: {results_path}")
    print(f"Saved method summary to: {summary_path}")
    print(f"Saved category summary to: {category_summary_path}")
    print(f"Saved failure cases to: {failure_cases_path}")
    return results_df, summary_df, category_summary_df


if __name__ == "__main__":
    evaluation_results_df, summary_results_df, category_results_df = run_evaluation()
    print(summary_results_df)
