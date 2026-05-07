import hashlib
import json
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
from chunking import create_chunks
from fact_extractor import save_structured_facts
from query_router import route_query


TRUTHY_VALUES = {"1", "true", "yes", "on"}
ROUTE_ALIASES = {
    "rule": "rule-based",
    "rule-based": "rule-based",
    "rag": "rag",
    "unsupported": "unsupported",
    "keyword": "keyword",
    "llm-only": "llm-only",
}
MODE_INCLUDE_COLUMNS = {
    "smoke": "include_in_smoke_run",
    "core": "include_in_core_run",
    "report": "include_in_report_run",
}
EVALUATION_OPT_IN_ENV = "ALLOW_EVALUATION_RUN"
EVALUATION_MODE_ENV = "EVALUATION_MODE"
REQUEST_ESTIMATE_SECONDS_ENV = "EVALUATION_REQUEST_ESTIMATE_SECONDS"
FINAL_ANSWER_SUMMARY_COLUMNS = [
    "answer_accuracy",
    "evidence_support_rate",
    "hallucination_rate",
    "unsupported_handling_accuracy",
    "average_response_time",
]


def normalize_text(text):
    """Normalize text for lenient automatic judging."""
    return " ".join(str(text).lower().replace("\n", " ").split())


def evaluation_is_explicitly_allowed():
    """Return True only when the caller explicitly opts in to a slow evaluation run."""
    return os.getenv(EVALUATION_OPT_IN_ENV, "0").strip().lower() in TRUTHY_VALUES


def parse_multi_value_field(raw_value):
    """Split a delimiter-separated field into a clean list."""
    if pd.isna(raw_value):
        return []
    return [part.strip() for part in str(raw_value).split("||") if part.strip()]


def parse_bool_field(raw_value, default=False):
    """Interpret a CSV field as a boolean flag."""
    if pd.isna(raw_value):
        return default
    return str(raw_value).strip().lower() in TRUTHY_VALUES


def parse_int_field(raw_value, default):
    """Interpret a CSV field as an integer, with a fallback default."""
    if pd.isna(raw_value) or str(raw_value).strip() == "":
        return default
    try:
        return int(str(raw_value).strip())
    except ValueError:
        return default


def normalize_route_name(route_name):
    """Map route aliases to a stable label for scoring."""
    normalized_route = str(route_name or "").strip().lower()
    return ROUTE_ALIASES.get(normalized_route, normalized_route)


def get_request_estimate_seconds():
    """Return the assumed per-request cooldown used for workload estimates."""
    raw_value = os.getenv(REQUEST_ESTIMATE_SECONDS_ENV, "8").strip()
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return 8.0


def resolve_mode(mode=None):
    """Resolve the evaluation mode to one of smoke, core, or report."""
    resolved_mode = (mode or os.getenv(EVALUATION_MODE_ENV, "report")).strip().lower()
    if resolved_mode not in MODE_INCLUDE_COLUMNS:
        raise ValueError(f"Unsupported evaluation mode: {resolved_mode}")
    return resolved_mode


def load_question_benchmark(question_path):
    """Load the master question bank for all evaluation layers."""
    return pd.read_csv(question_path)


def load_fact_ground_truth(ground_truth_path):
    """Load the field-level fact extraction ground truth file."""
    with open(ground_truth_path, "r", encoding="utf-8") as ground_truth_file:
        return json.load(ground_truth_file)


def load_retrieval_ground_truth(retrieval_path):
    """Load retrieval expectations keyed by benchmark question id."""
    with open(retrieval_path, "r", encoding="utf-8") as retrieval_file:
        payload = json.load(retrieval_file)
    return {item["question_id"]: item for item in payload}


def select_questions_for_mode(question_df, mode):
    """Select the question subset included in the requested evaluation mode."""
    include_column = MODE_INCLUDE_COLUMNS[mode]
    subset = question_df[question_df[include_column].apply(parse_bool_field)].copy()
    subset.reset_index(drop=True, inplace=True)
    return subset


def write_test_questions(question_df, output_path):
    """Write the simplified master question list used in the report appendix."""
    appendix_df = question_df[
        ["question_id", "question", "category", "question_type", "expected_route", "include_in_report_run"]
    ].copy()
    appendix_df.to_csv(output_path, index=False)


def estimate_evaluation_workload(question_df, mode="report", include_llm_only=False, use_llm_extraction=True):
    """Estimate how many live API requests a cold evaluation run would make."""
    selected_questions = select_questions_for_mode(question_df, mode)
    interpretive_questions = selected_questions[selected_questions["question_type"] == "interpretive"]
    llm_only_question_ids = set(
        selected_questions[selected_questions["include_in_llm_only_subset"].apply(parse_bool_field)]["question_id"].tolist()
    )
    selected_question_ids = set(selected_questions["question_id"].tolist())

    rag_only_requests = 0 if mode == "smoke" else len(selected_questions)
    hybrid_requests = 0 if mode == "smoke" else len(interpretive_questions)
    llm_only_requests = len(llm_only_question_ids) if include_llm_only and mode == "report" else 0
    extraction_requests = 1 if use_llm_extraction else 0

    api_backed_question_ids = set()
    if rag_only_requests > 0:
        api_backed_question_ids.update(selected_question_ids)
    elif hybrid_requests > 0:
        api_backed_question_ids.update(interpretive_questions["question_id"].tolist())
    if llm_only_requests > 0:
        api_backed_question_ids.update(llm_only_question_ids)

    request_estimate_seconds = get_request_estimate_seconds()
    total_api_requests = extraction_requests + rag_only_requests + hybrid_requests + llm_only_requests
    estimated_seconds = total_api_requests * request_estimate_seconds

    return {
        "mode": mode,
        "selected_question_count": len(selected_questions),
        "selected_interpretive_question_count": len(interpretive_questions),
        "api_backed_question_count": len(api_backed_question_ids),
        "fact_extraction_requests": extraction_requests,
        "rag_only_requests": rag_only_requests,
        "hybrid_generation_requests": hybrid_requests,
        "llm_only_requests": llm_only_requests,
        "total_api_requests": total_api_requests,
        "request_estimate_seconds": request_estimate_seconds,
        "estimated_total_seconds": round(estimated_seconds, 2),
        "estimated_total_minutes": round(estimated_seconds / 60.0, 2),
    }


def save_workload_summary(output_path, workload_summary):
    """Persist the workload estimate so report numbers stay reproducible."""
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(workload_summary, output_file, ensure_ascii=False, indent=2)


def prepare_runtime_assets(use_llm_extraction):
    """Rebuild the processed chunks and structured facts for the current evaluation run."""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = processed_dir / "chunks.json"
    facts_path = processed_dir / "structured_facts.json"
    create_chunks(chunks_path)
    return save_structured_facts(facts_path, use_llm=use_llm_extraction)


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
        "check the rules for whether and how you can use it",
        "should not be used",
        "should not replace",
    ]
    return any(marker in normalized_answer for marker in caution_markers)


def count_required_keyword_hits(answer_text, required_keywords):
    """Count how many expected keyword phrases appear in the answer."""
    normalized_answer = normalize_text(answer_text)
    return sum(1 for keyword in required_keywords if normalize_text(keyword) in normalized_answer)


def get_keyword_target(benchmark_row, required_keywords):
    """Return the number of keyword hits required for a correct answer."""
    if not required_keywords:
        return 0

    explicit_target = parse_int_field(benchmark_row.get("minimum_keyword_hits"), default=-1)
    if explicit_target >= 0:
        return min(len(required_keywords), explicit_target)

    if len(required_keywords) <= 3:
        return len(required_keywords)

    return max(2, len(required_keywords) - 1)


def judge_final_answer(result, benchmark_row):
    """Score one answer against type-aware correctness and grounding rules."""
    answer_text = result.get("answer", "")
    predicted_route = normalize_route_name(result.get("route", ""))
    acceptable_evidence_ids = parse_multi_value_field(benchmark_row["acceptable_evidence_ids"])
    required_keywords = parse_multi_value_field(benchmark_row["required_keywords"])
    evidence_ids = extract_evidence_ids(result)
    answerable = str(benchmark_row["answerable"]).strip().lower() == "yes"
    cautious_refusal = looks_like_cautious_refusal(answer_text)
    keyword_hit_count = count_required_keyword_hits(answer_text, required_keywords)
    keyword_target = get_keyword_target(benchmark_row, required_keywords)
    evidence_supported = int(bool(acceptable_evidence_ids) and any(evidence_id in acceptable_evidence_ids for evidence_id in evidence_ids))

    if answerable:
        content_correct = int(keyword_hit_count >= keyword_target and not cautious_refusal)
        answer_correct = content_correct
        unsupported_handling_correct = 1
    else:
        unsupported_handling_correct = int(predicted_route == "unsupported" or cautious_refusal)
        content_correct = unsupported_handling_correct
        answer_correct = unsupported_handling_correct

    if not answerable:
        hallucinated = int(not unsupported_handling_correct)
    else:
        hallucinated = int((not content_correct) and (not cautious_refusal))

    return {
        "answer_correct": answer_correct,
        "content_correct": content_correct,
        "keyword_hit_count": keyword_hit_count,
        "required_keyword_count": len(required_keywords),
        "required_keyword_target": keyword_target,
        "evidence_supported": evidence_supported,
        "hallucinated": hallucinated,
        "unsupported_handling_correct": unsupported_handling_correct,
    }


def build_cache_key(method_name, question):
    """Create a stable cache key for one method-question pair."""
    payload = {
        "cache_version": os.getenv("EVALUATION_CACHE_VERSION", "2026-05-four-layer-v1"),
        "method": method_name,
        "question": question,
        "model": os.getenv("OPENAI_MODEL", ""),
        "base_url": os.getenv("OPENAI_BASE_URL", ""),
    }
    serialized_payload = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()


def load_response_cache(cache_path):
    """Load cached method outputs from disk when available."""
    if not cache_path.exists():
        return {}

    try:
        with open(cache_path, "r", encoding="utf-8") as cache_file:
            return json.load(cache_file)
    except (OSError, json.JSONDecodeError):
        return {}


def save_response_cache(cache_path, cache_data):
    """Persist the response cache to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as cache_file:
        json.dump(cache_data, cache_file, ensure_ascii=False, indent=2)


def run_method_with_cache(method_name, runner, question, cache_data, cache_path):
    """Run a method or reuse its cached output when available."""
    cache_disabled = os.getenv("EVALUATION_DISABLE_CACHE", "0").strip().lower() in TRUTHY_VALUES
    force_refresh = os.getenv("EVALUATION_REFRESH_CACHE", "0").strip().lower() in TRUTHY_VALUES
    cache_key = build_cache_key(method_name, question)

    if not cache_disabled and not force_refresh and cache_key in cache_data:
        cached_entry = cache_data[cache_key]
        return cached_entry["result"], float(cached_entry.get("response_time", 0.0)), True

    start_time = time.time()
    result = runner(question)
    response_time = time.time() - start_time

    if not cache_disabled:
        cache_data[cache_key] = {
            "method": method_name,
            "question": question,
            "response_time": round(response_time, 4),
            "result": result,
        }
        save_response_cache(cache_path, cache_data)

    return result, response_time, False


def _list_contains_all(actual_items, expected_items):
    """Return True when every expected item is covered by the extracted list."""
    normalized_actual_items = [normalize_text(item) for item in actual_items]
    return all(
        any(normalize_text(expected_item) in actual_item for actual_item in normalized_actual_items)
        for expected_item in expected_items
    )


def _dict_contains_all(actual_mapping, expected_pairs):
    """Return True when all expected key-value pairs are present in the extracted dict."""
    normalized_actual_mapping = {str(key).lower(): normalize_text(value) for key, value in actual_mapping.items()}
    for expected_key, expected_value in expected_pairs.items():
        actual_value = normalized_actual_mapping.get(str(expected_key).lower(), "")
        if normalize_text(expected_value) not in actual_value:
            return False
    return True


def _rubric_titles_contains_all(actual_sections, expected_items):
    """Return True when all expected rubric titles were extracted."""
    actual_titles = [normalize_text(item.get("title", "")) for item in actual_sections if isinstance(item, dict)]
    return all(
        any(normalize_text(expected_title) in actual_title for actual_title in actual_titles)
        for expected_title in expected_items
    )


def fact_field_matches(record, field_spec):
    """Evaluate whether one extracted fact record matches the ground truth specification."""
    value = record.get("value")
    match_mode = field_spec.get("match_mode")

    if match_mode == "boolean":
        return value is field_spec.get("expected_value")
    if match_mode == "list_contains_all":
        if not isinstance(value, list):
            return False
        return _list_contains_all(value, field_spec.get("expected_items", []))
    if match_mode == "dict_contains_all":
        if not isinstance(value, dict):
            return False
        return _dict_contains_all(value, field_spec.get("expected_pairs", {}))
    if match_mode == "rubric_titles_contains_all":
        if not isinstance(value, list):
            return False
        return _rubric_titles_contains_all(value, field_spec.get("expected_items", []))
    if match_mode == "keywords":
        normalized_value = normalize_text(value)
        return all(normalize_text(keyword) in normalized_value for keyword in field_spec.get("expected_keywords", []))
    return False


def evaluate_fact_extraction(extracted_facts, fact_ground_truth):
    """Compute Layer 1 metrics over the structured fact extraction output."""
    field_records = extracted_facts.get("field_records", {})
    rows = []

    for field_spec in fact_ground_truth["fields"]:
        field_name = field_spec["field"]
        record = field_records.get(field_name, {})
        value = record.get("value")
        evidence = record.get("evidence", "")
        missing = int(value in (None, "", [], {}))
        correct = int(fact_field_matches(record, field_spec))
        evidence_supported = int(bool(evidence)) if field_spec.get("require_evidence", True) else 1
        hallucinated = int((not missing) and (not correct))

        rows.append({
            "field": field_name,
            "category": field_spec["category"],
            "validation_status": record.get("validation_status", "missing"),
            "field_accuracy": correct,
            "missing_field": missing,
            "evidence_supported": evidence_supported,
            "hallucinated_fact": hallucinated,
            "confidence": record.get("confidence", 0.0),
            "extraction_method": record.get("extraction_method", ""),
            "source_section": record.get("source_section", ""),
            "value_preview": json.dumps(value, ensure_ascii=False)[:280],
        })

    results_df = pd.DataFrame(rows)
    category_summary = (
        results_df.groupby("category", as_index=False)
        .agg(
            field_accuracy=("field_accuracy", "mean"),
            missing_field_rate=("missing_field", "mean"),
            evidence_support_rate=("evidence_supported", "mean"),
            hallucinated_fact_rate=("hallucinated_fact", "mean"),
            average_confidence=("confidence", "mean"),
        )
    )
    overall_row = pd.DataFrame([{
        "category": "overall",
        "field_accuracy": results_df["field_accuracy"].mean(),
        "missing_field_rate": results_df["missing_field"].mean(),
        "evidence_support_rate": results_df["evidence_supported"].mean(),
        "hallucinated_fact_rate": results_df["hallucinated_fact"].mean(),
        "average_confidence": results_df["confidence"].mean(),
    }])
    summary_df = pd.concat([category_summary, overall_row], ignore_index=True)
    return results_df, summary_df


def compute_macro_f1(routing_df):
    """Compute macro-F1 for the router across the three expected route classes."""
    labels = ["rule-based", "rag", "unsupported"]
    f1_scores = []

    for label in labels:
        true_positives = len(routing_df[(routing_df["expected_route"] == label) & (routing_df["predicted_route"] == label)])
        false_positives = len(routing_df[(routing_df["expected_route"] != label) & (routing_df["predicted_route"] == label)])
        false_negatives = len(routing_df[(routing_df["expected_route"] == label) & (routing_df["predicted_route"] != label)])

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append((2 * precision * recall) / (precision + recall))

    return sum(f1_scores) / len(f1_scores)


def evaluate_routing(question_df):
    """Compute Layer 2 routing metrics for the hybrid router only."""
    rows = []
    for _, row in question_df.iterrows():
        predicted_route = normalize_route_name(route_query(row["question"]))
        expected_route = normalize_route_name(row["expected_route"])
        rows.append({
            "question_id": row["question_id"],
            "category": row["category"],
            "question_type": row["question_type"],
            "question": row["question"],
            "expected_route": expected_route,
            "predicted_route": predicted_route,
            "route_correct": int(predicted_route == expected_route),
        })

    results_df = pd.DataFrame(rows)
    question_type_summary = (
        results_df.groupby("question_type", as_index=False)
        .agg(routing_accuracy=("route_correct", "mean"))
    )
    overall_summary = pd.DataFrame([{
        "question_type": "overall",
        "routing_accuracy": results_df["route_correct"].mean(),
        "macro_f1": compute_macro_f1(results_df),
    }])
    question_type_summary["macro_f1"] = None
    summary_df = pd.concat([question_type_summary, overall_summary], ignore_index=True)
    return results_df, summary_df


def evaluate_retrieval(question_df, retrieval_ground_truth, retriever):
    """Compute Layer 3 retrieval metrics on questions with annotated evidence."""
    rows = []
    for _, row in question_df.iterrows():
        retrieval_spec = retrieval_ground_truth.get(row["question_id"])
        if retrieval_spec is None:
            continue

        acceptable_ids = retrieval_spec["acceptable_evidence_ids"]
        retrieved_chunks = retriever.retrieve(row["question"], top_k=5)
        retrieved_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]

        top1_hit = int(bool(retrieved_ids) and retrieved_ids[0] in acceptable_ids)
        recall_at_3 = int(any(chunk_id in acceptable_ids for chunk_id in retrieved_ids[:3]))
        recall_at_5 = int(any(chunk_id in acceptable_ids for chunk_id in retrieved_ids[:5]))

        reciprocal_rank = 0.0
        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in acceptable_ids:
                reciprocal_rank = 1.0 / rank
                break

        rows.append({
            "question_id": row["question_id"],
            "category": row["category"],
            "question": row["question"],
            "top1_accuracy": top1_hit,
            "recall_at_3": recall_at_3,
            "recall_at_5": recall_at_5,
            "mrr": reciprocal_rank,
            "retrieved_ids": " || ".join(retrieved_ids),
        })

    if not rows:
        results_df = pd.DataFrame(
            columns=["question_id", "category", "question", "top1_accuracy", "recall_at_3", "recall_at_5", "mrr", "retrieved_ids"]
        )
        summary_df = pd.DataFrame([{
            "category": "overall",
            "top1_accuracy": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
        }])
        return results_df, summary_df

    results_df = pd.DataFrame(rows)
    category_summary = (
        results_df.groupby("category", as_index=False)
        .agg(
            top1_accuracy=("top1_accuracy", "mean"),
            recall_at_3=("recall_at_3", "mean"),
            recall_at_5=("recall_at_5", "mean"),
            mrr=("mrr", "mean"),
        )
    )
    overall_row = pd.DataFrame([{
        "category": "overall",
        "top1_accuracy": results_df["top1_accuracy"].mean(),
        "recall_at_3": results_df["recall_at_3"].mean(),
        "recall_at_5": results_df["recall_at_5"].mean(),
        "mrr": results_df["mrr"].mean(),
    }])
    summary_df = pd.concat([category_summary, overall_row], ignore_index=True)
    return results_df, summary_df


def build_method_runners(assistant, mode):
    """Construct the final-answer method runners used in the empirical comparison."""
    keyword_baseline = KeywordSearchBaseline(assistant.retriever.chunks)
    rag_only_baseline = RagOnlyBaseline(assistant.retriever)

    runners = {
        "Keyword Search": keyword_baseline.answer,
        "Proposed Hybrid System": assistant.answer,
    }

    if mode != "smoke":
        runners["RAG-only"] = rag_only_baseline.answer

    enable_llm_only = os.getenv("ENABLE_LLM_ONLY_BASELINE", "0").strip().lower() in TRUTHY_VALUES
    if mode == "report" and enable_llm_only and llm_is_configured():
        llm_only_baseline = LlmOnlyBaseline()
        runners["LLM-only"] = llm_only_baseline.answer
    elif mode == "report" and enable_llm_only:
        print("Skipping LLM-only baseline because no compatible LLM client is configured.")

    return runners


def summarize_final_answer_results(results_df):
    """Aggregate final-answer rows into method-level and category-level summaries."""
    method_summary = (
        results_df.groupby("method", as_index=False)
        .agg(
            answer_accuracy=("answer_correct", "mean"),
            evidence_support_rate=("evidence_supported", "mean"),
            hallucination_rate=("hallucinated", "mean"),
            unsupported_handling_accuracy=("unsupported_handling_correct", "mean"),
            average_response_time=("response_time", "mean"),
        )
    )

    category_summary = (
        results_df.groupby(["method", "category"], as_index=False)
        .agg(
            answer_accuracy=("answer_correct", "mean"),
            evidence_support_rate=("evidence_supported", "mean"),
            hallucination_rate=("hallucinated", "mean"),
        )
    )

    return method_summary, category_summary


def build_failure_cases(results_df):
    """Collect incorrect or hallucinated answers for qualitative analysis."""
    failure_df = results_df[
        (results_df["answer_correct"] == 0)
        | (results_df["hallucinated"] == 1)
        | (results_df["unsupported_handling_correct"] == 0)
    ].copy()
    return failure_df[
        [
            "question_id",
            "category",
            "method",
            "question",
            "answer",
            "route",
            "keyword_hit_count",
            "required_keyword_target",
            "evidence_supported",
            "hallucinated",
            "used_cache",
        ]
    ]


def save_method_comparison_chart(summary_df, output_path):
    """Save a bar chart comparing the main final-answer metrics by method."""
    metric_columns = [
        "answer_accuracy",
        "evidence_support_rate",
        "unsupported_handling_accuracy",
    ]
    plot_df = summary_df.set_index("method")[metric_columns]
    ax = plot_df.plot(kind="bar", figsize=(10, 5))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Final Answer Comparison Across Core Metrics")
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


def run_final_answer_evaluation(question_df, mode, method_runners, cache_path):
    """Compute Layer 4 final-answer results across the selected methods."""
    response_cache = load_response_cache(cache_path)
    rows = []
    total_questions = len(question_df)
    llm_only_subset_ids = set(
        question_df[question_df["include_in_llm_only_subset"].apply(parse_bool_field)]["question_id"].tolist()
    )

    expanded_run_count = 0
    for method_name in method_runners:
        if method_name == "LLM-only":
            expanded_run_count += len(llm_only_subset_ids)
        else:
            expanded_run_count += total_questions

    completed_runs = 0
    for question_index, (_, row) in enumerate(question_df.iterrows(), start=1):
        print(
            f"[Evaluation] Question {question_index}/{total_questions}: "
            f"{row['question_id']} | {row['category']} | {row['question']}"
        )
        for method_name, runner in method_runners.items():
            if method_name == "LLM-only" and row["question_id"] not in llm_only_subset_ids:
                continue

            print(f"[Evaluation] Running {method_name} ({completed_runs + 1}/{expanded_run_count})...")
            result, response_time, used_cache = run_method_with_cache(
                method_name,
                runner,
                row["question"],
                response_cache,
                cache_path,
            )
            judgement = judge_final_answer(result, row)
            rows.append({
                "question_id": row["question_id"],
                "category": row["category"],
                "question_type": row["question_type"],
                "method": method_name,
                "question": row["question"],
                "answer": result.get("answer", ""),
                "route": normalize_route_name(result.get("route", "")),
                "response_time": round(response_time, 4),
                "used_cache": int(used_cache),
                **judgement,
            })
            completed_runs += 1

    results_df = pd.DataFrame(rows)
    summary_df, category_summary_df = summarize_final_answer_results(results_df)
    return results_df, summary_df, category_summary_df


def run_evaluation(mode=None):
    """Run the full four-layer evaluation suite and save report-ready artefacts."""
    if not evaluation_is_explicitly_allowed():
        raise RuntimeError(
            f"Evaluation is disabled by default. Set {EVALUATION_OPT_IN_ENV}=1 only after explicit user approval."
        )

    mode = resolve_mode(mode)
    use_llm_extraction = mode != "smoke"

    evaluation_dir = PROJECT_ROOT / "evaluation"
    figures_dir = PROJECT_ROOT / "report_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fact_ground_truth_path = evaluation_dir / "fact_ground_truth.json"
    question_benchmark_path = evaluation_dir / "question_benchmark.csv"
    retrieval_ground_truth_path = evaluation_dir / "retrieval_ground_truth.json"
    test_questions_path = evaluation_dir / "test_questions.csv"
    cache_path = evaluation_dir / "response_cache.json"
    workload_summary_path = evaluation_dir / "evaluation_workload_summary.json"

    fact_results_path = evaluation_dir / "fact_extraction_results.csv"
    fact_summary_path = evaluation_dir / "fact_extraction_summary.csv"
    routing_results_path = evaluation_dir / "routing_results.csv"
    routing_summary_path = evaluation_dir / "routing_summary.csv"
    retrieval_results_path = evaluation_dir / "retrieval_results.csv"
    retrieval_summary_path = evaluation_dir / "retrieval_summary.csv"
    final_results_path = evaluation_dir / "final_answer_results.csv"
    final_summary_path = evaluation_dir / "final_answer_summary.csv"
    final_category_summary_path = evaluation_dir / "final_answer_category_summary.csv"
    failure_cases_path = evaluation_dir / "failure_cases.csv"

    compatibility_results_path = evaluation_dir / "evaluation_results.csv"
    compatibility_summary_path = evaluation_dir / "summary_results.csv"
    compatibility_category_summary_path = evaluation_dir / "category_summary_results.csv"

    question_df = load_question_benchmark(question_benchmark_path)
    selected_questions_df = select_questions_for_mode(question_df, mode)
    fact_ground_truth = load_fact_ground_truth(fact_ground_truth_path)
    retrieval_ground_truth = load_retrieval_ground_truth(retrieval_ground_truth_path)
    write_test_questions(question_df, test_questions_path)

    method_runners_llm_only_enabled = (
        mode == "report" and os.getenv("ENABLE_LLM_ONLY_BASELINE", "0").strip().lower() in TRUTHY_VALUES
    )
    workload_summary = estimate_evaluation_workload(
        question_df,
        mode=mode,
        include_llm_only=method_runners_llm_only_enabled,
        use_llm_extraction=use_llm_extraction,
    )
    save_workload_summary(workload_summary_path, workload_summary)

    extracted_facts = prepare_runtime_assets(use_llm_extraction=use_llm_extraction)
    fact_results_df, fact_summary_df = evaluate_fact_extraction(extracted_facts, fact_ground_truth)
    fact_results_df.to_csv(fact_results_path, index=False)
    fact_summary_df.to_csv(fact_summary_path, index=False)

    routing_results_df, routing_summary_df = evaluate_routing(selected_questions_df)
    routing_results_df.to_csv(routing_results_path, index=False)
    routing_summary_df.to_csv(routing_summary_path, index=False)

    assistant = AcademicDocumentAssistant()
    retrieval_results_df, retrieval_summary_df = evaluate_retrieval(
        selected_questions_df,
        retrieval_ground_truth,
        assistant.retriever,
    )
    retrieval_results_df.to_csv(retrieval_results_path, index=False)
    retrieval_summary_df.to_csv(retrieval_summary_path, index=False)

    method_runners = build_method_runners(assistant, mode)
    final_results_df, final_summary_df, final_category_summary_df = run_final_answer_evaluation(
        selected_questions_df,
        mode,
        method_runners,
        cache_path,
    )
    final_results_df.to_csv(final_results_path, index=False)
    final_summary_df.to_csv(final_summary_path, index=False)
    final_category_summary_df.to_csv(final_category_summary_path, index=False)
    build_failure_cases(final_results_df).to_csv(failure_cases_path, index=False)

    final_results_df.to_csv(compatibility_results_path, index=False)
    final_summary_df.to_csv(compatibility_summary_path, index=False)
    final_category_summary_df.to_csv(compatibility_category_summary_path, index=False)

    save_method_comparison_chart(final_summary_df, figures_dir / "method_comparison_chart.png")
    save_hallucination_chart(final_summary_df, figures_dir / "hallucination_rate_chart.png")
    save_category_accuracy_chart(final_category_summary_df, figures_dir / "category_accuracy_chart.png")

    print(f"Saved workload summary to: {workload_summary_path}")
    print(f"Saved fact extraction results to: {fact_results_path}")
    print(f"Saved routing results to: {routing_results_path}")
    print(f"Saved retrieval results to: {retrieval_results_path}")
    print(f"Saved final answer results to: {final_results_path}")
    print(f"Saved failure cases to: {failure_cases_path}")

    return {
        "mode": mode,
        "workload_summary": workload_summary,
        "fact_summary": fact_summary_df,
        "routing_summary": routing_summary_df,
        "retrieval_summary": retrieval_summary_df,
        "final_answer_summary": final_summary_df,
    }


if __name__ == "__main__":
    if not evaluation_is_explicitly_allowed():
        raise SystemExit(
            f"Evaluation is disabled by default. Re-run with {EVALUATION_OPT_IN_ENV}=1 after explicit user approval."
        )

    evaluation_outputs = run_evaluation()
    print(evaluation_outputs["final_answer_summary"][["method", *FINAL_ANSWER_SUMMARY_COLUMNS]])
