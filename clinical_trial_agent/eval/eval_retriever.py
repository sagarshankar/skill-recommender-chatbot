"""Evaluate retriever quality using deepeval's contextual RAG metrics.

Supports two modes:
  - Local-only (default): runs deepeval metrics and prints results to terminal
  - LangSmith (--langsmith): pushes dataset + experiment results to LangSmith UI

Requires:
    - OPENAI_API_KEY environment variable (or --local-judge)
    - An ingested FAISS index (run ingest.py first)
    - LANGSMITH_API_KEY environment variable (only with --langsmith)

Usage:
    python -m clinical_trial_agent.eval.eval_retriever
    python -m clinical_trial_agent.eval.eval_retriever --golden-path path/to/custom.json
    python -m clinical_trial_agent.eval.eval_retriever --langsmith
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path

from deepeval import evaluate as deepeval_evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from clinical_trial_agent.src.retriever import DEFAULT_INDEX_DIR, retrieve

import time
from langsmith.evaluation.evaluator import EvaluationResult, EvaluationResults

load_dotenv()

DEFAULT_GOLDEN_PATH = Path(__file__).parent / "retriever_golden_synthetic.json"


def load_golden_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_retriever(query: str, index_dir: Path) -> dict:
    """Run the retriever and return docs and joined output."""
    state = {"query": query, "iteration": 0}
    result = retrieve(state, index_dir=index_dir)
    retrieved_docs = result.get("retrieved_docs", [])
    actual_output = "\n\n".join(retrieved_docs) if retrieved_docs else "No documents retrieved."
    return {"retrieved_docs": retrieved_docs, "actual_output": actual_output}


def build_test_cases(
    golden_data: list[dict], index_dir: Path
) -> tuple[list[LLMTestCase], list[str], list[dict]]:
    test_cases = []
    categories = []
    retriever_results = []
    for entry in golden_data:
        query = entry["input"]
        result = run_retriever(query, index_dir)

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=result["actual_output"],
                expected_output=entry["expected_output"],
                retrieval_context=result["retrieved_docs"],
            )
        )
        categories.append(entry.get("category", "unknown"))
        retriever_results.append(result)
    return test_cases, categories, retriever_results


def print_tier_summary(
    test_results: list,
    categories: list[str],
    metrics: list,
) -> None:
    tiers: dict[str, list[int]] = {}
    for i, cat in enumerate(categories):
        tiers.setdefault(cat, []).append(i)

    print("\n" + "=" * 60)
    print("PER-TIER AGGREGATE SCORES")
    print("=" * 60)

    for tier, indices in sorted(tiers.items()):
        print(f"\n  [{tier}] ({len(indices)} cases)")
        for metric in metrics:
            scores = []
            for idx in indices:
                tr = test_results[idx]
                if tr.metrics_data:
                    for m in tr.metrics_data:
                        if m.name == metric.__name__:
                            if m.score is not None:
                                scores.append(m.score)
            if scores:
                avg = sum(scores) / len(scores)
                print(f"    {metric.__name__}: avg={avg:.3f}  min={min(scores):.3f}  max={max(scores):.3f}")
            else:
                print(f"    {metric.__name__}: no scores")


def ensure_langsmith_dataset(
    client, dataset_name: str, golden_data: list[dict]
) -> None:
    """Create or update a LangSmith dataset from golden data."""
    if client.has_dataset(dataset_name=dataset_name):
        print(f"  LangSmith dataset '{dataset_name}' already exists, reusing it")
        return

    print(f"  Creating LangSmith dataset '{dataset_name}'...")
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Retriever evaluation golden dataset for clinical trial protocol deviation analysis",
    )
    for entry in golden_data:
        client.create_example(
            inputs={"query": entry["input"]},
            outputs={"expected_output": entry["expected_output"]},
            metadata={"category": entry.get("category", "unknown")},
            dataset_id=dataset.id,
        )
    print(f"  Pushed {len(golden_data)} examples")


def run_langsmith_eval(
    golden_data: list[dict],
    index_dir: Path,
    metrics: list,
    experiment_prefix: str,
    threshold: float,
) -> None:
    """Run evaluation through LangSmith, logging dataset + results to the UI."""
    from langsmith import Client
    from langsmith.evaluation import evaluate as ls_evaluate

    client = Client()
    dataset_name = experiment_prefix + "-dataset"

    print("\nLangSmith integration:")
    ensure_langsmith_dataset(client, dataset_name, golden_data)

    def target(inputs: dict) -> dict:
        result = run_retriever(inputs["query"], index_dir)
        return {
            "actual_output": result["actual_output"],
            "retrieved_docs": result["retrieved_docs"],
        }

    def combined_evaluator(run, example):
        query = example.inputs["query"]
        expected = example.outputs["expected_output"]
        actual = run.outputs["actual_output"]
        retrieved = run.outputs["retrieved_docs"]

        tc = LLMTestCase(
            input=query,
            actual_output=actual,
            expected_output=expected,
            retrieval_context=retrieved,
        )

        results = []
        for metric in metrics:
            time.sleep(5)
            metric.measure(tc)
            results.append(
                EvaluationResult(
                    key=metric.__name__,
                    score=metric.score,
                    comment=metric.reason,
                )
            )
        return EvaluationResults(results=results)

    evaluators = [combined_evaluator]

    print(f"\n  Running LangSmith experiment '{experiment_prefix}'...")
    results = ls_evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        description=f"Retriever eval with deepeval metrics (threshold={threshold})",
        max_concurrency=1,
    )

    print(f"\n  Experiment complete — view results in LangSmith UI")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retriever quality using deepeval contextual metrics"
    )
    parser.add_argument(
        "--golden-path",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="Path to the golden evaluation dataset JSON",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Path to the FAISS index directory",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="LLM model for deepeval judge (default: gpt-4o)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum passing score for each metric (default: 0.7)",
    )
    parser.add_argument(
        "--local-judge",
        action="store_true",
        help="Use a local HuggingFace model as the deepeval judge instead of OpenAI",
    )
    parser.add_argument(
        "--judge-model-name",
        default=None,
        help="HuggingFace model ID for the local judge (default: Mistral-7B-Instruct-v0.3)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Maximum concurrent eval API calls to avoid rate limits (default: 2)",
    )
    parser.add_argument(
        "--throttle",
        type=int,
        default=5,
        help="Seconds to wait between concurrent batches (default: 5)",
    )
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help="Push dataset and experiment results to LangSmith",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="retriever-eval",
        help="LangSmith experiment name prefix (default: retriever-eval)",
    )
    args = parser.parse_args()

    if not args.local_judge and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required (or use --local-judge)", file=sys.stderr)
        sys.exit(1)

    if args.langsmith and not os.environ.get("LANGSMITH_API_KEY"):
        print("Error: LANGSMITH_API_KEY environment variable is required with --langsmith", file=sys.stderr)
        sys.exit(1)

    if not args.index_dir.exists():
        print(
            f"Error: FAISS index not found at {args.index_dir}. Run ingest first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading golden dataset from {args.golden_path}")
    golden_data = load_golden_dataset(args.golden_path)
    print(f"Found {len(golden_data)} test cases")

    if args.local_judge:
        from clinical_trial_agent.eval.local_judge import LocalJudge

        judge_model_name = args.judge_model_name
        judge = LocalJudge(judge_model_name) if judge_model_name else LocalJudge()
        print(f"Using local judge: {judge.get_model_name()}")
    else:
        judge = args.model

    metrics = [
        ContextualPrecisionMetric(
            threshold=args.threshold, model=judge, include_reason=True
        ),
        ContextualRecallMetric(
            threshold=args.threshold, model=judge, include_reason=True
        ),
        ContextualRelevancyMetric(
            threshold=args.threshold, model=judge, include_reason=True
        ),
    ]

    if args.langsmith:
        run_langsmith_eval(
            golden_data,
            args.index_dir,
            metrics,
            args.experiment_prefix,
            args.threshold,
        )
    else:
        print("Running retriever on each query...")
        test_cases, categories, _ = build_test_cases(golden_data, args.index_dir)

        judge_label = judge.get_model_name() if args.local_judge else args.model
        print(f"\nEvaluating with deepeval (judge={judge_label}, threshold={args.threshold})...\n")
        eval_result = deepeval_evaluate(
            test_cases=test_cases,
            metrics=metrics,
            async_config=AsyncConfig(
                run_async=False,
                max_concurrent=args.max_concurrent,
                throttle_value=args.throttle,
            ),
        )

        print_tier_summary(eval_result.test_results, categories, metrics)


if __name__ == "__main__":
    main()
