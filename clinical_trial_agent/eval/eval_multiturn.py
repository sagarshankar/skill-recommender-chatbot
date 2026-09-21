"""Evaluate multi-turn conversation quality using deepeval metrics.

Replays golden conversations through the agent graph turn-by-turn,
captures responses and retrieval contexts, then evaluates with 6 metrics:
  - KnowledgeRetentionMetric
  - ConversationCompletenessMetric
  - TurnRelevancyMetric
  - RoleAdherenceMetric
  - TurnFaithfulnessMetric
  - TurnContextualRecallMetric

Requires:
    - OPENAI_API_KEY environment variable
    - An ingested FAISS index (run ingest.py first)

Usage:
    python -m clinical_trial_agent.eval.eval_multiturn
    python -m clinical_trial_agent.eval.eval_multiturn --golden-path path/to/custom.json
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
    TurnContextualRecallMetric,
    TurnFaithfulnessMetric,
    TurnRelevancyMetric,
)
from deepeval.test_case import ConversationalTestCase, Turn
from dotenv import load_dotenv

from clinical_trial_agent.src.graph import build_graph
from clinical_trial_agent.src.retriever import DEFAULT_INDEX_DIR

load_dotenv()

DEFAULT_GOLDEN_PATH = Path(__file__).parent / "multiturn_golden.json"


def load_golden_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def replay_conversation(
    app, conversation: dict, max_iterations: int = 3, delay: float = 2.0
) -> list[Turn]:
    """Replay a golden conversation through the graph, returning Turn objects."""
    accumulated_messages = []
    turns: list[Turn] = []

    user_turns = [t for t in conversation["turns"] if t["role"] == "user"]

    for i, user_turn in enumerate(user_turns):
        if i > 0:
            time.sleep(delay)

        user_content = user_turn["content"]
        turns.append(Turn(role="user", content=user_content))

        turn_state = {
            "query": user_content,
            "messages": list(accumulated_messages),
            "iteration": 0,
            "max_iterations": max_iterations,
        }

        result = app.invoke(turn_state)
        accumulated_messages = result.get("messages", accumulated_messages)

        assistant_content = result.get("report", "No response generated.")
        retrieved_docs = result.get("retrieved_docs", [])

        turns.append(
            Turn(
                role="assistant",
                content=assistant_content,
                retrieval_context=retrieved_docs if retrieved_docs else None,
            )
        )

    return turns


def build_test_cases(
    golden_data: list[dict], app, max_iterations: int, delay: float = 2.0
) -> tuple[list[ConversationalTestCase], list[str], list[int]]:
    test_cases = []
    categories = []
    user_turn_counts = []

    for i, conversation in enumerate(golden_data):
        if i > 0:
            time.sleep(delay)

        conv_id = conversation["id"]
        n_user_turns = sum(1 for t in conversation["turns"] if t["role"] == "user")
        print(f"  Replaying ({i + 1}/{len(golden_data)}): {conv_id} ({n_user_turns} user turn{'s' if n_user_turns != 1 else ''})")

        turns = replay_conversation(app, conversation, max_iterations, delay=delay)

        test_case = ConversationalTestCase(
            turns=turns,
        )

        test_cases.append(test_case)
        categories.append(conversation.get("category", "unknown"))
        user_turn_counts.append(n_user_turns)

    return test_cases, categories, user_turn_counts


def print_category_summary(
    test_results: list,
    categories: list[str],
    metrics: list,
) -> None:
    groups: dict[str, list[int]] = {}
    for i, cat in enumerate(categories):
        groups.setdefault(cat, []).append(i)

    print("\n" + "=" * 60)
    print("PER-CATEGORY AGGREGATE SCORES")
    print("=" * 60)

    for category, indices in sorted(groups.items()):
        print(f"\n  [{category}] ({len(indices)} conversations)")
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
                print(
                    f"    {metric.__name__}: "
                    f"avg={avg:.3f}  min={min(scores):.3f}  max={max(scores):.3f}"
                )
            else:
                print(f"    {metric.__name__}: no scores")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-turn conversation quality using deepeval"
    )
    parser.add_argument(
        "--golden-path",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="Path to the multi-turn golden evaluation dataset JSON",
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
        default=0.5,
        help="Minimum passing score for each metric (default: 0.5)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum retrieval-analysis retry loops per turn (default: 3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Only run the first N conversations (useful for testing)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds to wait between API calls to avoid rate limits (default: 3.0)",
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
    args = parser.parse_args()

    if not args.local_judge and not os.environ.get("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is required (or use --local-judge)",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.index_dir.exists():
        print(
            f"Error: FAISS index not found at {args.index_dir}. Run ingest first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading golden dataset from {args.golden_path}")
    golden_data = load_golden_dataset(args.golden_path)
    if args.limit:
        golden_data = golden_data[: args.limit]
    print(f"Running {len(golden_data)} conversations")

    print("Building agent graph...")
    app = build_graph()

    print(f"Replaying conversations (delay={args.delay}s between calls)...")
    test_cases, categories, user_turn_counts = build_test_cases(
        golden_data, app, args.max_iterations, delay=args.delay
    )

    if args.local_judge:
        from clinical_trial_agent.eval.local_judge import LocalJudge

        judge_model_name = args.judge_model_name
        judge = LocalJudge(judge_model_name) if judge_model_name else LocalJudge()
        print(f"Using local judge: {judge.get_model_name()}")
    else:
        judge = args.model

    universal_metrics = [
        ConversationCompletenessMetric(
            threshold=args.threshold, model=judge, include_reason=True
        ),
        TurnRelevancyMetric(
            threshold=args.threshold, model=judge, include_reason=True
        ),
        TurnFaithfulnessMetric(
            threshold=args.threshold, model=judge, include_reason=True
        )
    ]

    multiturn_only_metrics = [
        KnowledgeRetentionMetric(
            threshold=args.threshold, model=judge, include_reason=True
        ),
    ]

    judge_label = judge.get_model_name() if args.local_judge else args.model

    multiturn_cases = [tc for tc, n in zip(test_cases, user_turn_counts) if n >= 2]
    multiturn_categories = [cat for cat, n in zip(categories, user_turn_counts) if n >= 2]

    print(
        f"\nEvaluating all {len(test_cases)} conversations with universal metrics "
        f"(judge={judge_label}, threshold={args.threshold})...\n"
    )
    async_cfg = AsyncConfig(
        run_async=False,
        max_concurrent=args.max_concurrent,
        throttle_value=args.throttle,
    )
    eval_result = evaluate(
        test_cases=test_cases, metrics=universal_metrics, async_config=async_cfg
    )

    all_metrics = list(universal_metrics)
    multiturn_results = []
    if multiturn_cases:
        print(
            f"\nEvaluating {len(multiturn_cases)} multi-turn conversations with "
            f"KnowledgeRetention (skipping single-turn)...\n"
        )
        mt_eval_result = evaluate(
            test_cases=multiturn_cases,
            metrics=multiturn_only_metrics,
            async_config=async_cfg,
        )
        multiturn_results = mt_eval_result.test_results
        all_metrics.extend(multiturn_only_metrics)

    print_category_summary(eval_result.test_results, categories, universal_metrics)

    if multiturn_results:
        print("\n" + "=" * 60)
        print("KNOWLEDGE RETENTION (multi-turn conversations only)")
        print("=" * 60)
        print_category_summary(multiturn_results, multiturn_categories, multiturn_only_metrics)


if __name__ == "__main__":
    main()
