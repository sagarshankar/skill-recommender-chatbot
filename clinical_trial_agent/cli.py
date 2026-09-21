"""CLI entrypoint for the clinical trial deviation analysis agent.

Usage:
    python -m clinical_trial_agent.cli "Patient enrolled despite elevated creatinine above exclusion threshold"
    python -m clinical_trial_agent.cli -i
"""

from __future__ import annotations

import argparse
import os
import sys

from clinical_trial_agent.src import nodes
from clinical_trial_agent.src.graph import build_graph
from dotenv import load_dotenv

load_dotenv()


def _run_single(app, query: str, max_iterations: int) -> None:
    initial_state = {
        "query": query,
        "iteration": 0,
        "max_iterations": max_iterations,
    }
    result = app.invoke(initial_state)
    print(result.get("report", "No report generated."))


def _run_interactive(app, max_iterations: int) -> None:
    print("Clinical Trial Protocol Deviation Analyzer — Interactive Mode")
    print("Type 'quit' or 'exit' to end, 'reset' to clear conversation history.\n")

    accumulated_messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Exiting.")
            break
        if user_input.lower() == "reset":
            accumulated_messages = []
            print("Conversation history cleared.\n")
            continue

        turn_state = {
            "query": user_input,
            "messages": accumulated_messages,
            "iteration": 0,
            "max_iterations": max_iterations,
        }

        result = app.invoke(turn_state)
        accumulated_messages = result.get("messages", accumulated_messages)

        print(f"\nAssistant: {result.get('report', 'No response generated.')}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze clinical trial protocol deviations"
    )
    parser.add_argument(
        "query", nargs="?", default=None,
        help="Clinical scenario to analyze for deviations",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start an interactive multi-turn conversation",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum retrieval-analysis retry loops (default: 3)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Anthropic model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Path to FAISS index directory (default: clinical_trial_agent/data/faiss_index/)",
    )
    args = parser.parse_args()

    if not args.interactive and not args.query:
        parser.error("either provide a query or use --interactive / -i")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is required", file=sys.stderr)
        sys.exit(1)

    if args.model != nodes.DEFAULT_MODEL:
        nodes.DEFAULT_MODEL = args.model

    app = build_graph()

    if args.interactive:
        _run_interactive(app, args.max_iterations)
    else:
        _run_single(app, args.query, args.max_iterations)


if __name__ == "__main__":
    main()
