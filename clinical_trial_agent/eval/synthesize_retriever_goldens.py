"""Generate retriever evaluation goldens from protocol documents using deepeval's Synthesizer.

Uses the Synthesizer to read protocol PDFs, extract quality-filtered contexts,
and generate input/expected_output pairs grounded in the document content.

Requires:
    - OPENAI_API_KEY environment variable (for the synthesis LLM and embeddings)

Usage:
    python -m clinical_trial_agent.eval.synthesize_retriever_goldens
    python -m clinical_trial_agent.eval.synthesize_retriever_goldens --max-goldens-per-context 3
    python -m clinical_trial_agent.eval.synthesize_retriever_goldens --output-path custom_goldens.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "protocols"
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "retriever_golden_synthetic.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate retriever evaluation goldens from protocol documents"
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help="Directory containing protocol PDFs (default: protocols/)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path for generated goldens",
    )
    parser.add_argument(
        "--max-goldens-per-context",
        type=int,
        default=2,
        help="Maximum goldens generated per context (default: 2)",
    )
    parser.add_argument(
        "--max-contexts-per-document",
        type=int,
        default=10,
        help="Maximum contexts extracted per document (default: 5)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Chunk size in tokens for document splitting (default: 1024)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in tokens (default: 200)",
    )
    parser.add_argument(
        "--context-quality-threshold",
        type=float,
        default=0.5,
        help="Minimum quality score for context chunks (default: 0.5)",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is required for synthesis",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.docs_dir.exists():
        print(f"Error: docs directory not found at {args.docs_dir}", file=sys.stderr)
        sys.exit(1)

    doc_paths = sorted(
        str(p) for p in args.docs_dir.iterdir()
        if p.suffix.lower() in (".pdf", ".txt", ".md", ".docx")
    )
    if not doc_paths:
        print(f"Error: no documents found in {args.docs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(doc_paths)} document(s):")
    for p in doc_paths:
        print(f"  - {p}")

    from deepeval.synthesizer import Synthesizer
    from deepeval.synthesizer.config import ContextConstructionConfig, StylingConfig

    context_config = ContextConstructionConfig(
        max_contexts_per_document=args.max_contexts_per_document,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        context_quality_threshold=args.context_quality_threshold,
    )

    styling_config = StylingConfig(
        input_format=(
            "A clinical scenario describing a concrete participant event "
            "such as dosing, eligibility, procedure timing, adverse event "
            "reporting, or concomitant medications. Not a question about "
            "the protocol document itself."
        ),
        expected_output_format=(
            "A deviation analysis that classifies each finding by type and "
            "severity, cites the specific protocol section violated, and "
            "provides evidence from the protocol. If compliant, cite the "
            "relevant rule confirming compliance."
        ),
        task=(
            "Identify protocol deviations by comparing clinical scenarios "
            "against protocol rules. Classify each deviation by type and "
            "severity, and cite the specific protocol section violated."
        ),
        scenario=(
            "A compliance analyst reviewing participant events for the "
            "sutimlimab study (BIVV009-LTS17352) to check whether they "
            "comply with the study protocol."
        ),
    )

    print(
        f"\nSynthesizer config:"
        f"\n  max_goldens_per_context: {args.max_goldens_per_context}"
        f"\n  max_contexts_per_document: {args.max_contexts_per_document}"
        f"\n  chunk_size: {args.chunk_size}"
        f"\n  chunk_overlap: {args.chunk_overlap}"
        f"\n  context_quality_threshold: {args.context_quality_threshold}"
        f"\n  expected max goldens: {args.max_goldens_per_context * args.max_contexts_per_document}"
    )

    synthesizer = Synthesizer(styling_config=styling_config)

    print("\nGenerating goldens from documents...")
    synthesizer.generate_goldens_from_docs(
        document_paths=doc_paths,
        include_expected_output=True,
        max_goldens_per_context=args.max_goldens_per_context,
        context_construction_config=context_config,
    )

    goldens = synthesizer.synthetic_goldens
    print(f"\nGenerated {len(goldens)} goldens")

    output_data = []
    for i, golden in enumerate(goldens):
        entry = {
            "input": golden.input,
            "expected_output": golden.expected_output or "",
            "context": golden.context or [],
            "source_file": golden.source_file or "",
            "category": "synthetic",
        }
        output_data.append(entry)
        print(f"\n  [{i + 1}] {golden.input[:80]}...")
        if golden.expected_output:
            print(f"      Expected: {golden.expected_output[:80]}...")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(output_data)} goldens to {args.output_path}")


if __name__ == "__main__":
    main()
