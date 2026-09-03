"""Build a skill-skill co-occurrence matrix from extracted job posting skills.

Reads the CSV produced by extract_skills.py (job_id + JSON-encoded hard_skills/
soft_skills columns), builds a sparse job-posting x skill matrix, and derives
a skill x skill co-occurrence matrix from it (M^T @ M). This is the input for
a later SVD step to get low-dimensional skill embeddings for recommendations.

Usage:
    python scripts/build_cooccurrence.py
"""
import json

import pandas as pd
from scipy import sparse

from skill_recommender import config, cooccurrence


def main() -> None:
    print("Loading extracted skills...")
    df = pd.read_csv(config.SKILLS_SAMPLE_PATH)
    postings_skills = cooccurrence.parse_posting_skills(df)

    print("Building skill vocabulary...")
    vocab, skill_to_idx = cooccurrence.build_vocab(postings_skills)
    print(f"{len(vocab)} skills kept after min-frequency filter "
          f"(>= {config.MIN_SKILL_FREQ} postings).")

    print("Building posting x skill matrix and computing co-occurrence...")
    matrix = cooccurrence.build_cooccurrence_matrix(postings_skills, skill_to_idx)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.VOCAB_PATH, "w") as f:
        json.dump(vocab, f)
    sparse.save_npz(config.COOCCURRENCE_PATH, matrix)
    print(f"Saved vocab ({len(vocab)} skills) to {config.VOCAB_PATH}")
    print(f"Saved {matrix.shape} co-occurrence matrix to {config.COOCCURRENCE_PATH}")

    print("\n--- Sanity check ---")
    print(cooccurrence.summarize_cooccurrence(vocab, matrix))


if __name__ == "__main__":
    main()
