"""Fit SVD on the skill co-occurrence matrix to produce skill embeddings.

Reads the vocab/co-occurrence matrix produced by build_cooccurrence.py and
writes a (n_skills x SVD_COMPONENTS) embeddings array, the input for
"related skill" recommendations.

Usage:
    python scripts/factorize_skills.py
"""
import json

import numpy as np
from scipy import sparse

from skill_recommender import config, factorization

EXAMPLE_SKILLS = ["python", "communication", "excel"]


def main() -> None:
    print("Loading vocab and co-occurrence matrix...")
    with open(config.VOCAB_PATH) as f:
        vocab = json.load(f)
    cooccurrence = sparse.load_npz(config.COOCCURRENCE_PATH)

    print(f"Fitting TruncatedSVD (n_components={config.SVD_COMPONENTS}) on "
          f"{cooccurrence.shape} co-occurrence matrix...")
    embeddings, svd = factorization.fit_svd(cooccurrence)
    print(f"Explained variance ratio (sum): {svd.explained_variance_ratio_.sum():.3f}")

    np.save(config.EMBEDDINGS_PATH, embeddings)
    print(f"Saved {embeddings.shape} embeddings to {config.EMBEDDINGS_PATH}")

    print("\n--- Sanity check: nearest neighbors by skill embedding ---")
    for skill in EXAMPLE_SKILLS:
        neighbors = factorization.most_similar(skill, vocab, embeddings)
        if not neighbors:
            print(f"  '{skill}' not in vocab")
            continue
        print(f"  {skill}: {neighbors}")


if __name__ == "__main__":
    main()
