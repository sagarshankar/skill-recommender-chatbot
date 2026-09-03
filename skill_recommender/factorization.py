"""SVD factorization of the skill co-occurrence matrix into skill embeddings.

Truncated SVD on the sparse skill x skill co-occurrence matrix gives each
skill a dense low-dimensional vector such that skills that tend to appear
together in the same job postings end up close together in the embedding
space - the input for "related skill" recommendations.
"""
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from skill_recommender.config import RANDOM_STATE, SVD_COMPONENTS


def fit_svd(
    cooccurrence: sparse.csr_matrix, n_components: int | None = None
) -> tuple[np.ndarray, TruncatedSVD]:
    n_components = SVD_COMPONENTS if n_components is None else n_components
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    embeddings = svd.fit_transform(cooccurrence)
    return embeddings, svd


def most_similar(
    skill: str, vocab: list[str], embeddings: np.ndarray, top_n: int = 10
) -> list[tuple[str, float]]:
    """Nearest neighbors of `skill` in the embedding space by cosine similarity."""
    skill_to_idx = {s: i for i, s in enumerate(vocab)}
    if skill not in skill_to_idx:
        return []
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized = embeddings / norms
    sims = normalized @ normalized[skill_to_idx[skill]]

    order = sims.argsort()[::-1]
    results = []
    for i in order:
        if vocab[i] == skill:
            continue
        results.append((vocab[i], float(sims[i])))
        if len(results) >= top_n:
            break
    return results
