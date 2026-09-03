import numpy as np
from scipy import sparse

from skill_recommender.factorization import fit_svd, most_similar


class TestFitSvd:
    def test_output_shape_matches_n_components(self):
        matrix = sparse.random(20, 20, density=0.3, format="csr")
        matrix = (matrix.T @ matrix).tocsr()  # symmetric, like a real co-occurrence matrix
        embeddings, svd = fit_svd(matrix, n_components=5)
        assert embeddings.shape == (20, 5)
        assert svd.n_components == 5


class TestMostSimilar:
    VOCAB = ["a", "b", "c", "d"]
    # "a" and "b" point the same direction (identical after normalization);
    # "c" is orthogonal; "d" is a zero vector (edge case: no direction at all).
    EMBEDDINGS = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )

    def test_finds_closest_direction_first(self):
        result = most_similar("a", self.VOCAB, self.EMBEDDINGS, top_n=3)
        assert result[0][0] == "b"
        assert result[0][1] > 0.99  # cosine similarity ~1.0 for same direction

    def test_excludes_the_query_skill_itself(self):
        result = most_similar("a", self.VOCAB, self.EMBEDDINGS)
        assert "a" not in [skill for skill, _ in result]

    def test_unknown_skill_returns_empty(self):
        assert most_similar("not_in_vocab", self.VOCAB, self.EMBEDDINGS) == []

    def test_zero_vector_does_not_raise(self):
        # Regression guard: a skill with an all-zero embedding must not
        # produce a division-by-zero crash when normalizing.
        result = most_similar("d", self.VOCAB, self.EMBEDDINGS)
        assert isinstance(result, list)
