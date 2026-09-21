import json

import pandas as pd

from skill_recommender.cooccurrence import (
    build_cooccurrence_matrix,
    build_vocab,
    parse_posting_skills,
)


class TestParsePostingSkills:
    def test_unions_hard_and_soft_skills_per_row(self):
        df = pd.DataFrame(
            {
                "hard_skills": [json.dumps(["python", "sql"]), json.dumps(["excel"])],
                "soft_skills": [json.dumps(["communication"]), json.dumps(["sql"])],
            }
        )
        result = parse_posting_skills(df)
        assert result == [{"python", "sql", "communication"}, {"excel", "sql"}]


class TestBuildVocab:
    POSTINGS_SKILLS = [
        {"python", "sql"},
        {"python", "excel"},
        {"python"},
        {"excel"},
    ]

    def test_filters_by_min_frequency(self):
        vocab, skill_to_idx = build_vocab(self.POSTINGS_SKILLS, min_freq=2)
        assert vocab == ["excel", "python"]
        assert skill_to_idx == {"excel": 0, "python": 1}

    def test_uses_config_default_when_min_freq_not_given(self):
        # config.MIN_SKILL_FREQ is 3 - only "python" (freq 3) survives.
        vocab, _ = build_vocab(self.POSTINGS_SKILLS)
        assert vocab == ["python"]

    def test_vocab_is_alphabetically_sorted(self):
        vocab, _ = build_vocab(self.POSTINGS_SKILLS, min_freq=1)
        assert vocab == sorted(vocab)


class TestBuildCooccurrenceMatrix:
    def test_diagonal_is_skill_frequency(self):
        postings_skills = [{"python", "sql"}, {"python"}, {"sql"}]
        skill_to_idx = {"python": 0, "sql": 1}
        matrix = build_cooccurrence_matrix(postings_skills, skill_to_idx)
        dense = matrix.toarray()
        assert dense[0][0] == 2  # python appears in 2 postings
        assert dense[1][1] == 2  # sql appears in 2 postings

    def test_off_diagonal_is_co_occurrence_count(self):
        postings_skills = [{"python", "sql"}, {"python", "sql"}, {"python"}]
        skill_to_idx = {"python": 0, "sql": 1}
        matrix = build_cooccurrence_matrix(postings_skills, skill_to_idx)
        dense = matrix.toarray()
        assert dense[0][1] == 2
        assert dense[1][0] == 2  # matrix is symmetric

    def test_skills_outside_vocab_are_ignored(self):
        postings_skills = [{"python", "rare_skill"}]
        skill_to_idx = {"python": 0}
        matrix = build_cooccurrence_matrix(postings_skills, skill_to_idx)
        assert matrix.shape == (1, 1)
        assert matrix.toarray()[0][0] == 1
