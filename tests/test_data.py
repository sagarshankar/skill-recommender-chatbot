import json

import pandas as pd

from skill_recommender.data import assemble_output, explode_to_sentences, load_postings_sample


class TestLoadPostingsSample:
    def _write_csv(self, tmp_path, rows):
        path = tmp_path / "postings.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_drops_missing_descriptions(self, tmp_path):
        path = self._write_csv(
            tmp_path,
            [
                {"job_id": "1", "description": "Needs Python."},
                {"job_id": "2", "description": None},
            ],
        )
        result = load_postings_sample(path=path, sample_size=10, random_state=0)
        assert list(result["job_id"]) == ["1"]

    def test_drops_duplicate_descriptions(self, tmp_path):
        path = self._write_csv(
            tmp_path,
            [
                {"job_id": "1", "description": "Same posting text."},
                {"job_id": "2", "description": "Same posting text."},
                {"job_id": "3", "description": "Different posting."},
            ],
        )
        result = load_postings_sample(path=path, sample_size=10, random_state=0)
        assert len(result) == 2

    def test_sample_size_caps_result(self, tmp_path):
        rows = [{"job_id": str(i), "description": f"Posting number {i}."} for i in range(10)]
        path = self._write_csv(tmp_path, rows)
        result = load_postings_sample(path=path, sample_size=3, random_state=0)
        assert len(result) == 3


class TestExplodeToSentences:
    def test_pairs_each_sentence_with_its_job_id(self):
        sample = pd.DataFrame(
            {
                "job_id": ["1", "2"],
                "description": ["Knows Python. Knows SQL.", "Only one sentence here."],
            }
        )
        job_ids, sentences = explode_to_sentences(sample)
        assert job_ids == ["1", "1", "2"]
        assert sentences == ["Knows Python.", "Knows SQL.", "Only one sentence here."]


class TestAssembleOutput:
    def test_adds_sorted_json_skill_columns(self):
        sample = pd.DataFrame({"job_id": ["1", "2"], "description": ["desc a", "desc b"]})
        hard_skills_by_job = {"1": {"sql", "python"}}
        soft_skills_by_job = {"1": {"communication"}}
        out = assemble_output(sample, hard_skills_by_job, soft_skills_by_job)

        assert json.loads(out.loc[out["job_id"] == "1", "hard_skills"].iloc[0]) == [
            "python",
            "sql",
        ]
        # Job "2" had no extracted skills - should get an empty list, not a missing value.
        assert json.loads(out.loc[out["job_id"] == "2", "hard_skills"].iloc[0]) == []
        assert json.loads(out.loc[out["job_id"] == "2", "soft_skills"].iloc[0]) == []
