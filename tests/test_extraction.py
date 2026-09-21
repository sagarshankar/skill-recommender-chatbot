"""Tests for extract_skills_for_all's batching/grouping logic using a fake
pipe, instead of loading a real HF model - keeps tests fast and offline.
build_pipelines() itself (real model download) is intentionally not tested
here; that's an integration concern, not a unit test.
"""
import skill_recommender.extraction as extraction


def fake_pipe_factory(entity_lookup: dict[str, list[dict]]):
    def fake_pipe(sentences, batch_size=None):
        return [entity_lookup.get(s, []) for s in sentences]

    return fake_pipe


class TestExtractSkillsForAll:
    def test_groups_entities_by_job_id(self):
        entity_lookup = {
            "Knows Python.": [{"word": "python"}],
            "Knows SQL.": [{"word": "sql"}],
            "Loves teamwork.": [{"word": "teamwork"}],
        }
        pipe = fake_pipe_factory(entity_lookup)
        job_ids = ["job1", "job1", "job2"]
        sentences = ["Knows Python.", "Knows SQL.", "Loves teamwork."]

        result = extraction.extract_skills_for_all(pipe, job_ids, sentences)

        assert result == {"job1": {"python", "sql"}, "job2": {"teamwork"}}

    def test_filters_malformed_spans_via_clean_span(self):
        entity_lookup = {"Some sentence.": [{"word": "##broken"}, {"word": "python"}]}
        pipe = fake_pipe_factory(entity_lookup)

        result = extraction.extract_skills_for_all(pipe, ["job1"], ["Some sentence."])

        assert result == {"job1": {"python"}}

    def test_jobs_with_no_matches_are_absent_not_empty(self):
        pipe = fake_pipe_factory({})
        result = extraction.extract_skills_for_all(pipe, ["job1"], ["Nothing found here."])
        assert "job1" not in result

    def test_aggregates_correctly_across_multiple_chunks(self, monkeypatch):
        # Force every sentence into its own chunk to exercise the
        # multi-chunk accumulation path, not just the common single-chunk case.
        monkeypatch.setattr(extraction, "CHUNK_SIZE", 1)
        entity_lookup = {
            "One.": [{"word": "a"}],
            "Two.": [{"word": "b"}],
            "Three.": [{"word": "c"}],
        }
        pipe = fake_pipe_factory(entity_lookup)
        job_ids = ["job1", "job1", "job2"]
        sentences = ["One.", "Two.", "Three."]

        result = extraction.extract_skills_for_all(pipe, job_ids, sentences)

        assert result == {"job1": {"a", "b"}, "job2": {"c"}}
