"""Job posting loading/sampling and output assembly."""
import json

import pandas as pd

from skill_recommender import config
from skill_recommender.text import split_sentences


def load_postings_sample(
    path=None, sample_size: int | None = None, random_state: int | None = None
) -> pd.DataFrame:
    path = path or config.POSTINGS_PATH
    sample_size = sample_size or config.SAMPLE_SIZE
    random_state = config.RANDOM_STATE if random_state is None else random_state
    df = pd.read_csv(path, usecols=["job_id", "description"], dtype={"job_id": str})
    df = df.dropna(subset=["description"])
    df = df.drop_duplicates(subset=["description"])
    return df.sample(n=min(sample_size, len(df)), random_state=random_state)


def explode_to_sentences(sample: pd.DataFrame) -> tuple[list[str], list[str]]:
    job_ids: list[str] = []
    sentences: list[str] = []
    for job_id, description in zip(sample["job_id"], sample["description"]):
        for sentence in split_sentences(description):
            job_ids.append(job_id)
            sentences.append(sentence)
    return job_ids, sentences


def assemble_output(
    sample: pd.DataFrame,
    hard_skills_by_job: dict[str, set[str]],
    soft_skills_by_job: dict[str, set[str]],
) -> pd.DataFrame:
    # Store skill lists as JSON strings (not raw set() reprs) so they round-trip
    # cleanly through CSV via json.loads instead of needing ast.literal_eval.
    out = sample.copy()
    out["hard_skills"] = [
        json.dumps(sorted(hard_skills_by_job.get(job_id, set()))) for job_id in sample["job_id"]
    ]
    out["soft_skills"] = [
        json.dumps(sorted(soft_skills_by_job.get(job_id, set()))) for job_id in sample["job_id"]
    ]
    return out
