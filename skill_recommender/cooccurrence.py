"""Skill vocabulary and skill x skill co-occurrence matrix construction."""
import json

import pandas as pd
from scipy import sparse

from skill_recommender.config import MIN_SKILL_FREQ

def parse_posting_skills(df: pd.DataFrame) -> list[set[str]]:
    postings_skills = []
    for hard_json, soft_json in zip(df["hard_skills"], df["soft_skills"]):
        postings_skills.append(set(json.loads(hard_json)) | set(json.loads(soft_json)))
    return postings_skills


def build_vocab(
    postings_skills: list[set[str]], min_freq: int | None = None
) -> tuple[list[str], dict[str, int]]:
    min_freq = MIN_SKILL_FREQ if min_freq is None else min_freq
    skill_counts: dict[str, int] = {}
    for skills in postings_skills:
        for skill in skills:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
    vocab = sorted(skill for skill, count in skill_counts.items() if count >= min_freq)
    skill_to_idx = {skill: i for i, skill in enumerate(vocab)}
    return vocab, skill_to_idx


def build_cooccurrence_matrix(
    postings_skills: list[set[str]], skill_to_idx: dict[str, int]
) -> sparse.csr_matrix:
    rows, cols = [], []
    for posting_idx, skills in enumerate(postings_skills):
        for skill in skills:
            skill_idx = skill_to_idx.get(skill)
            if skill_idx is not None:
                rows.append(posting_idx)
                cols.append(skill_idx)
    data = [1] * len(rows)
    posting_skill_matrix = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(postings_skills), len(skill_to_idx))
    )
    return (posting_skill_matrix.T @ posting_skill_matrix).tocsr()


def summarize_cooccurrence(
    vocab: list[str], cooccurrence: sparse.csr_matrix, top_n: int = 10
) -> str:
    skill_to_idx = {skill: i for i, skill in enumerate(vocab)}
    freq = cooccurrence.diagonal()
    top_idx = freq.argsort()[::-1][:15]
    lines = ["Most frequent skills:"]
    for i in top_idx:
        lines.append(f"  {vocab[i]}: {int(freq[i])} postings")

    lines.append("\nTop co-occurring skills for a few examples:")
    for example in ["python", "communication", "excel"]:
        if example not in skill_to_idx:
            lines.append(f"  '{example}' not in vocab (below min-frequency threshold)")
            continue
        i = skill_to_idx[example]
        row = cooccurrence.getrow(i).toarray().ravel()
        row[i] = -1  # exclude self
        partners = row.argsort()[::-1][:top_n]
        lines.append(f"  {example}: {[(vocab[j], int(row[j])) for j in partners if row[j] > 0]}")
    return "\n".join(lines)
