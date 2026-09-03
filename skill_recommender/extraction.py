"""NER pipeline construction and batched skill extraction."""
from collections import defaultdict

from tqdm import tqdm
from transformers import pipeline

from skill_recommender.config import BATCH_SIZE, CHUNK_SIZE, HARD_SKILL_MODEL, SOFT_SKILL_MODEL
from skill_recommender.text import clean_span


def build_pipelines():
    """Load the hard/soft skill NER pipelines.

    "max" aggregates at the word level (using the tokenizer's word-boundary
    info) before merging entities, so a span always contains every subword
    piece of a word - this is what avoids leftover "##" fragments that
    "simple" aggregation produces when only a word's tail subtoken is tagged.
    """
    hard_skill_pipe = pipeline(
        "token-classification", model=HARD_SKILL_MODEL, aggregation_strategy="max"
    )
    soft_skill_pipe = pipeline(
        "token-classification", model=SOFT_SKILL_MODEL, aggregation_strategy="max"
    )
    # The model cards don't set tokenizer.model_max_length, so it defaults to
    # "no limit" and long sentences overflow the model's 512 position embeddings.
    for pipe in (hard_skill_pipe, soft_skill_pipe):
        pipe.tokenizer.model_max_length = pipe.model.config.max_position_embeddings
    return hard_skill_pipe, soft_skill_pipe

def extract_skills_for_all(pipe, job_ids: list[str], sentences: list[str]) -> dict[str, set[str]]:
    """Run the NER pipeline over every sentence (chunked, with internal batching)
    and group extracted skill spans back by job_id.

    Sentences are processed in ascending length order so each batch pads to
    roughly the length of its own sentences rather than to a random outlier -
    a handful of 800-word run-ons no longer inflate every batch they land in.
    """
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    skills_by_job: dict[str, set[str]] = defaultdict(set)
    for start in tqdm(range(0, len(order), CHUNK_SIZE), desc="NER chunks"):
        chunk_idx = order[start : start + CHUNK_SIZE]
        chunk_ids = [job_ids[i] for i in chunk_idx]
        chunk_sentences = [sentences[i] for i in chunk_idx]
        outputs = pipe(chunk_sentences, batch_size=BATCH_SIZE)
        for job_id, entities in zip(chunk_ids, outputs):
            for entity in entities:
                span = clean_span(entity["word"])
                if span:
                    skills_by_job[job_id].add(span)
    return skills_by_job
