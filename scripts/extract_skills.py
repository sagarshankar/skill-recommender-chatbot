"""Extract structured skills from LinkedIn job posting descriptions using
open-source HF token-classification models (no paid LLM calls).

Usage:
    python scripts/extract_skills.py
"""
from skill_recommender import config, data, extraction


def main() -> None:
    print("Loading job postings...")
    sample = data.load_postings_sample()
    print(f"Sampled {len(sample)} unique job postings.")

    print("Splitting descriptions into sentences...")
    job_ids, sentences = data.explode_to_sentences(sample)
    print(f"{len(sentences)} sentences total across {len(sample)} postings.")

    print("Loading skill extraction models (first run downloads ~440MB each)...")
    hard_pipe, soft_pipe = extraction.build_pipelines()

    print("Extracting hard skills (knowledge model)...")
    hard_skills_by_job = extraction.extract_skills_for_all(hard_pipe, job_ids, sentences)
    print("Extracting soft skills (skill model)...")
    soft_skills_by_job = extraction.extract_skills_for_all(soft_pipe, job_ids, sentences)

    out = data.assemble_output(sample, hard_skills_by_job, soft_skills_by_job)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.SKILLS_SAMPLE_PATH, index=False)
    print(f"Wrote {len(out)} rows to {config.SKILLS_SAMPLE_PATH}")


if __name__ == "__main__":
    main()
