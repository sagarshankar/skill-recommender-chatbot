"""Shared paths and constants for the skill_recommender pipeline."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"

POSTINGS_PATH = DATA_DIR / "postings.csv"
SKILLS_SAMPLE_PATH = OUTPUT_DIR / "skills_sample.csv"
VOCAB_PATH = OUTPUT_DIR / "skill_vocab.json"
COOCCURRENCE_PATH = OUTPUT_DIR / "skill_cooccurrence.npz"
EMBEDDINGS_PATH = OUTPUT_DIR / "skill_embeddings.npy"

# Extraction run
RANDOM_STATE = 42
BATCH_SIZE = 32
CHUNK_SIZE = 500
MAX_SPAN_WORDS = 5
SAMPLE_SIZE = 10000
HARD_SKILL_MODEL = "jjzha/jobbert_knowledge_extraction"
SOFT_SKILL_MODEL = "jjzha/jobbert_skill_extraction"

# Co-occurrence
MIN_SKILL_FREQ = 3

# SVD factorization
SVD_COMPONENTS = 100
