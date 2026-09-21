"""Pure text-preprocessing helpers - no ML or pandas dependencies."""
import re

from skill_recommender.config import MAX_SPAN_WORDS

# Split on sentence-ending punctuation AND on newlines/bullet markers, since job
# postings often use bullet lists with no terminal punctuation - without this,
# those lists collapse into a single multi-hundred-word "sentence" that forces
# every other sentence batched alongside it to pad up to its length.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+|(?:^|\n)\s*[•\-\*]\s*")

# The scraped `description` field often loses the boundary between adjacent
# HTML <li> bullet items entirely (no space or newline survives), e.g.
# "...as assignedServices or replaces..." - the only remaining signal is the
# lowercase-to-uppercase letter transition, so insert a space there.
CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    text = CAMEL_BOUNDARY_RE.sub(" ", text.strip())
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def clean_span(span: str) -> str | None:
    span = span.strip().lower().strip(".,;:()")
    if not span or "#" in span:
        # "#" survives when aggregation groups start mid-word (a leftover
        # WordPiece continuation token); such spans are malformed, drop them.
        return None
    if len(span.split()) > MAX_SPAN_WORDS:
        # jjzha's "Skill" (soft-skill) labels are trained on full duty/
        # responsibility clauses, not keywords - anything this long reads as
        # a sentence fragment rather than a skill, so drop it for a clean list.
        return None
    return span
