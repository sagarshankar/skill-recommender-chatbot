from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from clinical_trial_agent.src.state import AgentState

DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
BASE_K = 6
K_INCREMENT = 4

_vectorstore_cache: dict[str, FAISS] = {}


def _load_vectorstore(index_dir: Path) -> FAISS:
    key = str(index_dir)
    if key not in _vectorstore_cache:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        _vectorstore_cache[key] = FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )
    return _vectorstore_cache[key]


def retrieve(state: AgentState, index_dir: Path = DEFAULT_INDEX_DIR) -> dict:
    if not index_dir.exists():
        return {
            "analysis_status": "failed",
            "error": f"FAISS index not found at {index_dir}. Run ingest first.",
            "retrieved_docs": [],
            "retrieval_scores": [],
        }

    vectorstore = _load_vectorstore(index_dir)

    iteration = state.get("iteration", 0)
    k = BASE_K + (iteration * K_INCREMENT)

    query = state.get("rewritten_query") or state["query"]
    if iteration > 0 and state.get("missing_info"):
        query = f"{query}\n\nAdditional context needed: {state['missing_info']}"

    results = vectorstore.similarity_search_with_score(query, k=k)

    return {
        "retrieved_docs": [doc.page_content for doc, _ in results],
        "retrieval_scores": [float(score) for _, score in results],
    }
