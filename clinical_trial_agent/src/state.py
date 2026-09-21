from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from clinical_trial_agent.src.models import DeviationFinding


class AgentState(TypedDict, total=False):
    query: str
    messages: Annotated[list[AnyMessage], add_messages]

    retrieved_docs: list[str]
    retrieval_scores: list[float]

    findings: list[DeviationFinding]
    analysis_status: str  # "complete" | "needs_more_context" | "failed"
    analyst_reasoning: str
    missing_info: str

    iteration: int
    max_iterations: int

    report: str
    error: str

    response_mode: str  # "deviation_analysis" | "conversational"
    rewritten_query: str
