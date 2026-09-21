from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_anthropic import ChatAnthropic

from clinical_trial_agent.src.models import (
    DeviationAnalysisOutput,
    DeviationFinding,
    IntentClassification,
    SeverityLevel,
)
from clinical_trial_agent.src.prompts import (
    build_analyst_messages,
    build_conversational_messages,
    build_intent_classification_messages,
)
from clinical_trial_agent.src.state import AgentState

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
INTENT_MODEL = "claude-haiku-4-5-20251001"


def classify_intent(state: AgentState) -> dict:
    query = state["query"]
    conversation_history = state.get("messages", [])

    if not conversation_history:
        return {
            "response_mode": "deviation_analysis",
            "rewritten_query": query,
            "messages": [HumanMessage(content=query)],
        }

    llm = ChatAnthropic(
        model=INTENT_MODEL, temperature=0
    ).with_structured_output(IntentClassification)

    messages = build_intent_classification_messages(query, conversation_history)
    result: IntentClassification = llm.invoke(messages)

    return {
        "response_mode": result.mode,
        "rewritten_query": result.rewritten_query,
        "messages": [HumanMessage(content=query)],
    }


def analyze_deviation(state: AgentState) -> dict:
    if state.get("analysis_status") == "failed":
        return {}

    llm = ChatAnthropic(
        model=DEFAULT_MODEL, temperature=0
    ).with_structured_output(DeviationAnalysisOutput)

    conversation_context = state.get("messages", [])
    messages = build_analyst_messages(state, conversation_context=conversation_context)
    result: DeviationAnalysisOutput = llm.invoke(messages)

    return {
        "findings": [f.model_dump() for f in result.findings],
        "analysis_status": result.status,
        "analyst_reasoning": result.reasoning,
        "missing_info": result.missing_info,
        "iteration": state.get("iteration", 0) + 1,
    }


def conversational_reply(state: AgentState) -> dict:
    query = state["query"]
    conversation_history = state.get("messages", [])

    llm = ChatAnthropic(model=DEFAULT_MODEL, temperature=0)
    messages = build_conversational_messages(query, conversation_history)
    response = llm.invoke(messages)

    return {
        "report": response.content,
        "messages": [AIMessage(content=response.content)],
    }


def format_report(state: AgentState) -> dict:
    if state.get("analysis_status") == "failed":
        report = f"# Analysis Failed\n\n{state.get('error', 'Unknown error')}"
        return {
            "report": report,
            "messages": [AIMessage(content=report)],
        }

    findings_data = state.get("findings", [])
    findings = [
        DeviationFinding(**f) if isinstance(f, dict) else f for f in findings_data
    ]

    if not findings:
        report = "# Protocol Deviation Report\n\nNo deviations identified."
        return {
            "report": report,
            "messages": [AIMessage(content=report)],
        }

    severity_order = [
        SeverityLevel.CRITICAL,
        SeverityLevel.MAJOR,
        SeverityLevel.MINOR,
        SeverityLevel.INFORMATIONAL,
    ]
    counts = {s: sum(1 for f in findings if f.severity == s) for s in severity_order}
    summary_parts = [f"- **{s.value.capitalize()}**: {c}" for s, c in counts.items() if c]

    lines = [
        "# Protocol Deviation Report",
        "",
        f"**Total deviations found**: {len(findings)}",
        "",
        "## Summary by Severity",
        *summary_parts,
        "",
    ]

    for i, f in enumerate(findings, 1):
        lines.extend([
            f"## Deviation {i}: {f.deviation_type.value.replace('_', ' ').title()}",
            "",
            f"**Severity**: {f.severity.value.capitalize()}",
            f"**Protocol Reference**: {f.protocol_reference}",
            "",
            f"**Description**: {f.description}",
            "",
            f"**Evidence**: {f.evidence}",
            "",
            f"**Recommendation**: {f.recommendation}",
            "",
        ])

    if state.get("analyst_reasoning"):
        lines.extend([
            "## Analyst Reasoning",
            "",
            state["analyst_reasoning"],
        ])

    report = "\n".join(lines)
    return {
        "report": report,
        "messages": [AIMessage(content=report)],
    }
