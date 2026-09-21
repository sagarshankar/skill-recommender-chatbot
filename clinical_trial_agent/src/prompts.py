from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    from clinical_trial_agent.src.state import AgentState

SYSTEM_PROMPT = """You are a clinical trial protocol compliance analyst. Your role is to \
identify protocol deviations given protocol excerpts and a clinical scenario.

Instructions:
- Carefully compare the scenario against the protocol rules in the provided excerpts.
- Identify ALL deviations, classifying each by type and severity.
- For each deviation, cite the specific protocol section that is violated.
- If the provided protocol excerpts do not contain enough information to fully \
analyze the scenario, set status to "needs_more_context" and describe what \
additional information is needed in missing_info.
- If you can fully analyze the scenario, set status to "complete".
- Only set status to "failed" if the scenario is unintelligible or completely \
unrelated to clinical trials.
- Be precise and evidence-based. Do not speculate beyond what the protocol states."""

INTENT_CLASSIFICATION_PROMPT = """You are a routing classifier for a clinical trial protocol \
deviation analysis system. Given a conversation history and the user's latest message, determine:

1. **mode**: Is the user describing a clinical scenario that needs deviation analysis, or are \
they asking a conversational follow-up (clarification, general question, greeting, etc.)?
   - "deviation_analysis": The message contains or implies a clinical scenario to check against \
the protocol (e.g., dosing, eligibility, procedure violations, safety reporting).
   - "conversational": The message is a follow-up question about a previous analysis, a \
clarification request, a greeting, or a general question not requiring fresh protocol retrieval.

2. **rewritten_query**: Rewrite the user's message as a self-contained query by resolving \
any anaphora or implicit references from the conversation history. For example:
   - "What about for patients over 75 kg?" → "What is the correct sutimlimab dose for \
patients weighing 75 kg or more?"
   - "Is that a deviation?" → Expand to the full scenario being referenced.
   - If the query is already self-contained, return it unchanged."""

CONVERSATIONAL_SYSTEM_PROMPT = """You are a clinical trial protocol compliance analyst engaged \
in a conversation. You have access to the conversation history below.

Instructions:
- Answer the user's follow-up question based on the conversation context and your \
knowledge of the protocol analysis performed so far.
- If the question asks about a previous analysis, reference the findings already discussed.
- If you cannot answer from the conversation context, say so clearly.
- Stay in your role as a compliance analyst. Do not provide medical advice, financial advice, \
or information outside your scope.
- Be concise and precise."""


def build_analyst_messages(
    state: AgentState, conversation_context: list | None = None
) -> list:
    docs = state.get("retrieved_docs", [])
    numbered_docs = "\n\n".join(
        f"[Excerpt {i + 1}]\n{doc}" for i, doc in enumerate(docs)
    )

    query = state.get("rewritten_query") or state["query"]

    user_parts = [
        f"## Clinical Scenario\n{query}",
        f"## Protocol Excerpts\n{numbered_docs}",
    ]

    if conversation_context:
        context_lines = []
        for msg in conversation_context[-6:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content if hasattr(msg, "content") else str(msg)
            context_lines.append(f"{role}: {content}")
        if context_lines:
            user_parts.insert(
                0, f"## Conversation Context\n" + "\n".join(context_lines)
            )

    iteration = state.get("iteration", 0)
    if iteration > 0:
        prev_findings = state.get("findings", [])
        if prev_findings:
            user_parts.append(
                f"## Previous Findings (iteration {iteration})\n"
                f"Review and refine these with the additional context:\n"
                f"{prev_findings}"
            )
        if state.get("missing_info"):
            user_parts.append(
                f"## Previously Missing Information\n{state['missing_info']}"
            )

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="\n\n".join(user_parts)),
    ]


def build_intent_classification_messages(
    query: str, conversation_history: list
) -> list:
    parts = []
    for msg in conversation_history[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content if hasattr(msg, "content") else str(msg)
        parts.append(f"{role}: {content}")

    history_text = "\n".join(parts) if parts else "(no prior conversation)"

    return [
        SystemMessage(content=INTENT_CLASSIFICATION_PROMPT),
        HumanMessage(
            content=f"## Conversation History\n{history_text}\n\n"
            f"## Current User Message\n{query}"
        ),
    ]


def build_conversational_messages(
    query: str, conversation_history: list
) -> list:
    messages = [SystemMessage(content=CONVERSATIONAL_SYSTEM_PROMPT)]
    for msg in conversation_history:
        messages.append(msg)
    messages.append(HumanMessage(content=query))
    return messages
