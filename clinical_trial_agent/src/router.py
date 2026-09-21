from clinical_trial_agent.src.state import AgentState


def route_after_intent(state: AgentState) -> str:
    if state.get("response_mode") == "conversational":
        return "conversational_reply"
    return "retriever"


def route_after_analysis(state: AgentState) -> str:
    status = state.get("analysis_status", "failed")
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    if status == "needs_more_context" and iteration < max_iter:
        return "retriever"
    return "format_report"
