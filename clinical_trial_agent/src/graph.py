from langgraph.graph import END, START, StateGraph

from clinical_trial_agent.src.nodes import (
    analyze_deviation,
    classify_intent,
    conversational_reply,
    format_report,
)
from clinical_trial_agent.src.retriever import retrieve
from clinical_trial_agent.src.router import route_after_analysis, route_after_intent
from clinical_trial_agent.src.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retriever", retrieve)
    graph.add_node("analyst", analyze_deviation)
    graph.add_node("conversational_reply", conversational_reply)
    graph.add_node("format_report", format_report)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route_after_intent)
    graph.add_edge("retriever", "analyst")
    graph.add_conditional_edges("analyst", route_after_analysis)
    graph.add_edge("format_report", END)
    graph.add_edge("conversational_reply", END)

    return graph.compile()

