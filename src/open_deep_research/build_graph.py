from langgraph.graph import StateGraph, START, END
from open_deep_research.state import AgentState
from open_deep_research.nodes.planner import planner 
from open_deep_research.nodes.research import research
from open_deep_research.nodes.assess_gaps import assess_gaps, replan, writer

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("planner", planner)
    g.add_node("research", research)
    g.add_node("assess_gaps", assess_gaps)
    g.add_node("replan", replan)
    g.add_node("writer", writer)

    g.add_edge(START, "planner")
    g.add_edge("planner", "research")
    g.add_edge("research", "assess_gaps")

    g.add_edge("replan", "research")
    g.add_edge("writer", END)

    return g.compile()