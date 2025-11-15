# graphs/paper_graph.py  (excerpt – extend your existing LangGraph graph)
from __future__ import annotations
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from agents.critics import MathCritic, PhysicsCritic, adapt_plan_if_uncertain

def build_paper_graph():
    g = StateGraph()

    # existing nodes…
    g.add_node("analysis", run_analysis)
    g.add_node("posterior_monitor", posterior_monitor)  # new
    g.add_node("math_critic", math_critic_node)         # new
    g.add_node("physics_critic", physics_critic_node)   # new
    g.add_node("planner", planner_node)
    g.add_node("paper_writer", paper_writer)
    g.add_node("done", lambda s: s)

    # flow
    g.add_edge("analysis", "posterior_monitor")
    g.add_edge("posterior_monitor", "math_critic")
    g.add_edge("math_critic", "physics_critic")

    # conditional branch: if uncertain -> replan, else write paper
    def guard_replan(state):
        return state.get("replan", False)

    g.add_conditional_edges("physics_critic", {"replan": guard_replan},
                            {"replan": "planner"},
                            default="paper_writer")

    g.add_edge("planner", "analysis")      # adaptive loop
    g.add_edge("paper_writer", "done")
    g.set_entry_point("analysis")
    g.set_finish_point("done")
    return g

def posterior_monitor(state: Dict[str, Any]) -> Dict[str, Any]:
    post = state.get("posteriors", {})  # {param: {"q05":..., "q95":...}}
    thresholds = state.get("posterior_thresholds", {"H0": 5.0, "Omega_m": 0.1})
    new_plan = adapt_plan_if_uncertain(state["plan"], post, thresholds)
    must_replan = (new_plan != state["plan"])
    state["plan"] = new_plan
    state["replan"] = must_replan
    return state

def math_critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ct = MathCritic()(state.get("manuscript", ""), extras=state)
    state.setdefault("reviews", {})["math"] = ct.__dict__
    return state

def physics_critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ct = PhysicsCritic()(state.get("manuscript", ""), extras=state)
    state.setdefault("reviews", {})["physics"] = ct.__dict__
    # escalate replan if severe issues detected
    if ct.severity == "error":
        state["replan"] = True
    return state
