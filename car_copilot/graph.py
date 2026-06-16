"""The supervisor graph.

    START → router → {insurance | maintenance | recall | cost | general | out_of_scope} → END

The router is an LLM with structured output that classifies each request into
exactly one destination. `out_of_scope` is a guardrail node that answers
statically (no LLM call) to keep the agent on-topic and save latency/cost.
`run_turn` wraps a graph invocation and reports route, latency, tokens, and an
estimated dollar cost.
"""

from __future__ import annotations

import time
from typing import Literal, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .agents import run_specialist
from .config import DEFAULT_MODEL, estimate_cost
from .llm import get_llm
from .state import CopilotState

Destination = Literal["insurance", "maintenance", "recall", "cost", "general", "out_of_scope"]

ROUTER_SYSTEM = (
    "You are the router for Jerry's Car Ownership Copilot. Read the full conversation and "
    "classify the user's most recent request into exactly one destination:\n"
    "- insurance: quotes, premiums, coverage, switching carriers\n"
    "- maintenance: what service is due, schedules, oil changes, booking service appointments\n"
    "- recall: safety recalls, defects, NHTSA campaigns\n"
    "- cost: how much a repair should cost, repair/ownership cost estimates\n"
    "- general: greetings or general car-ownership advice that needs no tool\n"
    "- out_of_scope: anything not about owning or maintaining a car\n"
    "Pick the single best destination."
)

GUARDRAIL_MESSAGE = (
    "I'm Jerry's Car Ownership Copilot, so I can only help with car ownership — insurance, "
    "maintenance, recalls, and repair costs. Is there something about your vehicle I can help with?"
)


class Route(BaseModel):
    """Structured routing decision from the supervisor."""

    destination: Destination = Field(..., description="The single best specialist to handle this.")
    reasoning: str = Field(..., description="One short sentence explaining the choice.")


def _model_of(state: CopilotState) -> str:
    return state.get("model") or DEFAULT_MODEL


def _key_of(state: CopilotState) -> Optional[str]:
    return state.get("api_key") or None


def _router_node(state: CopilotState) -> dict:
    llm = get_llm(
        model=_model_of(state), temperature=0, api_key=_key_of(state)
    ).with_structured_output(Route)
    decision = llm.invoke([SystemMessage(content=ROUTER_SYSTEM)] + list(state["messages"]))
    return {"route": decision.destination}


def _make_specialist_node(name: str):
    def node(state: CopilotState) -> dict:
        answer = run_specialist(
            name, state["messages"], model=_model_of(state), api_key=_key_of(state)
        )
        return {"messages": [answer]}

    return node


def _out_of_scope_node(state: CopilotState) -> dict:
    return {"messages": [AIMessage(content=GUARDRAIL_MESSAGE)]}


SPECIALIST_NODES = ["insurance", "maintenance", "recall", "cost", "general"]


def build_graph():
    graph = StateGraph(CopilotState)
    graph.add_node("router", _router_node)
    for name in SPECIALIST_NODES:
        graph.add_node(name, _make_specialist_node(name))
    graph.add_node("out_of_scope", _out_of_scope_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {name: name for name in SPECIALIST_NODES} | {"out_of_scope": "out_of_scope"},
    )
    for name in SPECIALIST_NODES + ["out_of_scope"]:
        graph.add_edge(name, END)
    return graph.compile()


_GRAPH = None


def get_graph():
    """Return the compiled graph (built once, reused)."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


class _UsageTracker(BaseCallbackHandler):
    """Accumulates token usage across every LLM call in a single turn."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        try:
            usage = (response.llm_output or {}).get("token_usage", {}) or {}
            self.input_tokens += usage.get("prompt_tokens", 0) or 0
            self.output_tokens += usage.get("completion_tokens", 0) or 0
        except Exception:
            pass


def run_turn(history: list, model: Optional[str] = None, api_key: Optional[str] = None) -> dict:
    """Run one conversational turn through the graph and return the answer + metrics.

    ``api_key``, when provided, is used for every LLM call in the turn instead of
    any ambient OPENAI_API_KEY. The web app passes the visitor's key here.
    """
    model = model or DEFAULT_MODEL
    tracker = _UsageTracker()
    start = time.perf_counter()
    result = get_graph().invoke(
        {"messages": history, "route": "", "model": model, "api_key": api_key or ""},
        config={"callbacks": [tracker]},
    )
    latency_ms = (time.perf_counter() - start) * 1000

    answer = result["messages"][-1].content
    return {
        "answer": answer,
        "route": result.get("route", ""),
        "messages": result["messages"],
        "latency_ms": round(latency_ms, 1),
        "input_tokens": tracker.input_tokens,
        "output_tokens": tracker.output_tokens,
        "total_tokens": tracker.input_tokens + tracker.output_tokens,
        "cost_usd": estimate_cost(model, tracker.input_tokens, tracker.output_tokens),
    }


def route_only(query: str, model: Optional[str] = None, api_key: Optional[str] = None) -> str:
    """Just the routing decision for a single query — used by the eval harness."""
    llm = get_llm(model=model, temperature=0, api_key=api_key).with_structured_output(Route)
    decision = llm.invoke([SystemMessage(content=ROUTER_SYSTEM), HumanMessage(content=query)])
    return decision.destination
