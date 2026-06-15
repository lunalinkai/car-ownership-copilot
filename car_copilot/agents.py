"""Specialist agents.

Each specialist is a system prompt + a set of tools. `run_specialist` runs a
bounded ReAct-style loop: the model decides whether to call a tool, we execute
the tool, feed the result back, and repeat until it produces a final answer.
Only that final answer is written back to the shared graph state — the
intermediate tool calls stay local to the specialist.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from . import tools as T
from .llm import get_llm

MAX_TOOL_STEPS = 5

SPECIALISTS = {
    "insurance": {
        "label": "Insurance",
        "tools": T.INSURANCE_TOOLS,
        "system": (
            "You are Jerry's auto-insurance specialist. Help users compare and understand "
            "insurance quotes. Call get_insurance_quote once you have make, model, year, ZIP "
            "code, and the primary driver's age. If anything is missing, ask for it concisely "
            "in one message. Present the quotes as a short comparison and remind the user they "
            "are estimates. Never invent prices — only report what the tool returns."
        ),
    },
    "maintenance": {
        "label": "Maintenance",
        "tools": T.MAINTENANCE_TOOLS,
        "system": (
            "You are Jerry's maintenance & service specialist. Use get_maintenance_schedule to "
            "tell users what service is due based on their vehicle and mileage. You can book an "
            "appointment with book_service_appointment, but ONLY after the user has explicitly "
            "confirmed the exact service and date. If they have not confirmed, summarize what "
            "you intend to book and ask them to confirm first. Be concise."
        ),
    },
    "recall": {
        "label": "Recalls",
        "tools": T.RECALL_TOOLS,
        "system": (
            "You are Jerry's vehicle safety & recall specialist. Use lookup_recalls with the "
            "make, model, and year. If there are open recalls, explain the risk and the (free) "
            "remedy plainly and urge the user to schedule the fix. If there are none, reassure "
            "them. Ask for any missing vehicle details first."
        ),
    },
    "cost": {
        "label": "Repair Cost",
        "tools": T.COST_TOOLS,
        "system": (
            "You are Jerry's repair-cost specialist. Use estimate_repair_cost to give realistic "
            "parts + labor ranges. Ask for the repair type and the vehicle if they are missing. "
            "Always present a low–high range and note that regional rates vary."
        ),
    },
    "general": {
        "label": "General",
        "tools": [],
        "system": (
            "You are Jerry's friendly Car Ownership Copilot. Answer general car-ownership "
            "questions helpfully and briefly. When a request would be better served by a quote, "
            "a recall check, a maintenance schedule, or a repair-cost estimate, let the user "
            "know you can do that too. Keep answers short."
        ),
    },
}


def run_specialist(name: str, history: list, model: Optional[str] = None) -> AIMessage:
    """Run one specialist over the conversation and return its final answer."""
    spec = SPECIALISTS[name]
    tools = spec["tools"]
    tools_by_name = {t.name: t for t in tools}

    llm = get_llm(model=model)
    if tools:
        llm = llm.bind_tools(tools)

    convo = [SystemMessage(content=spec["system"])] + list(history)

    for _ in range(MAX_TOOL_STEPS):
        ai = llm.invoke(convo)
        convo.append(ai)
        tool_calls = getattr(ai, "tool_calls", None)
        if not tool_calls:
            return AIMessage(content=ai.content or "")
        for call in tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                result = {"error": f"unknown tool {call['name']}"}
            else:
                try:
                    result = tool.invoke(call["args"])
                except Exception as exc:  # surface tool errors back to the model
                    result = {"error": str(exc)}
            convo.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # Hit the step cap — force one final, tool-free answer.
    final = get_llm(model=model).invoke(
        convo + [SystemMessage(content="Summarize your best answer for the user now, without tools.")]
    )
    return AIMessage(content=final.content or "I wasn't able to complete that request.")
