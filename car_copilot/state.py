"""Shared graph state.

`messages` accumulates the conversation (the `add_messages` reducer appends
new messages instead of overwriting). `route` records the supervisor's
decision; `model` lets the UI swap models per turn without rebuilding the graph.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class CopilotState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str
    model: str
    # Per-request key. Passed explicitly so a shared server process never
    # uses an ambient key or leaks one visitor's key to another.
    api_key: str
