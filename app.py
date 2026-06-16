"""Streamlit chat UI for the Car Ownership Copilot.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from car_copilot.config import DEFAULT_MODEL, MODEL_PRICING
from car_copilot.graph import run_turn

st.set_page_config(page_title="Car Ownership Copilot", page_icon="🚗", layout="centered")

ROUTE_BADGES = {
    "insurance": "🛡️ Insurance",
    "maintenance": "🔧 Maintenance",
    "recall": "⚠️ Recall",
    "cost": "💲 Repair Cost",
    "general": "💬 General",
    "out_of_scope": "🚫 Out of scope",
}

EXAMPLES = [
    "I'm 23, my 2019 Honda Civic is in 94107 — what would insurance run me?",
    "My RAV4 has 47,000 miles. What service is coming up?",
    "Are there any open recalls on a 2019 Honda Civic?",
    "How much should new brake pads cost on a BMW 3 Series?",
]

# --- Sidebar --------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🚗 Car Ownership Copilot")
    st.caption(
        "A LangGraph multi-agent system: a supervisor routes each question to a "
        "tool-using specialist (insurance · maintenance · recalls · repair cost)."
    )

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        st.caption(
            "Bring your own key — used only for your session, never stored or logged. "
            "Runs on `gpt-4o-mini` (a few cents). [Get a key ↗](https://platform.openai.com/api-keys)"
        )
        if key:
            os.environ["OPENAI_API_KEY"] = key

    model = st.selectbox("Model", list(MODEL_PRICING.keys()), index=0)
    st.caption(f"Default: {DEFAULT_MODEL} — chosen for low latency & cost.")

    st.markdown("**Try an example:**")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.turns = []
        st.rerun()

st.title("Car Ownership Copilot")
st.caption("Ask about insurance, maintenance, recalls, or repair costs for your car.")

if "turns" not in st.session_state:
    st.session_state.turns = []

if not os.getenv("OPENAI_API_KEY"):
    st.info(
        "👈 **Live demo — bring your own OpenAI key.** Paste a key in the sidebar to start "
        "chatting. It's used only for your session and never stored. Here's what it does:"
    )
    st.markdown("#### How it works")
    st.markdown(
        "A **supervisor agent** reads your question and routes it to the right tool-using "
        "specialist — insurance, maintenance, recalls, or repair cost — which answers from "
        "(mock) internal-API tools. Off-topic questions hit a guardrail."
    )
    try:
        st.image(
            "architecture.png",
            caption="Supervisor routes each request to a specialist agent (auto-generated from the LangGraph).",
        )
    except Exception:
        pass

    st.markdown("#### Example exchange")
    with st.chat_message("user"):
        st.markdown("Are there any open recalls on a 2019 Honda Civic?")
    with st.chat_message("assistant"):
        st.markdown(
            "Yes — there's **1 open recall** on a 2019 Honda Civic:\n\n"
            "- **19V-001 — Fuel pump:** may fail and cause an engine stall.\n"
            "- **Fix (free):** the dealer replaces the fuel pump.\n\n"
            "I'd schedule the repair soon."
        )
        st.caption("⚠️ Recall · routed by the supervisor · ~3.6s")

    st.markdown(
        "[View the source on GitHub →](https://github.com/lunalinkai/car-ownership-copilot)"
    )
    st.stop()


def _to_lc(turns: list) -> list:
    out = []
    for t in turns:
        cls = HumanMessage if t["role"] == "user" else AIMessage
        out.append(cls(content=t["content"]))
    return out


def _render(turn: dict) -> None:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and "meta" in turn:
            m = turn["meta"]
            badge = ROUTE_BADGES.get(m["route"], m["route"])
            st.caption(
                f"{badge}  ·  {m['latency_ms']:.0f} ms  ·  "
                f"{m['total_tokens']} tokens  ·  ${m['cost_usd']:.5f}"
            )


# Render existing history.
for turn in st.session_state.turns:
    _render(turn)

prompt = st.chat_input("Type your question...")
if not prompt and "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    user_turn = {"role": "user", "content": prompt}
    st.session_state.turns.append(user_turn)
    _render(user_turn)

    with st.chat_message("assistant"):
        with st.spinner("Routing to a specialist..."):
            try:
                result = run_turn(_to_lc(st.session_state.turns), model=model)
            except Exception as exc:
                result = {
                    "answer": f"⚠️ Something went wrong: {exc}",
                    "route": "",
                    "latency_ms": 0.0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                }
        st.markdown(result["answer"])
        badge = ROUTE_BADGES.get(result["route"], result["route"] or "—")
        st.caption(
            f"{badge}  ·  {result['latency_ms']:.0f} ms  ·  "
            f"{result['total_tokens']} tokens  ·  ${result['cost_usd']:.5f}"
        )

    st.session_state.turns.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "meta": {
                "route": result["route"],
                "latency_ms": result["latency_ms"],
                "total_tokens": result["total_tokens"],
                "cost_usd": result["cost_usd"],
            },
        }
    )
