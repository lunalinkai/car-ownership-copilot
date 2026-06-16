"""Central configuration.

The job description calls for "balancing latency, cost, and accuracy." That
trade-off lives here: gpt-5.4-mini is the default (fast + cheap), with the
flagship gpt-5.5 a dropdown away. The pricing table powers the per-turn cost
estimate surfaced in the UI/CLI.

Pricing verified against OpenAI's published rates (June 2026).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Fast, low-cost default. Override via env or the UI model selector.
DEFAULT_MODEL = os.getenv("COPILOT_MODEL", "gpt-5.4-mini")
TEMPERATURE = float(os.getenv("COPILOT_TEMPERATURE", "0.1"))

# Approximate USD pricing per 1M tokens: (input, output). Current GPT-5.x line.
MODEL_PRICING = {
    "gpt-5.4-mini": (0.75, 4.50),   # default — fast, cheap workhorse
    "gpt-5.5": (5.00, 30.00),       # flagship
    "gpt-5.4": (2.50, 15.00),       # standard
    "gpt-5.4-nano": (0.20, 1.25),   # cheapest / lowest latency
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of a turn from token counts."""
    in_price, out_price = MODEL_PRICING.get(model, (0.75, 4.50))  # default to gpt-5.4-mini rates
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
