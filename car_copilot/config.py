"""Central configuration.

The job description calls for "balancing latency, cost, and accuracy." That
trade-off lives here: gpt-4o-mini is the default (fast + cheap), and the
pricing table powers the per-turn cost estimate surfaced in the UI/CLI.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Fast, low-cost default. Override via env or the UI model selector.
DEFAULT_MODEL = os.getenv("COPILOT_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("COPILOT_TEMPERATURE", "0.1"))

# Approximate USD pricing per 1M tokens: (input, output).
MODEL_PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of a turn from token counts."""
    in_price, out_price = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
