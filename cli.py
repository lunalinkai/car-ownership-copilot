"""Terminal chat loop for the Car Ownership Copilot.

Run with:  python cli.py
"""

from __future__ import annotations

import os
import sys

from langchain_core.messages import AIMessage, HumanMessage

from car_copilot.graph import run_turn

BANNER = "🚗 Jerry Car Ownership Copilot  —  type 'exit' to quit\n"


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY first:  export OPENAI_API_KEY=sk-...", file=sys.stderr)
        return 1

    print(BANNER)
    history: list = []
    while True:
        try:
            user = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue

        history.append(HumanMessage(content=user))
        result = run_turn(history)
        history.append(AIMessage(content=result["answer"]))

        print(f"\ncopilot › [{result['route']}] {result['answer']}")
        print(
            f"  ({result['latency_ms']:.0f} ms · {result['total_tokens']} tokens · "
            f"${result['cost_usd']:.5f})\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
