"""Routing-accuracy eval.

Runs every labeled case through the supervisor and reports per-case results
plus overall accuracy. This is the smallest useful "evaluation framework":
a regression gate you can run before shipping a prompt or model change.

Run with:  python -m evals.run_eval
"""

from __future__ import annotations

import os
import sys

from car_copilot.graph import route_only

from evals.eval_cases import CASES


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY first:  export OPENAI_API_KEY=sk-...", file=sys.stderr)
        return 1

    print(f"Running {len(CASES)} routing cases...\n")
    correct = 0
    for case in CASES:
        got = route_only(case["query"])
        ok = got == case["expected"]
        correct += ok
        mark = "✅" if ok else "❌"
        line = f"{mark} expected={case['expected']:<13} got={got:<13} | {case['query']}"
        if not ok:
            line += "   <-- MISROUTE"
        print(line)

    accuracy = correct / len(CASES)
    print(f"\nRouting accuracy: {correct}/{len(CASES)} = {accuracy:.0%}")
    # Non-zero exit if below threshold, so this can gate CI.
    return 0 if accuracy >= 0.85 else 2


if __name__ == "__main__":
    raise SystemExit(main())
