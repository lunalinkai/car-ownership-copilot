"""Offline smoke test — verifies the graph wires up and tools run.

Does NOT call OpenAI, so it needs no API key. Run with:  python smoke_test.py
"""

from __future__ import annotations

from car_copilot import tools as T
from car_copilot.agents import SPECIALISTS
from car_copilot.graph import SPECIALIST_NODES, build_graph


def main() -> int:
    # 1. Graph compiles and has every expected node.
    graph = build_graph()
    nodes = set(graph.get_graph().nodes)
    for name in SPECIALIST_NODES + ["router", "out_of_scope"]:
        assert name in nodes, f"missing node: {name}"
    print(f"✅ graph compiled with nodes: {sorted(n for n in nodes if n not in ('__start__', '__end__'))}")

    # 2. Every specialist has a system prompt.
    for name, spec in SPECIALISTS.items():
        assert spec["system"].strip(), f"{name} has no system prompt"
    print(f"✅ {len(SPECIALISTS)} specialists configured")

    # 3. Mock tools return sensible, deterministic shapes.
    q = T.get_insurance_quote.invoke(
        {"make": "Honda", "model": "Civic", "year": 2019, "zip_code": "94107", "driver_age": 23}
    )
    assert q["quotes"] and all("premium_usd" in c for c in q["quotes"])
    print(f"✅ insurance quote: {[c['premium_usd'] for c in q['quotes']]}")

    sched = T.get_maintenance_schedule.invoke({"make": "Toyota", "model": "RAV4", "mileage": 47000})
    assert sched["upcoming"] and sched["upcoming"][0]["miles_away"] >= 0
    print(f"✅ maintenance schedule: next due in {sched['upcoming'][0]['miles_away']} mi")

    rec = T.lookup_recalls.invoke({"make": "Honda", "model": "Civic", "year": 2019})
    assert rec["count"] == 1
    print(f"✅ recall lookup: {rec['count']} recall for 2019 Civic")

    booking = T.book_service_appointment.invoke(
        {"service_type": "oil change", "preferred_date": "2026-06-20"}
    )
    assert booking["status"] == "confirmed" and booking["confirmation_id"].startswith("JRY-")
    print(f"✅ booking: {booking['confirmation_id']}")

    cost = T.estimate_repair_cost.invoke({"repair_type": "brake pads", "make": "BMW", "model": "3 Series"})
    assert cost["estimate_usd"]["high"] > cost["estimate_usd"]["low"]
    print(f"✅ repair cost: ${cost['estimate_usd']['low']}–${cost['estimate_usd']['high']} (luxury markup applied)")

    print("\nAll smoke tests passed. Add an OPENAI_API_KEY to run the live agent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
