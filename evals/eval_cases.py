"""Labeled routing cases.

A real production system would have hundreds of these, version-controlled and
expanded from misroutes seen in production. This small set is enough to catch
routing regressions when you change a prompt or swap models.
"""

CASES = [
    # insurance
    {"query": "What's the cheapest insurance for my 2020 Camry?", "expected": "insurance"},
    {"query": "Can I lower my premium if I switch carriers?", "expected": "insurance"},
    {"query": "I'm 24 in ZIP 30301, quote me on a 2018 Accord.", "expected": "insurance"},
    # maintenance
    {"query": "My car has 60k miles, what service is due?", "expected": "maintenance"},
    {"query": "Book me an oil change for next Tuesday.", "expected": "maintenance"},
    {"query": "When should I rotate my tires?", "expected": "maintenance"},
    # recall
    {"query": "Any recalls on a 2019 Honda Civic?", "expected": "recall"},
    {"query": "Is my car affected by a safety defect?", "expected": "recall"},
    # cost
    {"query": "How much should a new alternator cost?", "expected": "cost"},
    {"query": "What's a fair price for brake pads on a BMW?", "expected": "cost"},
    # general
    {"query": "Hi there!", "expected": "general"},
    {"query": "What does TPMS mean?", "expected": "general"},
    # out of scope
    {"query": "What's the weather in Paris tomorrow?", "expected": "out_of_scope"},
    {"query": "Write me a poem about the ocean.", "expected": "out_of_scope"},
]
