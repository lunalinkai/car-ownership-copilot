"""Mock "internal API" tools.

In production these would hit Jerry's real services: a quoting engine, the
service-scheduling system, the NHTSA recall feed, and a parts/labor pricing DB.
Here they return deterministic mock data so the whole agent graph runs
end-to-end with no external dependencies. The tool *contracts* (typed args +
docstrings) are what the LLM reasons over, so they're written as if real.
"""

from __future__ import annotations

import zlib

from langchain_core.tools import tool

CURRENT_YEAR = 2026


# --------------------------------------------------------------------------- #
# Insurance
# --------------------------------------------------------------------------- #
@tool
def get_insurance_quote(make: str, model: str, year: int, zip_code: str, driver_age: int) -> dict:
    """Return estimated 6-month auto-insurance quotes for a vehicle and driver.

    Args:
        make: Vehicle make, e.g. "Toyota".
        model: Vehicle model, e.g. "Camry".
        year: Model year, e.g. 2020.
        zip_code: 5-digit US ZIP code where the car is garaged.
        driver_age: Primary driver's age in years.
    """
    base = 850.0
    age_factor = 1.6 if driver_age < 25 else (1.1 if driver_age < 30 else 1.0)
    if driver_age > 70:
        age_factor *= 1.15
    vehicle_age = max(0, CURRENT_YEAR - year)
    vehicle_factor = 1.25 if vehicle_age < 2 else (1.0 if vehicle_age < 10 else 0.85)
    premium = round(base * age_factor * vehicle_factor)
    return {
        "vehicle": f"{year} {make} {model}",
        "zip_code": zip_code,
        "term": "6 months",
        "quotes": [
            {"carrier": "Progressive", "premium_usd": premium},
            {"carrier": "GEICO", "premium_usd": round(premium * 0.93)},
            {"carrier": "State Farm", "premium_usd": round(premium * 1.08)},
        ],
        "note": "Estimates only; final pricing depends on driving history & coverage.",
    }


# --------------------------------------------------------------------------- #
# Maintenance & service
# --------------------------------------------------------------------------- #
@tool
def get_maintenance_schedule(make: str, model: str, mileage: int) -> dict:
    """Return the next upcoming maintenance items for a vehicle at a given mileage.

    Args:
        make: Vehicle make.
        model: Vehicle model.
        mileage: Current odometer reading in miles.
    """
    intervals = [
        (5000, "Oil & filter change, tire rotation"),
        (15000, "Engine air filter, cabin air filter"),
        (30000, "Brake inspection, transmission fluid check"),
        (60000, "Spark plugs, coolant flush"),
        (90000, "Timing belt inspection, suspension check"),
    ]
    upcoming = []
    for miles, desc in intervals:
        next_due = ((mileage // miles) + 1) * miles
        upcoming.append(
            {"service": desc, "due_at_mileage": next_due, "miles_away": next_due - mileage}
        )
    upcoming.sort(key=lambda x: x["miles_away"])
    return {"vehicle": f"{make} {model}", "current_mileage": mileage, "upcoming": upcoming[:3]}


@tool
def book_service_appointment(
    service_type: str, preferred_date: str, location: str = "nearest Jerry-partner shop"
) -> dict:
    """Book a service appointment. Only call after the user has explicitly confirmed.

    Args:
        service_type: What work to perform, e.g. "oil change".
        preferred_date: Requested date in YYYY-MM-DD form.
        location: Shop or area; defaults to the nearest partner shop.
    """
    conf = zlib.crc32((service_type + preferred_date).encode()) % 100000
    return {
        "status": "confirmed",
        "confirmation_id": f"JRY-{conf:05d}",
        "service_type": service_type,
        "date": preferred_date,
        "location": location,
    }


# --------------------------------------------------------------------------- #
# Recalls & safety
# --------------------------------------------------------------------------- #
@tool
def lookup_recalls(make: str, model: str, year: int) -> dict:
    """Look up open safety recalls for a vehicle (mock NHTSA feed).

    Args:
        make: Vehicle make.
        model: Vehicle model.
        year: Model year.
    """
    known = {
        ("Honda", "Civic", 2019): [
            {
                "campaign": "19V-001",
                "component": "Fuel pump",
                "summary": "Fuel pump may fail, causing an engine stall.",
                "remedy": "Dealer replaces the fuel pump, free of charge.",
            }
        ],
        ("Toyota", "Rav4", 2021): [
            {
                "campaign": "21V-330",
                "component": "Backup camera",
                "summary": "Rearview image may fail to display.",
                "remedy": "Free software update at a dealer.",
            }
        ],
    }
    recalls = known.get((make.title(), model.title(), year))
    if recalls is None:
        # Deterministic synthetic recall so arbitrary vehicles still demo well.
        h = zlib.crc32(f"{make}{model}{year}".lower().encode())
        if h % 3 == 0:
            components = [
                "Airbag inflator",
                "Brake hydraulics",
                "Electrical wiring harness",
                "Seatbelt pretensioner",
            ]
            recalls = [
                {
                    "campaign": f"{str(year)[-2:]}V-{h % 900 + 100}",
                    "component": components[h % len(components)],
                    "summary": "A potential safety defect was identified in this component.",
                    "remedy": "Visit an authorized dealer for a free inspection and repair.",
                }
            ]
        else:
            recalls = []
    return {
        "vehicle": f"{year} {make} {model}",
        "open_recalls": recalls,
        "count": len(recalls),
        "source": "NHTSA (mock)",
    }


# --------------------------------------------------------------------------- #
# Repair cost
# --------------------------------------------------------------------------- #
@tool
def estimate_repair_cost(repair_type: str, make: str, model: str) -> dict:
    """Estimate parts + labor cost for a common repair on a given vehicle.

    Args:
        repair_type: The repair, e.g. "brake pads", "alternator", "battery".
        make: Vehicle make.
        model: Vehicle model.
    """
    table = {
        "brake pads": (150, 300),
        "battery": (120, 280),
        "alternator": (350, 700),
        "timing belt": (500, 1000),
        "transmission": (1800, 3500),
        "ac compressor": (600, 1200),
        "oil change": (40, 90),
        "tires": (400, 900),
        "starter": (300, 650),
        "water pump": (350, 750),
    }
    low, high = table.get(repair_type.lower().strip(), (200, 600))
    luxury = make.title() in {"Bmw", "Mercedes-Benz", "Audi", "Lexus", "Tesla", "Porsche"}
    mult = 1.4 if luxury else 1.0
    return {
        "repair": repair_type,
        "vehicle": f"{make} {model}",
        "estimate_usd": {"low": round(low * mult), "high": round(high * mult)},
        "note": "Parts + labor; regional labor rates vary.",
    }


INSURANCE_TOOLS = [get_insurance_quote]
MAINTENANCE_TOOLS = [get_maintenance_schedule, book_service_appointment]
RECALL_TOOLS = [lookup_recalls]
COST_TOOLS = [estimate_repair_cost]
