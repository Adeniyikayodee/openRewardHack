"""traffic event and dynamic episode event synthesis.
all functions are deterministic given the rng argument — same rng seed always
produces the same events. late-request placeholder ids (r-late-N) are resolved
by the caller in generate_tasks.py.
"""
import random


def synthesize_traffic_events(
    nodes: list[dict],
    distance_matrix: list[list[float]] | None,
    n_events: int,
    rng: random.Random,
) -> list[dict]:
    """generate n_events plausible traffic disruptions over the episode day.

    mix:
      ~1/3 morning rush   (t_reveal=0,  episode 00:00–03:30 = real 06:00–09:30)
      ~1/3 midday incident (t_reveal=120–540)
      ~1/3 evening rush   (t_reveal=480–600)

    returns list sorted by t_reveal.
    """
    if n_events == 0:
        return []

    events = []
    n = len(nodes)

    def random_pair() -> tuple[int, int]:
        a = rng.randrange(n)
        b = rng.randrange(n)
        while b == a:
            b = rng.randrange(n)
        return a, b

    rush_morning = max(1, n_events // 3)
    incidents    = max(1, n_events // 3)
    rush_evening = n_events - rush_morning - incidents

    for _ in range(rush_morning):
        a, b = random_pair()
        events.append({
            "t_reveal": 0,
            "node_a": a,
            "node_b": b,
            "speed_factor": round(rng.uniform(0.4, 0.7), 2),
            "reason": "synthesized: morning rush-hour congestion",
        })

    for _ in range(incidents):
        a, b = random_pair()
        events.append({
            "t_reveal": rng.randint(120, 540),
            "node_a": a,
            "node_b": b,
            "speed_factor": round(rng.uniform(0.3, 0.6), 2),
            "reason": rng.choice([
                "synthesized: a-road incident",
                "synthesized: roadworks",
                "synthesized: lane closure",
                "synthesized: collision cleanup",
            ]),
        })

    for _ in range(rush_evening):
        a, b = random_pair()
        events.append({
            "t_reveal": rng.randint(480, 600),
            "node_a": a,
            "node_b": b,
            "speed_factor": round(rng.uniform(0.4, 0.7), 2),
            "reason": "synthesized: evening rush-hour congestion",
        })

    events.sort(key=lambda e: e["t_reveal"])
    return events


def synthesize_dynamic_events(
    vehicles: list[dict],
    requests_: list[dict],
    n_events: int,
    horizon: int,
    rng: random.Random,
    nodes: list[dict] | None = None,
) -> list[dict]:
    """generate n_events mid-episode disruptions.

    mix:
      ~1/3 vehicle breakdowns (not in first hour)
      ~1/4 capacity drops
      ~1/5 road disruptions — construction or surface damage on a segment
      remainder new requests (late arrivals; ids are placeholders r-late-N)

    returns list sorted by t.
    """
    if n_events == 0:
        return []

    events = []
    n_breakdowns      = max(0, n_events // 3)
    n_capacity_drops  = max(0, n_events // 4)
    n_road_disruptions = max(0, n_events // 5) if nodes and len(nodes) >= 2 else 0
    n_new_requests    = n_events - n_breakdowns - n_capacity_drops - n_road_disruptions

    for _ in range(n_breakdowns):
        v = rng.choice(vehicles)
        events.append({
            "t": rng.randint(60, max(61, horizon - 120)),
            "type": "vehicle_breakdown",
            "vehicle_id": v["id"],
        })

    for _ in range(n_capacity_drops):
        v = rng.choice(vehicles)
        if v["capacity_seats"] <= 4:
            continue
        events.append({
            "t": rng.randint(120, max(121, horizon - 60)),
            "type": "capacity_drop",
            "vehicle_id": v["id"],
            "new_capacity_seats": max(2, v["capacity_seats"] // 2),
            "reason": rng.choice([
                "wheelchair passenger requires extra space",
                "luggage takes additional seats",
                "vehicle partial mechanical issue limits capacity",
            ]),
        })

    for _ in range(n_road_disruptions):
        a = rng.randrange(len(nodes))
        b = rng.randrange(len(nodes))
        while b == a:
            b = rng.randrange(len(nodes))
        events.append({
            "t": rng.randint(0, max(1, horizon - 240)),
            "type": "road_disruption",
            "node_a": nodes[a]["idx"],
            "node_b": nodes[b]["idx"],
            "speed_factor": round(rng.uniform(0.2, 0.6), 2),
            "duration_minutes": rng.randint(60, 360),
            "reason": rng.choice([
                "road construction: carriageway works",
                "road construction: utility works",
                "road surface deterioration",
                "flooding: surface water on road",
                "emergency pothole repair",
            ]),
        })

    for i in range(n_new_requests):
        events.append({
            "t": rng.randint(120, max(121, horizon - 180)),
            "type": "new_request",
            "request_id": f"r-late-{i}",  # resolved by caller
        })

    events.sort(key=lambda e: e["t"])
    return events
