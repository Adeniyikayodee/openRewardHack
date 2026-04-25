"""tests for synthesize_events — fully deterministic, no network."""
import random

import pytest

from scripts.synthesize_events import synthesize_dynamic_events, synthesize_traffic_events

NODES    = [{"idx": i} for i in range(10)]
VEHICLES = [{"id": f"v-{i}", "capacity_seats": 16} for i in range(4)]
HORIZON  = 960

# helper: call with nodes so road_disruption events can be generated
def _dynamic(n, seed=42):
    return synthesize_dynamic_events(VEHICLES, [], n, HORIZON, random.Random(seed), nodes=NODES)


# ---------------------------------------------------------------------------
# traffic events
# ---------------------------------------------------------------------------

def test_traffic_event_count():
    evs = synthesize_traffic_events(NODES, None, 6, random.Random(42))
    assert len(evs) == 6


def test_traffic_event_keys():
    evs = synthesize_traffic_events(NODES, None, 3, random.Random(1))
    for e in evs:
        for key in ("t_reveal", "node_a", "node_b", "speed_factor", "reason"):
            assert key in e, f"missing key {key}"


def test_traffic_event_nodes_in_range():
    evs = synthesize_traffic_events(NODES, None, 9, random.Random(7))
    for e in evs:
        assert 0 <= e["node_a"] < len(NODES)
        assert 0 <= e["node_b"] < len(NODES)
        assert e["node_a"] != e["node_b"]


def test_traffic_speed_factor_range():
    evs = synthesize_traffic_events(NODES, None, 12, random.Random(3))
    for e in evs:
        assert 0.3 <= e["speed_factor"] <= 0.7, f"speed_factor {e['speed_factor']} out of range"


def test_traffic_events_sorted_by_t_reveal():
    evs = synthesize_traffic_events(NODES, None, 9, random.Random(1))
    times = [e["t_reveal"] for e in evs]
    assert times == sorted(times)


def test_traffic_zero_events():
    evs = synthesize_traffic_events(NODES, None, 0, random.Random(0))
    assert evs == []


def test_traffic_events_deterministic():
    a = synthesize_traffic_events(NODES, None, 6, random.Random(42))
    b = synthesize_traffic_events(NODES, None, 6, random.Random(42))
    assert a == b


# ---------------------------------------------------------------------------
# dynamic events
# ---------------------------------------------------------------------------

def test_dynamic_zero_events():
    assert _dynamic(0) == []


def test_dynamic_event_types_present():
    types = {e["type"] for e in _dynamic(9, seed=7)}
    assert "vehicle_breakdown" in types
    assert "new_request" in types
    assert "road_disruption" in types


def test_dynamic_breakdown_not_first_hour():
    for e in _dynamic(9, seed=5):
        if e["type"] == "vehicle_breakdown":
            assert e["t"] >= 60, f"breakdown at t={e['t']} before first hour"


def test_dynamic_events_sorted_by_t():
    times = [e["t"] for e in _dynamic(12, seed=2)]
    assert times == sorted(times)


def test_dynamic_events_deterministic():
    assert _dynamic(6, seed=99) == _dynamic(6, seed=99)


def test_dynamic_late_request_placeholder_ids():
    # ids are assigned in generation order then sorted by t, check pattern only
    evs = _dynamic(9, seed=11)
    late = [e for e in evs if e["type"] == "new_request"]
    ids = {e["request_id"] for e in late}
    expected = {f"r-late-{i}" for i in range(len(late))}
    assert ids == expected, f"unexpected ids: {ids}"


# ---------------------------------------------------------------------------
# road_disruption events
# ---------------------------------------------------------------------------

def test_road_disruption_required_keys():
    evs = _dynamic(10, seed=3)
    for e in [e for e in evs if e["type"] == "road_disruption"]:
        for key in ("t", "node_a", "node_b", "speed_factor", "duration_minutes", "reason"):
            assert key in e, f"missing key {key}"


def test_road_disruption_nodes_differ():
    evs = _dynamic(10, seed=4)
    for e in [e for e in evs if e["type"] == "road_disruption"]:
        assert e["node_a"] != e["node_b"]


def test_road_disruption_speed_factor_range():
    evs = _dynamic(10, seed=5)
    for e in [e for e in evs if e["type"] == "road_disruption"]:
        assert 0.2 <= e["speed_factor"] <= 0.6, f"speed_factor {e['speed_factor']} out of range"


def test_road_disruption_duration_range():
    evs = _dynamic(10, seed=6)
    for e in [e for e in evs if e["type"] == "road_disruption"]:
        assert 60 <= e["duration_minutes"] <= 360, f"duration {e['duration_minutes']} out of range"


def test_road_disruption_reason_is_string():
    evs = _dynamic(10, seed=7)
    for e in [e for e in evs if e["type"] == "road_disruption"]:
        assert isinstance(e["reason"], str) and len(e["reason"]) > 0


def test_road_disruption_absent_without_nodes():
    # when nodes=None, road_disruption events should not be generated
    evs = synthesize_dynamic_events(VEHICLES, [], 10, HORIZON, random.Random(42), nodes=None)
    types = {e["type"] for e in evs}
    assert "road_disruption" not in types
