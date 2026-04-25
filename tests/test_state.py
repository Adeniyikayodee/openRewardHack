import json
import pytest
from src.state import EpisodeState, Stop


def load(fixture):
    task = json.load(open(f"tests/fixtures/{fixture}"))
    s = EpisodeState(task=task)
    s.initialize()
    return s


def test_initialize_vehicles():
    s = load("trivial_task.json")
    assert "v-0" in s.vehicle_status
    assert s.vehicle_status["v-0"] == "available"
    assert s.routes["v-0"] == []


def test_initialize_requests_pending():
    s = load("trivial_task.json")
    assert s.request_status["r-0"] == "pending"


def test_initialize_requests_unreleased():
    s = load("late_request_task.json")
    assert s.request_status["r-late-0"] == "unreleased"


def test_pending_request_ids():
    s = load("trivial_task.json")
    assert s.pending_request_ids() == ["r-0"]


def test_route_load_at_position():
    s = load("trivial_task.json")
    pu = Stop(request_id="r-0", kind="pickup", node_idx=1, eta_minutes=10)
    do = Stop(request_id="r-0", kind="dropoff", node_idx=2, eta_minutes=20)
    s.routes["v-0"] = [pu, do]
    expected = s.request("r-0")["passengers"]
    assert s.route_load_at_position("v-0", 0) == 0
    assert s.route_load_at_position("v-0", 1) == expected   # after pickup
    assert s.route_load_at_position("v-0", 2) == 0          # after dropoff


def test_vehicle_capacity_override():
    s = load("trivial_task.json")
    s.vehicle_capacity_override["v-0"] = {"capacity_seats": 4}
    v = s.vehicle("v-0")
    assert v["capacity_seats"] == 4
