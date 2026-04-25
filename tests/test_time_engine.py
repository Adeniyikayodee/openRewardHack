import json
from src.state import EpisodeState, Stop
from src.time_engine import apply_tick, recompute_etas, handle_breakdown
from src.feasibility import check_feasibility


def make_env(fixture):
    task = json.load(open(f"tests/fixtures/{fixture}"))
    s = EpisodeState(task=task)
    s.initialize()
    return s


def test_tick_reveals_traffic():
    s = make_env("traffic_at_60.json")
    apply_tick(s, 30)
    assert s.revealed_traffic == {}
    apply_tick(s, 40)  # now t=70 > 60
    assert s.revealed_traffic, s.revealed_traffic


def test_tick_processes_completion():
    s = make_env("trivial_task.json")
    pu = Stop(request_id="r-0", kind="pickup",
              node_idx=s.task["requests"][0]["pickup_node_idx"], eta_minutes=20)
    do = Stop(request_id="r-0", kind="dropoff",
              node_idx=s.task["requests"][0]["dropoff_node_idx"], eta_minutes=40)
    s.routes["v-0"] = [pu, do]
    s.request_status["r-0"] = "assigned"
    delta, log = apply_tick(s, 50)
    assert "r-0" in s.served_requests
    assert delta > 0


def test_breakdown_returns_to_pending():
    s = make_env("trivial_task.json")
    pu = Stop(request_id="r-0", kind="pickup", node_idx=1, eta_minutes=200)
    do = Stop(request_id="r-0", kind="dropoff", node_idx=2, eta_minutes=300)
    s.routes["v-0"] = [pu, do]
    s.request_status["r-0"] = "assigned"
    handle_breakdown(s, "v-0")
    assert s.vehicle_status["v-0"] == "broken"
    assert s.request_status["r-0"] == "pending"
    assert s.routes["v-0"] == []


def test_new_request_released():
    s = make_env("late_request_task.json")
    assert s.request_status["r-late-0"] == "unreleased"
    apply_tick(s, 200)
    assert s.request_status["r-late-0"] == "pending"
