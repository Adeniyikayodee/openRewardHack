import json
from src.state import EpisodeState
from src.feasibility import check_feasibility


def make_env(fixture):
    task = json.load(open(f"tests/fixtures/{fixture}"))
    s = EpisodeState(task=task)
    s.initialize()
    return s


def test_assign_feasible_trivial():
    s = make_env("trivial_task.json")
    ok, reason, marg = check_feasibility(s, "r-0", "v-0", 0, 1)
    assert ok, reason
    assert marg > 0


def test_assign_infeasible_capacity():
    s = make_env("oversize_request.json")
    ok, reason, _ = check_feasibility(s, "r-big", "v-small", 0, 1)
    assert not ok
    assert "capacity" in reason


def test_assign_infeasible_window():
    s = make_env("tight_window_task.json")
    ok, reason, _ = check_feasibility(s, "r-late", "v-0", 0, 1)
    assert not ok
    assert "window" in reason or "latest" in reason


def test_pickup_before_dropoff_required():
    s = make_env("trivial_task.json")
    ok, reason, _ = check_feasibility(s, "r-0", "v-0", 1, 1)
    assert not ok
    assert "positions" in reason


def test_unknown_vehicle():
    s = make_env("trivial_task.json")
    ok, reason, _ = check_feasibility(s, "r-0", "v-999", 0, 1)
    assert not ok
    assert "unknown vehicle" in reason


def test_not_pending_request():
    s = make_env("trivial_task.json")
    s.request_status["r-0"] = "assigned"
    ok, reason, _ = check_feasibility(s, "r-0", "v-0", 0, 1)
    assert not ok
    assert "not pending" in reason
