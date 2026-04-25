"""tests for solver.py — or-tools must be installed."""
import pytest

from scripts.solver import solve_baseline

# ---------------------------------------------------------------------------
# shared fixture builders
# ---------------------------------------------------------------------------

def _base_task(task_type: str) -> dict:
    return {
        "task_type": task_type,
        "horizon_minutes": 480,
        "nodes": [{"idx": 0}, {"idx": 1}, {"idx": 2}],
        "depots": [{"node_idx": 0}],
        "vehicles": [{"id": "v0", "capacity_seats": 8}],
        "distance_matrix_km":  [[0.0, 2.0, 3.0], [2.0, 0.0, 1.0], [3.0, 1.0, 0.0]],
        "duration_matrix_min": [[0.0, 5.0, 8.0], [5.0, 0.0, 3.0], [8.0, 3.0, 0.0]],
        "traffic_events": [],
    }


def _cvrptw_task() -> dict:
    task = _base_task("cvrptw")
    task["requests"] = [
        {
            "id": "r0",
            "dropoff_node_idx": 2,
            "passengers": 1,
            "earliest_dropoff": 0,
            "latest_dropoff": 300,
            "released_at": 0,
        }
    ]
    return task


def _pdptw_task() -> dict:
    task = _base_task("pdptw")
    task["requests"] = [
        {
            "id": "r0",
            "pickup_node_idx": 1,
            "dropoff_node_idx": 2,
            "passengers": 1,
            "earliest_pickup": 0,
            "latest_pickup": 200,
            "earliest_dropoff": 0,
            "latest_dropoff": 300,
            "released_at": 0,
        }
    ]
    return task


# ---------------------------------------------------------------------------
# cvrptw tests
# ---------------------------------------------------------------------------

def test_cvrptw_trivial_solves():
    out = solve_baseline(_cvrptw_task(), time_limit_s=5)
    assert out["n_unserved"] == 0
    assert out["n_served"] == 1
    assert 0 < out["total_cost_km"] < 20


def test_cvrptw_returns_required_keys():
    out = solve_baseline(_cvrptw_task(), time_limit_s=5)
    for key in ("total_cost_km", "n_unserved", "n_served", "routes"):
        assert key in out


def test_cvrptw_infeasible_capacity():
    task = _cvrptw_task()
    task["requests"][0]["passengers"] = 999
    out = solve_baseline(task, time_limit_s=5)
    assert out["n_unserved"] == 1
    assert out["n_served"] == 0


def test_cvrptw_tight_time_window_infeasible():
    task = _cvrptw_task()
    # window that has already closed before the vehicle can reach node 2 (8 min)
    task["requests"][0]["earliest_dropoff"] = 0
    task["requests"][0]["latest_dropoff"] = 2  # too tight
    out = solve_baseline(task, time_limit_s=5)
    assert out["n_unserved"] == 1


# ---------------------------------------------------------------------------
# pdptw tests
# ---------------------------------------------------------------------------

def test_pdptw_trivial_solves():
    out = solve_baseline(_pdptw_task(), time_limit_s=5)
    assert out["n_unserved"] == 0
    assert out["n_served"] == 1
    assert 0 < out["total_cost_km"] < 20


def test_pdptw_returns_required_keys():
    out = solve_baseline(_pdptw_task(), time_limit_s=5)
    for key in ("total_cost_km", "n_unserved", "n_served", "routes"):
        assert key in out


def test_pdptw_infeasible_capacity():
    task = _pdptw_task()
    task["requests"][0]["passengers"] = 999
    out = solve_baseline(task, time_limit_s=5)
    assert out["n_unserved"] == 1
    assert out["n_served"] == 0


def test_pdptw_multiple_requests():
    # 4-node task: depot(0), A(1), B(2), C(3)
    # r0: pickup A(1) → drop B(2); r1: pickup A(1) → drop C(3)
    task = {
        "task_type": "pdptw",
        "horizon_minutes": 480,
        "nodes": [{"idx": i} for i in range(4)],
        "depots": [{"node_idx": 0}],
        "vehicles": [{"id": "v0", "capacity_seats": 10}],
        "distance_matrix_km": [
            [0.0, 2.0, 3.0, 4.0],
            [2.0, 0.0, 1.0, 2.0],
            [3.0, 1.0, 0.0, 1.5],
            [4.0, 2.0, 1.5, 0.0],
        ],
        "duration_matrix_min": [
            [0.0, 5.0, 8.0, 10.0],
            [5.0, 0.0, 3.0, 6.0],
            [8.0, 3.0, 0.0, 4.0],
            [10.0, 6.0, 4.0, 0.0],
        ],
        "traffic_events": [],
        "requests": [
            {
                "id": "r0", "pickup_node_idx": 1, "dropoff_node_idx": 2,
                "passengers": 2, "earliest_pickup": 0, "latest_pickup": 200,
                "earliest_dropoff": 0, "latest_dropoff": 300, "released_at": 0,
            },
            {
                "id": "r1", "pickup_node_idx": 1, "dropoff_node_idx": 3,
                "passengers": 2, "earliest_pickup": 0, "latest_pickup": 200,
                "earliest_dropoff": 0, "latest_dropoff": 300, "released_at": 0,
            },
        ],
    }
    out = solve_baseline(task, time_limit_s=5)
    assert out["n_served"] + out["n_unserved"] == 2
    assert out["total_cost_km"] >= 0


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def test_dispatch_on_task_type():
    """cvrptw and pdptw tasks with equivalent structure should both solve."""
    c_out = solve_baseline(_cvrptw_task(), time_limit_s=5)
    p_out = solve_baseline(_pdptw_task(), time_limit_s=5)
    assert c_out["n_served"] == 1
    assert p_out["n_served"] == 1
