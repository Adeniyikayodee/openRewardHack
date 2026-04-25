"""tests for generate_tasks.py — runs against data/tasks.jsonl produced by --quick."""
import json
from pathlib import Path

import pytest

JSONL = Path("data/tasks.jsonl")
PARQUET = Path("data/tasks.parquet")

REQUIRED_KEYS = {
    "id", "task_type", "difficulty", "split", "episode_date",
    "horizon_minutes", "tick_minutes", "depots", "vehicles", "requests",
    "nodes", "distance_matrix_km", "duration_matrix_min",
    "weather_timeline", "traffic_events", "dynamic_events",
    "osrm_fallback", "or_tools_baseline_cost",
    "or_tools_baseline_unserved", "or_tools_baseline_served",
}


def _tasks():
    if not JSONL.exists():
        pytest.skip("data/tasks.jsonl not found — run: python scripts/generate_tasks.py --quick")
    return [json.loads(l) for l in JSONL.open()]


def test_jsonl_exists():
    assert JSONL.exists(), "run: PYTHONPATH=. python scripts/generate_tasks.py --quick"


def test_parquet_exists():
    assert PARQUET.exists(), "parquet file not generated"


def test_task_count_positive():
    tasks = _tasks()
    assert len(tasks) > 0


def test_required_keys_present():
    for task in _tasks():
        missing = REQUIRED_KEYS - set(task.keys())
        assert not missing, f"task {task.get('id')} missing keys: {missing}"


def test_task_ids_unique():
    tasks = _tasks()
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"


def test_task_type_valid():
    for task in _tasks():
        assert task["task_type"] in ("cvrptw", "pdptw"), \
            f"invalid task_type: {task['task_type']}"


def test_difficulty_positive():
    for task in _tasks():
        assert task["difficulty"] >= 1, f"difficulty must be >= 1, got {task['difficulty']}"


def test_split_valid():
    for task in _tasks():
        assert task["split"] in ("tutorial", "train", "test"), \
            f"invalid split: {task['split']}"


def test_distance_matrix_square():
    for task in _tasks():
        n = len(task["nodes"])
        dm = task["distance_matrix_km"]
        assert len(dm) == n, f"dist matrix rows != n_nodes"
        assert all(len(r) == n for r in dm), "dist matrix not square"


def test_duration_matrix_square():
    for task in _tasks():
        n = len(task["nodes"])
        dur = task["duration_matrix_min"]
        assert len(dur) == n
        assert all(len(r) == n for r in dur)


def test_diagonal_zero():
    for task in _tasks():
        dm = task["distance_matrix_km"]
        for i in range(len(dm)):
            assert dm[i][i] == 0.0, f"non-zero diagonal at ({i},{i})"


def test_weather_timeline_24h():
    for task in _tasks():
        assert len(task["weather_timeline"]) == 24


def test_requests_not_empty():
    for task in _tasks():
        assert len(task["requests"]) > 0


def test_vehicles_not_empty():
    for task in _tasks():
        assert len(task["vehicles"]) > 0


def test_cvrptw_requests_have_dropoff_only():
    for task in _tasks():
        if task["task_type"] != "cvrptw":
            continue
        for r in task["requests"]:
            assert "dropoff_node_idx" in r, f"cvrptw request missing dropoff_node_idx: {r['id']}"
            # cvrptw requests should NOT have a pickup_node_idx
            assert "pickup_node_idx" not in r, f"cvrptw request has pickup_node_idx: {r['id']}"


def test_pdptw_requests_have_both():
    for task in _tasks():
        if task["task_type"] != "pdptw":
            continue
        for r in task["requests"]:
            assert "pickup_node_idx" in r, f"pdptw request missing pickup_node_idx: {r['id']}"
            assert "dropoff_node_idx" in r, f"pdptw request missing dropoff_node_idx: {r['id']}"


def test_baseline_counts_consistent():
    for task in _tasks():
        n = len(task["requests"])
        served = task["or_tools_baseline_served"]
        unserved = task["or_tools_baseline_unserved"]
        assert served + unserved == n, \
            f"served ({served}) + unserved ({unserved}) != n_requests ({n})"


def test_node_indices_in_range():
    for task in _tasks():
        n = len(task["nodes"])
        for r in task["requests"]:
            for key in ("pickup_node_idx", "dropoff_node_idx"):
                if key in r:
                    assert 0 <= r[key] < n, f"{key}={r[key]} out of range [0,{n})"


def test_osrm_fallback_is_bool():
    for task in _tasks():
        assert isinstance(task["osrm_fallback"], bool)


def test_parquet_row_count_matches_jsonl():
    import pandas as pd
    tasks = _tasks()
    df = pd.read_parquet(PARQUET)
    assert len(df) == len(tasks), \
        f"parquet has {len(df)} rows but jsonl has {len(tasks)} tasks"
