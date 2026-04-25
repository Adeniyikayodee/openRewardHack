"""Shared fixtures for all tests. Kayode owns the integration/rollout
fixtures; this file provides the lightweight unit-test helpers."""
import json
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


@pytest.fixture
def trivial_task():
    return load_fixture("trivial_task.json")


@pytest.fixture
def oversize_task():
    return load_fixture("oversize_request.json")


@pytest.fixture
def tight_window_task():
    return load_fixture("tight_window_task.json")


@pytest.fixture
def traffic_task():
    return load_fixture("traffic_at_60.json")


@pytest.fixture
def late_request_task():
    return load_fixture("late_request_task.json")


# ---------------------------------------------------------------------------
# server-backed fixtures (used by test_integration / test_rollout)
# ---------------------------------------------------------------------------
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_tasks(fixtures_dir):
    tasks = {}
    for f in sorted(fixtures_dir.glob("*.json")):
        spec = json.loads(f.read_text())
        tasks[spec["id"]] = spec
    return tasks


@pytest.fixture(scope="session")
def tasks_parquet_dir(tmp_path_factory, fixture_tasks):
    """Build a tasks.parquet file from the fixture set in a tmp dir."""
    import pandas as pd
    rows = [{**spec, "split": "tutorial"} for spec in fixture_tasks.values()]
    data_dir = tmp_path_factory.mktemp("orwd_data")
    pd.DataFrame(rows).to_parquet(data_dir / "tasks.parquet")
    return data_dir


@pytest.fixture(scope="module")
def server(tasks_parquet_dir):
    """Spin up `python -m src.server` with ORWD_DATA_DIR pointing at the
    fixture-backed parquet. Module-scoped so test_integration and
    test_rollout share one server."""
    env = {**os.environ, "ORWD_DATA_DIR": str(tasks_parquet_dir)}
    proc = subprocess.Popen(
        ["python", "-m", "src.server"], env=env, cwd=str(ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(4)
    if proc.poll() is not None:
        err = (proc.stderr.read(2000).decode(errors="replace")
               if proc.stderr else "")
        pytest.fail(f"server failed to start: {err}")
    yield proc
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def required_task_keys():
    return {
        "id", "difficulty", "split", "episode_date", "horizon_minutes",
        "tick_minutes", "depots", "vehicles", "requests", "nodes",
        "distance_matrix_km", "duration_matrix_min", "weather_timeline",
        "traffic_events", "dynamic_events",
        "or_tools_baseline_cost", "or_tools_baseline_unserved",
        "or_tools_baseline_served",
    }
