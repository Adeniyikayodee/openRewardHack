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
