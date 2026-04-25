"""tests for build_distance_matrix.

osrm calls are monkey-patched; a temp cache path is injected per test so the
real data/osrm_cache.json is never touched and tests are fully deterministic.
"""
import time
from unittest.mock import MagicMock

import pytest

from scripts.build_distance_matrix import build_matrix, haversine_km

CHARING    = (-0.1278, 51.5074)
TOWER      = (-0.0759, 51.5081)
PADDINGTON = (-0.1755, 51.5154)

PAIR   = [CHARING, TOWER]
TRIPLE = [CHARING, TOWER, PADDINGTON]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """redirect cache writes to a throwaway temp file for every test."""
    monkeypatch.setattr("scripts.build_distance_matrix.CACHE_PATH", tmp_path / "osrm_cache.json")


def _osrm_stub(coords):
    """synthetic osrm /table response: 2 km / 5 min between every pair."""
    n = len(coords)
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "code": "Ok",
        "durations": [[0.0 if i == j else 300.0 for j in range(n)] for i in range(n)],
        "distances": [[0.0 if i == j else 2000.0 for j in range(n)] for i in range(n)],
    }
    return mock


# ---------------------------------------------------------------------------
# pure haversine computation — no network
# ---------------------------------------------------------------------------

def test_haversine_london_paris():
    d = haversine_km((-0.1276, 51.5074), (2.3522, 48.8566))
    assert 320 < d < 360, f"expected ~344 km, got {d:.1f}"


def test_haversine_same_point_zero():
    assert haversine_km(CHARING, CHARING) == 0.0


# ---------------------------------------------------------------------------
# build_matrix — osrm path (stubbed)
# ---------------------------------------------------------------------------

def test_build_matrix_uses_osrm_values(monkeypatch):
    monkeypatch.setattr(
        "scripts.build_distance_matrix.requests.get",
        lambda url, **kw: _osrm_stub(PAIR),
    )
    dist, dur = build_matrix(PAIR)
    assert dist[0][1] == pytest.approx(2.0)  # 2000 m → 2.0 km
    assert dur[0][1]  == pytest.approx(5.0)  # 300 s → 5.0 min


def test_build_matrix_shape_pair(monkeypatch):
    monkeypatch.setattr(
        "scripts.build_distance_matrix.requests.get",
        lambda url, **kw: _osrm_stub(PAIR),
    )
    dist, dur = build_matrix(PAIR)
    assert len(dist) == 2 and all(len(r) == 2 for r in dist)
    assert len(dur)  == 2 and all(len(r) == 2 for r in dur)


def test_build_matrix_shape_triple(monkeypatch):
    monkeypatch.setattr(
        "scripts.build_distance_matrix.requests.get",
        lambda url, **kw: _osrm_stub(TRIPLE),
    )
    dist, dur = build_matrix(TRIPLE)
    assert len(dist) == 3 and all(len(r) == 3 for r in dist)
    assert len(dur)  == 3 and all(len(r) == 3 for r in dur)


def test_build_matrix_self_distance_zero(monkeypatch):
    monkeypatch.setattr(
        "scripts.build_distance_matrix.requests.get",
        lambda url, **kw: _osrm_stub(PAIR),
    )
    dist, dur = build_matrix(PAIR)
    assert dist[0][0] == 0.0 and dur[0][0] == 0.0


# ---------------------------------------------------------------------------
# build_matrix — fallback path (osrm stubbed to fail)
# ---------------------------------------------------------------------------

def test_build_matrix_falls_back_on_osrm_failure(monkeypatch):
    def raise_error(url, **kw):
        raise RuntimeError("osrm down")

    monkeypatch.setattr("scripts.build_distance_matrix.requests.get", raise_error)
    dist, dur = build_matrix(PAIR, allow_fallback=True)
    assert dist[0][0] == 0.0
    assert dist[0][1] > 0.0
    assert dur[0][1]  > 0.0


def test_build_matrix_raises_when_fallback_disabled(monkeypatch):
    def raise_error(url, **kw):
        raise RuntimeError("osrm down")

    monkeypatch.setattr("scripts.build_distance_matrix.requests.get", raise_error)
    with pytest.raises(RuntimeError):
        build_matrix(PAIR, allow_fallback=False)


# ---------------------------------------------------------------------------
# cache behaviour
# ---------------------------------------------------------------------------

def test_cache_hit_skips_network(monkeypatch):
    call_count = {"n": 0}

    def counting_get(url, **kw):
        call_count["n"] += 1
        return _osrm_stub(PAIR)

    monkeypatch.setattr("scripts.build_distance_matrix.requests.get", counting_get)
    build_matrix(PAIR)          # miss → network call
    build_matrix(PAIR)          # hit → no network call
    assert call_count["n"] == 1


def test_cache_hit_is_fast(monkeypatch):
    monkeypatch.setattr(
        "scripts.build_distance_matrix.requests.get",
        lambda url, **kw: _osrm_stub(PAIR),
    )
    build_matrix(PAIR)          # warm cache
    t0 = time.time()
    build_matrix(PAIR)
    assert time.time() - t0 < 0.1
