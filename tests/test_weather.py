"""tests for fetch_weather.

fetch_weather() network calls are monkey-patched. synthetic_weather(),
weather_at(), and weather_speed_factor() are pure functions — no patching needed.
"""
from unittest.mock import MagicMock

import pytest

from scripts.fetch_weather import (
    fetch_weather,
    synthetic_weather,
    weather_at,
    weather_speed_factor,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _open_meteo_stub(date_iso: str) -> MagicMock:
    """synthetic open-meteo response for a single day (24 hours)."""
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "hourly": {
            "time":             [f"{date_iso}T{h:02d}:00" for h in range(24)],
            "precipitation":    [0.0] * 24,
            "wind_speed_10m":   [10.0] * 24,
            "visibility":       [10_000.0] * 24,
            "temperature_2m":   [15.0] * 24,
        }
    }
    return mock


# ---------------------------------------------------------------------------
# fetch_weather — network stubbed
# ---------------------------------------------------------------------------

def test_fetch_weather_returns_24_entries(monkeypatch):
    monkeypatch.setattr(
        "scripts.fetch_weather.requests.get",
        lambda *a, **kw: _open_meteo_stub("2026-04-25"),
    )
    tl = fetch_weather("2026-04-25")
    assert len(tl) == 24


def test_fetch_weather_required_keys(monkeypatch):
    monkeypatch.setattr(
        "scripts.fetch_weather.requests.get",
        lambda *a, **kw: _open_meteo_stub("2026-04-25"),
    )
    tl = fetch_weather("2026-04-25")
    for entry in tl:
        for key in ("t", "precip_mm", "wind_kph", "visibility_km", "temp_c"):
            assert key in entry, f"missing key {key}"


def test_fetch_weather_t_values(monkeypatch):
    monkeypatch.setattr(
        "scripts.fetch_weather.requests.get",
        lambda *a, **kw: _open_meteo_stub("2026-04-25"),
    )
    tl = fetch_weather("2026-04-25")
    assert [e["t"] for e in tl] == list(range(0, 24 * 60, 60))


def test_fetch_weather_raises_on_network_error(monkeypatch):
    def raise_error(*a, **kw):
        raise RuntimeError("api down")

    monkeypatch.setattr("scripts.fetch_weather.requests.get", raise_error)
    with pytest.raises(RuntimeError):
        fetch_weather("2026-04-25")


# ---------------------------------------------------------------------------
# synthetic_weather — pure function, no network
# ---------------------------------------------------------------------------

def test_synthetic_weather_length():
    assert len(synthetic_weather(0.5)) == 24


def test_synthetic_weather_required_keys():
    for entry in synthetic_weather(0.0):
        for key in ("t", "precip_mm", "wind_kph", "visibility_km", "temp_c"):
            assert key in entry


def test_synthetic_weather_t_values():
    assert [e["t"] for e in synthetic_weather(0.0)] == list(range(0, 24 * 60, 60))


def test_synthetic_weather_deterministic():
    assert synthetic_weather(0.7) == synthetic_weather(0.7)


def test_synthetic_severity_scales_precipitation():
    mild   = synthetic_weather(0.0)
    severe = synthetic_weather(1.0)
    assert sum(e["precip_mm"] for e in mild) <= sum(e["precip_mm"] for e in severe)


# ---------------------------------------------------------------------------
# weather_speed_factor — pure function
# ---------------------------------------------------------------------------

def test_weather_speed_factor_clear():
    assert weather_speed_factor({"precip_mm": 0.0, "visibility_km": 10.0}) == 1.0


def test_weather_speed_factor_heavy_rain():
    assert weather_speed_factor({"precip_mm": 8.0, "visibility_km": 5.0}) == pytest.approx(0.80)


def test_weather_speed_factor_low_visibility():
    assert weather_speed_factor({"precip_mm": 0.0, "visibility_km": 0.5}) == pytest.approx(0.70)


def test_weather_speed_factor_stacks():
    assert weather_speed_factor({"precip_mm": 8.0, "visibility_km": 0.5}) == pytest.approx(0.56)


# ---------------------------------------------------------------------------
# weather_at — pure function
# ---------------------------------------------------------------------------

def test_weather_at_episode_start():
    tl = synthetic_weather(0.5)
    # episode t=0 → real 06:00 → hour index 6
    assert weather_at(tl, 0) == tl[6]


def test_weather_at_clamps_to_last():
    tl = synthetic_weather(0.5)
    assert weather_at(tl, 9999) == tl[-1]
