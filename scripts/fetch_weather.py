"""open-meteo weather fetcher and synthetic weather generator.
two modes:
  fetch_weather(date_iso)      — real hourly data from open-meteo (offline/generation only)
  synthetic_weather(severity)  — deterministic seeded timeline (used at episode runtime)
weather_at() and weather_speed_factor() are shared with src/feasibility.py.
do not change their signatures.
"""
import random

import requests


def fetch_weather(
    date_iso: str,
    lat: float = 51.5074,
    lon: float = -0.1278,
) -> list[dict]:
    """fetch real 24-hour hourly weather for date_iso from open-meteo (no api key).

    returns a list of 24 dicts, each with keys:
        t (minutes from midnight), precip_mm, wind_kph, visibility_km, temp_c
    """
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation,wind_speed_10m,visibility,temperature_2m",
            "start_date": date_iso,
            "end_date": date_iso,
            "timezone": "Europe/London",
        },
        timeout=20,
    )
    r.raise_for_status()
    h = r.json()["hourly"]
    timeline = []
    for i in range(len(h["time"])):
        timeline.append({
            "t": i * 60,
            "precip_mm":     round(h["precipitation"][i] or 0.0, 2),
            "wind_kph":      round(h["wind_speed_10m"][i] or 0.0, 1),
            "visibility_km": round((h["visibility"][i] or 10_000) / 1000.0, 2),
            "temp_c":        round(h["temperature_2m"][i] or 12.0, 1),
        })
    return timeline


def synthetic_weather(severity: float) -> list[dict]:
    """generate a deterministic 24-hour weather timeline from severity ∈ [0, 1].

    higher severity → more rain, lower visibility, stronger wind.
    seeded so same severity always produces the same timeline.
    """
    rng = random.Random(int(severity * 1_000_000))
    timeline = []
    for i in range(24):
        precip = max(0.0, rng.gauss(severity * 3, max(0.01, severity * 2)))
        timeline.append({
            "t": i * 60,
            "precip_mm":     round(precip, 2),
            "wind_kph":      round(8 + severity * rng.uniform(0, 30), 1),
            "visibility_km": round(max(0.5, 10 - severity * rng.uniform(0, 8)), 2),
            "temp_c":        round(8 + rng.uniform(0, 10), 1),
        })
    return timeline


def weather_at(timeline: list[dict], t_minutes: int) -> dict:
    """look up weather at episode-time t_minutes (offset from 06:00 episode start).

    episode starts at 06:00 real time → real_minute = 360 + t_minutes.
    """
    real_minute = 360 + t_minutes
    hour_idx = min(len(timeline) - 1, real_minute // 60)
    return timeline[hour_idx]


def weather_speed_factor(weather: dict) -> float:
    """combine precipitation and visibility into a driving speed multiplier.

    returns a value in (0, 1] where 1.0 = clear conditions.
    """
    factor = 1.0
    if weather["precip_mm"] > 5.0:
        factor *= 0.80
    elif weather["precip_mm"] > 2.0:
        factor *= 0.90
    if weather["visibility_km"] < 1.0:
        factor *= 0.70
    elif weather["visibility_km"] < 2.0:
        factor *= 0.85
    return factor
