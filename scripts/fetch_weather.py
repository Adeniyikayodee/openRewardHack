import requests


def fetch_weather(date_iso, lat=51.5074, lon=-0.1278):
    """Returns 24-hour weather timeline for the given date, indexed in
    minutes from 00:00 (so t=0 corresponds to midnight)."""
    url = "https://api.open-meteo.com/v1/forecast"
    r = requests.get(url, params={
        "latitude": lat, "longitude": lon,
        "hourly": "precipitation,wind_speed_10m,visibility,temperature_2m",
        "start_date": date_iso, "end_date": date_iso,
        "timezone": "Europe/London"
    }, timeout=20)
    r.raise_for_status()
    h = r.json()["hourly"]
    timeline = []
    for i, _ in enumerate(h["time"]):
        timeline.append({
            "t": i * 60,
            "precip_mm":     h["precipitation"][i] or 0.0,
            "wind_kph":      h["wind_speed_10m"][i] or 0.0,
            "visibility_km": (h["visibility"][i] or 10000) / 1000.0,
            "temp_c":        h["temperature_2m"][i] or 12.0,
        })
    return timeline


def weather_at(timeline, t_minutes):
    """Look up weather conditions at episode-time t_minutes (offset from 06:00)."""
    # Episode starts at 06:00 → real-time = 360 + t_minutes
    real_minute = 360 + t_minutes
    hour_idx = min(len(timeline) - 1, real_minute // 60)
    return timeline[hour_idx]


def weather_speed_factor(weather):
    """Combine precipitation + visibility into a speed multiplier."""
    factor = 1.0
    if weather["precip_mm"] > 5.0:    factor *= 0.80
    elif weather["precip_mm"] > 2.0:  factor *= 0.90
    if weather["visibility_km"] < 1.0:    factor *= 0.70
    elif weather["visibility_km"] < 2.0:  factor *= 0.85
    return factor


def synthetic_weather(severity: float):
    """Used when we want to scale weather by difficulty rather than real day.
    severity in [0,1]. Returns 24h timeline."""
    import random
    rng = random.Random(int(severity * 1e6))
    timeline = []
    for i in range(24):
        precip = max(0, rng.gauss(severity * 3, severity * 2))
        timeline.append({
            "t": i * 60,
            "precip_mm": round(precip, 2),
            "wind_kph":  round(8 + severity * rng.uniform(0, 30), 1),
            "visibility_km": max(0.5, 10 - severity * rng.uniform(0, 8)),
            "temp_c": round(8 + rng.uniform(0, 10), 1),
        })
    return timeline
