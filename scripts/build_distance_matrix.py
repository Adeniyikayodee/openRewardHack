"""osrm-backed distance matrix builder with disk cache.

primary: osrm /table/v1/driving — real driving distances and durations.
fallback: haversine * 1.4 at 30 km/h average speed — only used when osrm
          is completely unavailable after 3 retries. tasks built with the
          fallback carry "osrm_fallback": true in their spec.

usage:
    from scripts.build_distance_matrix import build_matrix
    dist_km, dur_min = build_matrix([(lon1, lat1), (lon2, lat2), ...])
"""
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path

import requests

_CACHE_LOCK = threading.Lock()

CACHE_PATH = Path("data/osrm_cache.json")
OSRM_BASE = os.environ.get("OSRM_BASE_URL", "https://router.project-osrm.org")
# max osrm table request: 100x100 = 10,000 cells. our largest task is ~78 nodes.
MAX_NODES = 100


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """straight-line distance in km between two (lon, lat) points."""
    lon1, lat1 = a
    lon2, lat2 = b
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    h = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(h))


def _haversine_fallback(coords: list[tuple[float, float]]) -> tuple[list, list]:
    """fallback when osrm is unavailable: haversine * 1.4 urban fudge, 30 km/h average."""
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    dur = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = haversine_km(coords[i], coords[j]) * 1.4
            dist[i][j] = round(d, 4)
            dur[i][j] = round(d / 30.0 * 60, 4)  # 30 km/h → minutes
    return dist, dur


def _cache_key(coords: list[tuple[float, float]]) -> str:
    """stable sha256 key from coord list (order matters — matrix is N×N)."""
    rounded = [(round(lon, 6), round(lat, 6)) for lon, lat in coords]
    return hashlib.sha256(json.dumps(rounded).encode()).hexdigest()


def _load_cache_unlocked() -> dict:
    """must be called under _CACHE_LOCK."""
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_cache_unlocked(cache: dict) -> None:
    """must be called under _CACHE_LOCK."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    tmp.replace(CACHE_PATH)


def _fetch_from_osrm(coords: list[tuple[float, float]]) -> tuple[list, list]:
    """call osrm table service. raises on failure."""
    if len(coords) > MAX_NODES:
        raise ValueError(f"too many nodes ({len(coords)}), osrm hard limit is {MAX_NODES}")

    coord_str = ";".join(f"{lon},{lat}" for lon, lat in coords)
    url = f"{OSRM_BASE}/table/v1/driving/{coord_str}?annotations=duration,distance"

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=45)
            r.raise_for_status()
            data = r.json()
            if data.get("code") != "Ok":
                raise RuntimeError(f"osrm returned code={data.get('code')}")
            # durations in seconds → minutes; distances in metres → km
            dur_min = [[(d or 0.0) / 60.0 for d in row] for row in data["durations"]]
            dist_km = [[(d or 0.0) / 1000.0 for d in row] for row in data["distances"]]
            return dist_km, dur_min
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)

    raise RuntimeError(f"osrm unavailable after 3 attempts: {last_err}") from last_err


def build_matrix(
    coords: list[tuple[float, float]],
    allow_fallback: bool = True,
) -> tuple[list, list]:
    """return (dist_km N×N, dur_min N×N) for the given (lon, lat) coordinate list.

    results are cached to disk by coord-list hash. subsequent calls with the
    same coords are instant. if osrm fails and allow_fallback=True, the
    haversine fallback is used and cached with a "fallback": true marker.
    thread-safe: cache reads/writes are protected by _CACHE_LOCK; slow network
    calls happen outside the lock.
    """
    key = _cache_key(coords)

    with _CACHE_LOCK:
        cache = _load_cache_unlocked()
        if key in cache:
            return cache[key]["dist_km"], cache[key]["dur_min"]

    # slow fetch outside lock
    try:
        dist_km, dur_min = _fetch_from_osrm(coords)
        entry: dict = {"dist_km": dist_km, "dur_min": dur_min}
    except Exception as e:
        if not allow_fallback:
            raise
        print(f"warning: osrm unavailable ({e}), using haversine fallback")
        dist_km, dur_min = _haversine_fallback(coords)
        entry = {"dist_km": dist_km, "dur_min": dur_min, "fallback": True}

    with _CACHE_LOCK:
        cache = _load_cache_unlocked()   # re-read in case another thread wrote
        if key not in cache:             # only write if not already present
            cache[key] = entry
            _save_cache_unlocked(cache)

    return dist_km, dur_min


def is_fallback(coords: list[tuple[float, float]]) -> bool:
    """return true if the cached entry for these coords used the haversine fallback."""
    with _CACHE_LOCK:
        cache = _load_cache_unlocked()
    entry = cache.get(_cache_key(coords), {})
    return entry.get("fallback", False)
