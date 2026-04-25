# `london-dynamic-routing-env` — Development Specification

> An OpenReward environment for **dynamic, multi-horizon, weather- and traffic-aware vehicle routing on the real London road network.** Built for the OpenReward × EnvCommons hackathon. Three developers — **Kosi, Daniel, Kayode** — ~3 hours, 100 tasks of monotonically increasing difficulty.

---

## 0. TL;DR for the implementing model

You are building a single OpenReward (ORS) environment named `LondonDynamicRouting`. The agent is a fleet dispatcher for a heterogeneous fleet (minibuses, vans, accessible vehicles) serving heterogeneous passenger/parcel demand across **Zones 1–4 of London**. The environment exposes **multiple tools** for dynamic dispatch and re-optimization, **dense rewards**, **monotonically increasing task complexity (labeled `difficulty: 1..100`)**, and **deterministic replay of synthesized traffic + real weather** baked into each task at generation time.

Key non-obvious design decisions (read carefully — these resolve trade-offs that will trip up implementers):

1. **All routing is OSRM.** Distance matrices are pre-computed offline via OSRM's table service and frozen into the task spec. The agent never calls OSRM at runtime; it queries a `get_distance(a, b)` tool that reads the cached matrix. This is essential for determinism, reproducibility, and rate-limit safety.
2. **Real weather is "real" but baked in.** Weather is fetched at task-generation time from Open-Meteo (no key needed) and stored as a time-indexed timeline. At episode runtime, calling `tick(minutes)` advances simulated time and replays weather effects deterministically. No external calls during episodes.
3. **Traffic is synthesized.** We generate plausible congestion events (rush hour on key corridors, random incidents) parameterized by difficulty. This lets us cleanly scale traffic severity per task without depending on what was happening in London on any given day.
4. **The episode is multi-horizon.** A single task spans one operational day (06:00–22:00, 16 hours). The agent dispatches in 15-minute decision intervals. New requests, vehicle breakdowns, and capacity changes arrive *during* the episode, forcing dynamic re-optimization.
5. **The action space is rich.** Not just `insert_customer`. The agent can `assign`, `reassign`, `defer`, `cancel`, `add_vehicle`, `swap_vehicles`, `query_traffic`, `query_weather`, `tick`, and `submit_plan`. Most actions return reward signal; only `submit_plan` and timeout end the episode.
6. **Hetero-everything.** Vehicles differ in capacity, speed, fuel cost, wheelchair accessibility, and shift schedule. Demands differ in passenger count, accessibility need, time window, priority, and pickup→dropoff structure (this is **PDPTW**, not classic CVRPTW).
7. **Reward shape is dense + step-penalty + terminal.** Per-action shaping rewards drive learning; a per-step trajectory length penalty incentivizes speed; the terminal reward grades against a precomputed OR-Tools optimum.
8. **Difficulty curve is calibrated so a strong agent solves at least the tutorial split.** Difficulty 1–10 is solvable by any reasonable agent (small instances, no dynamics). Difficulty 90+ is hard for any agent including OR-Tools given a fixed budget. Reward is positive for any episode with ≥ 50% coverage at d ≤ 30, providing a clear positive signal during demos.

---

## 1. Resources & references

### OpenReward
- Platform: https://openreward.ai/
- Docs root: https://docs.openreward.ai/
- First env tutorial: https://docs.openreward.ai/environments/your-first-environment
- Agentic env tutorial (sandboxes, async client, secrets): https://docs.openreward.ai/environments/building-agentic-environments
- ORS spec: https://openrewardstandard.io
- OpenReward Python lib: https://github.com/OpenReward/openreward-python
- Recording rollouts: https://docs.openreward.ai/rollouts/recording-rollouts
- CLI reference (`orwd init <name> --template basic`): https://docs.openreward.ai/environments/using-the-cli

### Real-world data APIs (offline only — only at task-generation time)

- **OSRM** — driving distance matrix and routing.
  - Public demo (free, best-effort): `https://router.project-osrm.org`
  - Self-hosted (recommended for hackathon if anyone has one running): `http://YOUR_HOST:5000`
  - Table service: `GET /table/v1/driving/{lon,lat;...}?annotations=duration,distance` — returns N×N matrix; hard limit 100×100 = 10,000 cells per request.
  - All three devs have an `OSRM_BASE_URL` env var configured. Default to public demo if unset.
- **Open-Meteo** (weather, no key): `https://api.open-meteo.com/v1/forecast?latitude=51.5&longitude=-0.12&hourly=precipitation,wind_speed_10m,visibility,temperature_2m`
- **Nominatim / OSM** for geocoding (1 req/sec, attribute usage): `https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1`. Used **only once** to seed the canonical POI list; results checked in to repo.

### Runtime API access (for executing rollouts)

All three devs have:
- `OSRM_BASE_URL` — for dataset generation (Kosi's stream)
- `OPENAI_API_KEY` — for executing rollouts against the agent (the rollout script uses this)
- `OPENREWARD_API_KEY` — for hitting the deployed OpenReward env

### Inspiration / sibling environments
- EnvCommons: https://github.com/EnvCommons (ATC, MobileEnv, GraphWalks, SETA, PowerGrid)
- PyVroom (VRP solver, OR-Tools wrapper for VRP): https://github.com/VROOM-Project/pyvroom
- Google OR-Tools routing guide: https://developers.google.com/optimization/routing

### Hugging Face (for the task dataset)
- Dataset hub: https://huggingface.co/datasets
- `huggingface_hub` upload guide: https://huggingface.co/docs/huggingface_hub/guides/upload

---

## 2. System architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TASK GENERATION (offline)                    │
│  scripts/generate_tasks.py        ← Kosi                        │
│    1. Load 250+ real London POIs (Zones 1–4)                    │
│    2. Build distance matrix via OSRM (cached to disk)           │
│    3. Pull weather forecast for episode date (Open-Meteo)       │
│    4. Synthesize traffic events (parameterized by difficulty)   │
│    5. Synthesize dynamic events (breakdowns, new requests)      │
│    6. Compute OR-Tools baseline → store optimum                 │
│    7. Write tasks.parquet (publish to HF Hub)                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                ORS SERVER (server.py)            ← Daniel       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  LondonDynamicRouting(Environment)                        │  │
│  │  ├─ list_splits / list_tasks / get_prompt                 │  │
│  │  ├─ State: routes, vehicles, time, pending events         │  │
│  │  └─ Tools (14):                                           │  │
│  │     • assign / reassign / defer / cancel                  │  │
│  │     • add_vehicle / swap_vehicles                         │  │
│  │     • query_traffic / query_weather                       │  │
│  │     • get_distance / get_eta                              │  │
│  │     • get_state / list_pending_requests                   │  │
│  │     • tick / submit_plan                                  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           ROLLOUT + INFRA (scripts/, Dockerfile) ← Kayode       │
│  Async OpenReward client + OpenAI API → end-to-end test         │
│  Dockerfile, deployment, integration tests, env card            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Data model

### 3.1 Task spec (the JSON object every task contains)

This schema is the **single contract** between Kosi (generation) and Daniel (consumption). Both sides validate against it.

```python
TaskSpec = {
    "id": "london-routing-007",
    "difficulty": 7,                 # 1..100, monotonic
    "split": "train",                # tutorial | train | test
    "episode_date": "2026-04-25",    # ISO date used to sample weather
    "horizon_minutes": 960,          # 16 hours, 06:00–22:00
    "tick_minutes": 15,              # decision interval

    "depots": [                      # 1–3 garages
        {"id": "depot-0", "name": "Stockwell Bus Garage",
         "lat": 51.4720, "lon": -0.1226, "node_idx": 0}
    ],

    "vehicles": [                    # heterogeneous fleet, available from t=0
        {
            "id": "v-0", "type": "minibus_16", "depot_id": "depot-0",
            "capacity_seats": 16, "capacity_wheelchair": 1,
            "speed_factor": 1.0,     # 1.0 = nominal road speed
            "cost_per_km": 0.85,
            "shift_start": 0, "shift_end": 960
        },
        {"id": "v-1", "type": "accessible_van_8",
         "capacity_seats": 8, "capacity_wheelchair": 2,
         "speed_factor": 0.95, "cost_per_km": 0.70,
         "shift_start": 0, "shift_end": 720}
    ],

    "requests": [                    # heterogeneous demand
        {
            "id": "r-0",
            "kind": "passenger",     # passenger | parcel
            "pickup_node_idx": 14,
            "dropoff_node_idx": 22,
            "pickup_lat": 51.514, "pickup_lon": -0.099,
            "dropoff_lat": 51.523, "dropoff_lon": -0.158,
            "passengers": 3,
            "wheelchairs": 0,
            "earliest_pickup": 30,   # minutes from episode start
            "latest_pickup": 90,
            "earliest_dropoff": 60,
            "latest_dropoff": 150,
            "service_time": 2,       # boarding minutes
            "priority": 1,           # 1 (low) .. 5 (critical)
            "released_at": 0         # appears in pending list at this time
        }
    ],

    "nodes": [                        # geocoded real London locations
        {"idx": 0, "name": "Stockwell Bus Garage",
         "lat": 51.4720, "lon": -0.1226,
         "zone": 2, "category": "depot"}
    ],

    "distance_matrix_km": [[0.0, 1.7, 4.2]],   # N×N from OSRM
    "duration_matrix_min": [[0.0, 6.5, 11.2]], # N×N nominal driving time

    "weather_timeline": [             # one entry per hour, len 16-24
        {"t": 0,   "precip_mm": 0.0, "wind_kph": 8,  "visibility_km": 10,
         "temp_c": 12},
        {"t": 60,  "precip_mm": 0.4, "wind_kph": 12, "visibility_km": 8,
         "temp_c": 13}
    ],

    "traffic_events": [               # disruptions revealed over time
        {"t_reveal": 0,   "node_a": 14, "node_b": 22, "speed_factor": 0.4,
         "reason": "Synthesized: A23 Brixton Hill rush-hour congestion"},
        {"t_reveal": 120, "node_a": 5,  "node_b": 9,  "speed_factor": 0.6,
         "reason": "Synthesized: A4 incident"}
    ],

    "dynamic_events": [               # things that happen mid-episode
        {"t": 180, "type": "vehicle_breakdown", "vehicle_id": "v-2"},
        {"t": 240, "type": "new_request",       "request_id": "r-37"},
        {"t": 480, "type": "capacity_drop",     "vehicle_id": "v-0",
         "new_capacity_seats": 8,
         "reason": "passenger requires extra space"}
    ],

    "or_tools_baseline_cost": 482.3,  # from solver.py — total km
    "or_tools_baseline_unserved": 2,  # some requests may be infeasible
    "or_tools_baseline_served": 38    # count of served requests in baseline
}
```

The schema is **stable and fully deterministic**. No external API call is required during an episode.

### 3.2 Splits and difficulty

| Split | Tasks | Difficulty range | Purpose |
|---|---|---|---|
| `tutorial` | 5 | 1, 2, 3, 4, 5 | Easiest, no dynamic events. **Designed so a strong agent (e.g. GPT-5/Claude Opus) achieves ≥ 50% coverage on at least one task.** Used in tests. |
| `train` | 70 | 1–80 | Bulk RL training. Smooth difficulty ramp. |
| `test` | 25 | 50–100 | Evaluation. Includes hardest cases. |

Total: **100 tasks**.

Difficulty `d ∈ [1, 100]` maps deterministically to:

```python
def difficulty_params(d: int) -> dict:
    return {
        "n_nodes":           int(8 + d * 0.7),         # 8 → 78 nodes
        "n_vehicles":        2 + d // 20,              # 2 → 7
        "n_requests":        int(5 + d * 0.6),         # 5 → 65 requests
        "weather_severity":  min(1.0, d / 100),        # 0 → 1
        "traffic_density":   min(1.0, d / 80),         # 0 → 1
        "n_dynamic_events":  d // 10,                  # 0 → 10
        "heterogeneity_pct": min(1.0, d / 50),         # mix of vehicle types
        "pdptw_ratio":       min(1.0, d / 70),         # share of P&D requests
        "tw_tightness":      0.3 + 0.5 * (d / 100),    # 0.3 → 0.8
    }
```

**Tutorial calibration.** Difficulty 1 has exactly: 9 nodes, 2 vehicles (both identical 16-seat minibuses), 6 requests with wide time windows (`tw_tightness=0.3`), no traffic events, no dynamic events, no PDPTW (every request is a single-pickup-single-dropoff), and clear weather. A naive nearest-neighbor heuristic solves it; a strong LLM agent should hit ≥80% coverage easily.

Difficulty 100: 78 nodes, 7 vehicles of mixed types, 65 requests with tight windows, traffic on ~30 edges, 10 mid-episode events, 100% PDPTW. OR-Tools itself often leaves 5+ unserved.

---

## 4. Sub-problem decomposition (3 developers, balanced workload)

The work is partitioned into three streams of approximately equal size and complexity. Each stream has its own validation tests and can be developed independently. Integration is at well-defined interface boundaries.

| Dev | Stream | Files owned | Approx. LOC |
|-----|--------|-------------|-------------|
| **Kosi** | Data & generation | `scripts/generate_tasks.py`, `scripts/build_distance_matrix.py`, `scripts/fetch_weather.py`, `scripts/solver.py`, `scripts/synthesize_events.py`, `scripts/publish_to_hf.py`, `data/london_zones_1_4_pois.json`, `tests/test_generation.py`, `tests/test_distance_matrix.py`, `tests/test_solver.py` | ~700 |
| **Daniel** | Environment core | `src/state.py`, `src/feasibility.py`, `src/reward.py`, `src/time_engine.py`, `src/server.py`, `tests/test_state.py`, `tests/test_feasibility.py`, `tests/test_reward.py`, `tests/test_time_engine.py` | ~800 |
| **Kayode** | Rollout, infra & deployment | `scripts/run_rollout.py`, `Dockerfile`, `requirements.txt`, `README.md` (env card), `tests/conftest.py`, `tests/fixtures/*.json`, `tests/test_integration.py`, `tests/test_rollout.py`, deployment to OpenReward, HF dataset card | ~600 |

**Critical interface contract** between Kosi and Daniel: the `TaskSpec` JSON schema in §3.1. Once frozen (target: 20 min in), they ship in parallel against fixtures. Kayode's fixtures (`tests/fixtures/*.json`) match the same schema and are checked in early so Daniel can develop without waiting for Kosi's full generation run.

The dependency graph:

```
Kayode  ─── fixtures ───▶  Daniel  ─── server.py ───▶  Kayode (rollout)
   │                          ▲                            │
   │                          │ tasks.parquet              │
   └─── Dockerfile ───────────┴── Kosi ──────────────────► Kayode (deploy)
```

Kayode unblocks both Kosi and Daniel by producing fixtures first, then unblocks the demo by integrating their work last.

---

## 5. Stream A — Data & task generation (Kosi)

### 5.1 Files owned

```
data/
├── london_zones_1_4_pois.json    # 250+ canonical real London locations
├── osrm_cache.json               # cached distance matrices keyed by node-set hash
└── tasks.parquet                 # final 100 tasks (output)
scripts/
├── build_distance_matrix.py      # OSRM table service wrapper, cached
├── fetch_weather.py              # Open-Meteo wrapper
├── synthesize_events.py          # traffic + dynamic event synthesis
├── solver.py                     # OR-Tools PDPTW baseline
├── generate_tasks.py             # ⭐ orchestrator, writes tasks.parquet
└── publish_to_hf.py              # upload tasks.parquet to HuggingFace
tests/
├── test_distance_matrix.py
├── test_solver.py
└── test_generation.py
```

### 5.2 Sub-task A1: Canonical London POI list — `data/london_zones_1_4_pois.json`

**Goal.** A static JSON with 250+ real London locations across Zones 1–4, suitable as nodes for any task.

**Categories to cover.** Tube/rail stations, hospitals, shopping centres, residential nodes (ward centroids), schools, business hubs, depots/garages.

**Approach.** Write a one-shot script `scripts/seed_pois.py` (run once, output committed to repo) that uses Nominatim with rate limiting:

```python
# scripts/seed_pois.py — RUN ONCE, commit the output
import json, time, requests
from pathlib import Path

CATEGORIES = {
    "tube_station": ["King's Cross St Pancras", "Westminster", "Bank",
                     "Oxford Circus", "Liverpool Street", "Victoria",
                     "Waterloo", "London Bridge", "Paddington", "Euston",
                     "Canary Wharf", "Bond Street", "Knightsbridge",
                     "Notting Hill Gate", "Camden Town", "Angel",
                     "Old Street", "Brixton", "Stockwell", "Clapham Common",
                     "Hammersmith", "Earl's Court", "Shepherd's Bush",
                     "Wembley Park", "Stratford", "Greenwich",
                     "Putney Bridge", "Walthamstow Central"],
    "hospital":    ["St Thomas' Hospital London", "Guy's Hospital London",
                    "King's College Hospital London", "St Mary's Hospital London",
                    "Royal Free Hospital London", "University College Hospital London",
                    "Royal London Hospital", "Lewisham Hospital",
                    "Whittington Hospital", "Homerton Hospital",
                    "Chelsea and Westminster Hospital", "Charing Cross Hospital London"],
    "shopping":    ["Westfield London", "Westfield Stratford City",
                    "Selfridges London", "Harrods", "Borough Market",
                    "Camden Market", "Brick Lane Market",
                    "Covent Garden Market", "Brixton Market"],
    "landmark":    ["Tower of London", "British Museum", "Natural History Museum",
                    "Tate Modern", "London Eye", "Big Ben",
                    "Buckingham Palace", "Trafalgar Square",
                    "Hyde Park Corner", "Regent's Park"],
    "depot":       ["Stockwell Bus Garage London", "Holloway Bus Garage London",
                    "West Ham Bus Garage London", "Norwood Bus Garage London"],
    "school":      ["UCL London", "King's College London", "Imperial College London",
                    "LSE London", "Queen Mary University London",
                    "City University London", "Goldsmiths London",
                    "SOAS London"],
    "residential": ["Islington London", "Hackney London", "Camden London",
                    "Lambeth London", "Southwark London", "Tower Hamlets London",
                    "Wandsworth London", "Lewisham London", "Greenwich London",
                    "Hammersmith London", "Kensington London", "Chelsea London",
                    "Westminster London", "Fulham London", "Notting Hill London"],
}

def geocode(query):
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params={"q": query, "format": "json", "limit": 1},
                     headers={"User-Agent": "openreward-london-routing/0.1"},
                     timeout=10)
    r.raise_for_status()
    j = r.json()
    return (float(j[0]["lat"]), float(j[0]["lon"])) if j else None

def zone_for(lat, lon):
    """Crude zone estimate by distance from Charing Cross."""
    cx_lat, cx_lon = 51.5074, -0.1278
    d_km = ((lat - cx_lat) * 111) ** 2 + ((lon - cx_lon) * 70) ** 2
    d_km = d_km ** 0.5
    if d_km <  3: return 1
    if d_km <  6: return 2
    if d_km < 10: return 3
    return 4

def main():
    out = []
    for category, names in CATEGORIES.items():
        for name in names:
            try:
                coords = geocode(name)
                if not coords: continue
                lat, lon = coords
                if not (51.28 < lat < 51.70 and -0.51 < lon < 0.33):
                    continue
                out.append({"name": name, "lat": lat, "lon": lon,
                            "zone": zone_for(lat, lon), "category": category})
                time.sleep(1.1)  # Nominatim 1 req/s policy
            except Exception as e:
                print(f"Skip {name}: {e}")
    Path("data/london_zones_1_4_pois.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(out)} POIs")

if __name__ == "__main__":
    main()
```

If Nominatim is flaky, fall back to a hand-curated JSON (we list ~80 well-known landmarks above; lat/lon can be looked up manually in 30 minutes — that's the emergency path).

**Validation tests** (`tests/test_pois.py`):

```python
import json
from pathlib import Path

def test_poi_count():
    pois = json.loads(Path("data/london_zones_1_4_pois.json").read_text())
    assert len(pois) >= 80, f"only {len(pois)} POIs"

def test_poi_inside_london_bbox():
    pois = json.loads(Path("data/london_zones_1_4_pois.json").read_text())
    for p in pois:
        assert 51.28 < p["lat"] < 51.70, p
        assert -0.51 < p["lon"] < 0.33, p

def test_poi_categories_diverse():
    pois = json.loads(Path("data/london_zones_1_4_pois.json").read_text())
    cats = {p["category"] for p in pois}
    for required in ["tube_station", "hospital", "depot", "landmark"]:
        assert required in cats, f"missing category {required}"

def test_poi_all_zones_covered():
    pois = json.loads(Path("data/london_zones_1_4_pois.json").read_text())
    zones = {p["zone"] for p in pois}
    assert zones >= {1, 2, 3}, f"missing zones: {zones}"
```

### 5.3 Sub-task A2: OSRM distance matrix builder — `scripts/build_distance_matrix.py`

**Goal.** Given a list of `(lon, lat)` pairs, return `(distance_km_matrix, duration_min_matrix)` of floats, with disk caching.

**Endpoint.** OSRM table service:
```
GET {OSRM_BASE_URL}/table/v1/driving/{lon,lat;lon,lat;...}?annotations=duration,distance
```

Hard limits: 100 sources × 100 destinations = 10,000 cells per request. Our largest task has 78 nodes (78×78 = 6,084 cells), so we never split. We *do* watch URL length — at 78 nodes × ~17 chars ≈ 1.4KB, comfortably under any limit.

```python
# scripts/build_distance_matrix.py
import hashlib, json, os, requests, time
from pathlib import Path

CACHE_PATH = Path("data/osrm_cache.json")
OSRM_BASE = os.environ.get("OSRM_BASE_URL", "https://router.project-osrm.org")

def _cache_key(coords):
    """Stable key from coords. Don't sort — order matters (matrix is N×N)."""
    rounded = [(round(lon, 6), round(lat, 6)) for lon, lat in coords]
    return hashlib.sha256(json.dumps(rounded).encode()).hexdigest()

def _load_cache():
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text())
    except Exception:
        return {}

def _save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    tmp.replace(CACHE_PATH)

def haversine_km(a, b):
    import math
    lon1, lat1 = a; lon2, lat2 = b
    R = 6371.0
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    h = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(h))

def _haversine_fallback(coords):
    """Emergency: if OSRM is down, use Haversine × 1.4 (urban driving fudge)
    and 30 km/h average speed. Documented in README."""
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    dur = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            d = haversine_km(coords[i], coords[j]) * 1.4
            dist[i][j] = d
            dur[i][j] = d / 30.0 * 60  # 30 km/h → minutes
    return dist, dur

def build_matrix(coords, allow_fallback=True):
    """coords: list of (lon, lat) tuples. Returns (dist_km, dur_min) — both N×N."""
    cache = _load_cache()
    key = _cache_key(coords)
    if key in cache:
        return cache[key]["dist_km"], cache[key]["dur_min"]

    coord_str = ";".join(f"{lon},{lat}" for lon, lat in coords)
    url = f"{OSRM_BASE}/table/v1/driving/{coord_str}?annotations=duration,distance"

    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=45)
            r.raise_for_status()
            data = r.json()
            if data.get("code") != "Ok":
                raise RuntimeError(f"OSRM not Ok: {data.get('code')}")
            dur_min = [[(d or 0) / 60.0 for d in row] for row in data["durations"]]
            dist_km = [[(d or 0) / 1000.0 for d in row] for row in data["distances"]]
            cache[key] = {"dist_km": dist_km, "dur_min": dur_min}
            _save_cache(cache)
            return dist_km, dur_min
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)

    if allow_fallback:
        print(f"⚠ OSRM failed ({last_err}); using Haversine fallback")
        dist, dur = _haversine_fallback(coords)
        cache[key] = {"dist_km": dist, "dur_min": dur, "fallback": True}
        _save_cache(cache)
        return dist, dur
    raise last_err
```

**Validation tests** (`tests/test_distance_matrix.py`):

```python
import time, pytest
from scripts.build_distance_matrix import build_matrix, haversine_km

def test_matrix_self_distance_zero():
    coords = [(-0.1276, 51.5074), (-0.1419, 51.5014)]
    dist, dur = build_matrix(coords)
    assert dist[0][0] == 0.0 and dur[0][0] == 0.0
    assert dist[1][1] == 0.0 and dur[1][1] == 0.0

def test_matrix_shape():
    coords = [(-0.1276, 51.5074), (-0.1419, 51.5014), (-0.0759, 51.5081)]
    dist, dur = build_matrix(coords)
    assert len(dist) == 3 and all(len(r) == 3 for r in dist)
    assert len(dur) == 3 and all(len(r) == 3 for r in dur)

def test_matrix_positive_offdiag():
    coords = [(-0.1276, 51.5074), (-0.1419, 51.5014)]
    dist, _ = build_matrix(coords)
    assert dist[0][1] > 0 and dist[1][0] > 0

def test_matrix_cache_hit_fast():
    coords = [(-0.1276, 51.5074), (-0.1419, 51.5014)]
    build_matrix(coords)  # populate
    t0 = time.time()
    build_matrix(coords)
    assert time.time() - t0 < 0.1, "second call should be cached"

def test_haversine_sanity():
    # London to Paris ~ 344 km
    london = (-0.1276, 51.5074); paris = (2.3522, 48.8566)
    assert 320 < haversine_km(london, paris) < 360
```

### 5.4 Sub-task A3: Weather fetcher — `scripts/fetch_weather.py`

```python
# scripts/fetch_weather.py
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
        # Rain bursts more likely under high severity
        precip = max(0, rng.gauss(severity * 3, severity * 2))
        timeline.append({
            "t": i * 60,
            "precip_mm": round(precip, 2),
            "wind_kph":  round(8 + severity * rng.uniform(0, 30), 1),
            "visibility_km": max(0.5, 10 - severity * rng.uniform(0, 8)),
            "temp_c": round(8 + rng.uniform(0, 10), 1),
        })
    return timeline
```

Validation:

```python
# tests/test_weather.py
from scripts.fetch_weather import (fetch_weather, weather_speed_factor,
                                   synthetic_weather, weather_at)

def test_weather_24h():
    tl = fetch_weather("2026-04-25")
    assert len(tl) == 24
    assert all("precip_mm" in e for e in tl)

def test_weather_speed_factor_clear():
    f = weather_speed_factor({"precip_mm": 0, "visibility_km": 10})
    assert f == 1.0

def test_weather_speed_factor_storm():
    f = weather_speed_factor({"precip_mm": 8, "visibility_km": 0.8})
    assert f < 0.6  # both penalties stack

def test_synthetic_severity_scales():
    mild = synthetic_weather(0.0)
    severe = synthetic_weather(1.0)
    assert sum(e["precip_mm"] for e in mild) <= sum(e["precip_mm"] for e in severe)

def test_weather_at_lookup():
    tl = synthetic_weather(0.5)
    w = weather_at(tl, 0)        # episode minute 0 → real-time 06:00 → hour 6
    assert w == tl[6]
```

### 5.5 Sub-task A4: Event synthesis — `scripts/synthesize_events.py`

This module creates the **traffic events** and **dynamic events** that drive multi-horizon dynamics. Both are seeded from the task seed for full reproducibility.

```python
# scripts/synthesize_events.py
import random

def synthesize_traffic_events(nodes, distance_matrix, n_events, rng):
    """Generate plausible traffic disruptions.
    Mix of (a) rush-hour congestion (07:00–09:30 in episode time = 60–210 min),
            (b) midday incidents on random arterials,
            (c) evening rush (15:00–18:00 = 540–720 min)."""
    events = []
    n = len(nodes)

    def random_pair():
        a = rng.randrange(n); b = rng.randrange(n)
        while b == a: b = rng.randrange(n)
        return a, b

    rush_morning  = max(1, n_events // 3)
    incidents     = max(1, n_events // 3)
    rush_evening  = n_events - rush_morning - incidents

    for _ in range(rush_morning):
        a, b = random_pair()
        events.append({
            "t_reveal": 0,                              # known at start
            "node_a": a, "node_b": b,
            "speed_factor": rng.uniform(0.4, 0.7),
            "reason": "Synthesized: morning rush-hour congestion"
        })
    for _ in range(incidents):
        a, b = random_pair()
        events.append({
            "t_reveal": rng.randint(120, 540),          # 08:00–15:00
            "node_a": a, "node_b": b,
            "speed_factor": rng.uniform(0.3, 0.6),
            "reason": rng.choice([
                "Synthesized: A-road incident", "Synthesized: roadworks",
                "Synthesized: lane closure", "Synthesized: collision cleanup"])
        })
    for _ in range(rush_evening):
        a, b = random_pair()
        events.append({
            "t_reveal": rng.randint(480, 600),          # 14:00–16:00
            "node_a": a, "node_b": b,
            "speed_factor": rng.uniform(0.4, 0.7),
            "reason": "Synthesized: evening rush-hour congestion"
        })
    events.sort(key=lambda e: e["t_reveal"])
    return events


def synthesize_dynamic_events(vehicles, requests_, n_events, horizon, rng):
    """Mix of breakdowns, new requests, and capacity drops."""
    events = []
    if n_events == 0:
        return events

    # New requests appear later in the day, must be added to requests list separately
    n_breakdowns      = max(0, n_events // 3)
    n_capacity_drops  = max(0, n_events // 4)
    n_new_requests    = n_events - n_breakdowns - n_capacity_drops

    # Breakdowns: not in first hour (give agent time to plan)
    for _ in range(n_breakdowns):
        v = rng.choice(vehicles)
        events.append({
            "t": rng.randint(60, horizon - 120),
            "type": "vehicle_breakdown",
            "vehicle_id": v["id"],
        })

    # Capacity drops: existing vehicle gets smaller mid-day
    for _ in range(n_capacity_drops):
        v = rng.choice(vehicles)
        if v["capacity_seats"] <= 4: continue  # too small to drop
        events.append({
            "t": rng.randint(120, horizon - 60),
            "type": "capacity_drop",
            "vehicle_id": v["id"],
            "new_capacity_seats": max(2, v["capacity_seats"] // 2),
            "reason": rng.choice([
                "wheelchair passenger requires extra space",
                "luggage takes additional seats",
                "vehicle partial mechanical issue limits capacity"])
        })

    # New requests: ids must be coordinated with the requests list — return
    # placeholders here; the caller resolves them.
    for i in range(n_new_requests):
        events.append({
            "t": rng.randint(120, horizon - 180),
            "type": "new_request",
            "request_id": f"r-late-{i}",  # caller must create these
        })

    events.sort(key=lambda e: e["t"])
    return events
```

Validation:

```python
# tests/test_events.py
import random
from scripts.synthesize_events import (synthesize_traffic_events,
                                       synthesize_dynamic_events)

def test_traffic_event_shape():
    nodes = [{"idx": i} for i in range(10)]
    rng = random.Random(42)
    evs = synthesize_traffic_events(nodes, None, 6, rng)
    assert len(evs) == 6
    for e in evs:
        assert 0 <= e["node_a"] < 10 and 0 <= e["node_b"] < 10
        assert e["node_a"] != e["node_b"]
        assert 0.3 <= e["speed_factor"] <= 0.7
        assert "reason" in e

def test_traffic_events_sorted():
    nodes = [{"idx": i} for i in range(10)]
    evs = synthesize_traffic_events(nodes, None, 9, random.Random(1))
    times = [e["t_reveal"] for e in evs]
    assert times == sorted(times)

def test_dynamic_events_balanced():
    vehicles = [{"id": f"v-{i}", "capacity_seats": 16} for i in range(4)]
    evs = synthesize_dynamic_events(vehicles, [], 9, 960, random.Random(7))
    types = [e["type"] for e in evs]
    assert "vehicle_breakdown" in types
    assert any(t == "new_request" for t in types)

def test_dynamic_events_zero():
    evs = synthesize_dynamic_events([], [], 0, 960, random.Random(0))
    assert evs == []
```

### 5.6 Sub-task A5: OR-Tools baseline solver — `scripts/solver.py`

For each generated task, compute a reference solution to grade the agent against. This is the "optimum" denominator in terminal reward.

```python
# scripts/solver.py
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Penalty for dropping a request — large enough that the solver prefers to
# serve a request unless infeasible. Tune if instances become unsolvable.
DISJUNCTION_PENALTY = 100_000

def solve_baseline(task, time_limit_s=10):
    """Returns dict with total_cost_km, n_unserved, n_served, routes."""
    n_nodes = len(task["nodes"])
    n_vehicles = len(task["vehicles"])
    depot_idx = task["depots"][0]["node_idx"]
    horizon = task["horizon_minutes"]

    # Adjusted duration matrix: include traffic events known at t=0
    dur = [row[:] for row in task["duration_matrix_min"]]
    for ev in task["traffic_events"]:
        if ev["t_reveal"] == 0:
            a, b = ev["node_a"], ev["node_b"]
            if dur[a][b] > 0:
                dur[a][b] /= ev["speed_factor"]
                dur[b][a] /= ev["speed_factor"]

    manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)

    # Cost = distance in metres
    def dist_cb(i, j):
        return int(task["distance_matrix_km"][manager.IndexToNode(i)]
                                              [manager.IndexToNode(j)] * 1000)
    transit_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # ---- Capacity dimension (pickup adds, dropoff removes) ----
    demands = [0] * n_nodes
    for r in task["requests"]:
        demands[r["pickup_node_idx"]] += r["passengers"]
        demands[r["dropoff_node_idx"]] -= r["passengers"]

    def demand_cb(i):
        return demands[manager.IndexToNode(i)]

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    capacities = [v["capacity_seats"] for v in task["vehicles"]]
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx, 0, capacities, True, "Capacity")

    # ---- Time dimension ----
    def time_cb(i, j):
        a = manager.IndexToNode(i); b = manager.IndexToNode(j)
        # Round up to ensure we don't beat physics
        return max(1, int(dur[a][b] + 0.5))

    time_cb_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_cb_idx, 60, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    # ---- PDPTW pairs + time windows ----
    for r in task["requests"]:
        p = manager.NodeToIndex(r["pickup_node_idx"])
        d = manager.NodeToIndex(r["dropoff_node_idx"])
        routing.AddPickupAndDelivery(p, d)
        routing.solver().Add(routing.VehicleVar(p) == routing.VehicleVar(d))
        routing.solver().Add(time_dim.CumulVar(p) <= time_dim.CumulVar(d))
        time_dim.CumulVar(p).SetRange(int(r["earliest_pickup"]),
                                      int(r["latest_pickup"]))
        time_dim.CumulVar(d).SetRange(int(r["earliest_dropoff"]),
                                      int(r["latest_dropoff"]))
        # Allow drop with penalty (handles infeasible requests gracefully)
        routing.AddDisjunction([p, d], DISJUNCTION_PENALTY)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.seconds = time_limit_s

    sol = routing.SolveWithParameters(params)
    if not sol:
        return {"total_cost_km": float("inf"),
                "n_unserved": len(task["requests"]),
                "n_served": 0, "routes": []}

    # Extract routes + count served
    served_pickups = set()
    routes = []
    total_dist_m = 0
    for v in range(n_vehicles):
        idx = routing.Start(v)
        route_nodes = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route_nodes.append(node)
            next_idx = sol.Value(routing.NextVar(idx))
            if not routing.IsEnd(next_idx):
                total_dist_m += dist_cb(idx, next_idx)
            idx = next_idx
        routes.append(route_nodes)

    pickup_nodes = {r["pickup_node_idx"]: r["id"] for r in task["requests"]}
    for route in routes:
        for node in route:
            if node in pickup_nodes:
                served_pickups.add(pickup_nodes[node])
    n_served = len(served_pickups)
    n_unserved = len(task["requests"]) - n_served

    return {"total_cost_km": total_dist_m / 1000.0,
            "n_unserved": n_unserved,
            "n_served": n_served,
            "routes": routes}
```

Validation:

```python
# tests/test_solver.py
from scripts.solver import solve_baseline

def _trivial_task():
    return {
        "horizon_minutes": 480,
        "nodes": [{"idx": 0}, {"idx": 1}, {"idx": 2}],
        "depots": [{"node_idx": 0}],
        "vehicles": [{"id": "v0", "capacity_seats": 8}],
        "requests": [{"id": "r0", "pickup_node_idx": 1, "dropoff_node_idx": 2,
                      "passengers": 1, "earliest_pickup": 0,
                      "latest_pickup": 200, "earliest_dropoff": 0,
                      "latest_dropoff": 300}],
        "distance_matrix_km":  [[0,2,3],[2,0,1],[3,1,0]],
        "duration_matrix_min": [[0,5,8],[5,0,3],[8,3,0]],
        "traffic_events": [],
    }

def test_solver_serves_trivial():
    out = solve_baseline(_trivial_task(), time_limit_s=3)
    assert out["n_unserved"] == 0
    assert 0 < out["total_cost_km"] < 20

def test_solver_infeasible_capacity():
    t = _trivial_task()
    t["requests"][0]["passengers"] = 999  # overflow capacity
    out = solve_baseline(t, time_limit_s=3)
    assert out["n_unserved"] == 1
```


### 5.7 Sub-task A6: The orchestrator — `scripts/generate_tasks.py`

Reads POIs, samples per difficulty, builds matrix, fetches weather, synthesizes events, calls solver, writes parquet.

```python
# scripts/generate_tasks.py
"""Generate the 100-task london-dynamic-routing dataset.

Usage:
  export OSRM_BASE_URL=https://router.project-osrm.org   # or self-hosted
  python scripts/generate_tasks.py
  python scripts/generate_tasks.py --quick   # 5 tutorial tasks only

Outputs:
  data/tasks.parquet
"""
import argparse, json, random, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from scripts.build_distance_matrix import build_matrix
from scripts.fetch_weather import synthetic_weather
from scripts.synthesize_events import (synthesize_traffic_events,
                                       synthesize_dynamic_events)
from scripts.solver import solve_baseline


def difficulty_params(d: int) -> dict:
    return {
        "n_nodes":           int(8 + d * 0.7),
        "n_vehicles":        2 + d // 20,
        "n_requests":        int(5 + d * 0.6),
        "weather_severity":  min(1.0, d / 100),
        "traffic_density":   min(1.0, d / 80),
        "n_dynamic_events":  d // 10,
        "heterogeneity_pct": min(1.0, d / 50),
        "pdptw_ratio":       min(1.0, d / 70),
        "tw_tightness":      0.3 + 0.5 * (d / 100),
    }


VEHICLE_TYPES = [
    {"type": "minibus_16", "capacity_seats": 16, "capacity_wheelchair": 1,
     "speed_factor": 1.0, "cost_per_km": 0.85},
    {"type": "accessible_van_8", "capacity_seats": 8, "capacity_wheelchair": 2,
     "speed_factor": 0.95, "cost_per_km": 0.70},
    {"type": "minibus_24", "capacity_seats": 24, "capacity_wheelchair": 1,
     "speed_factor": 0.90, "cost_per_km": 1.05},
    {"type": "small_van_4", "capacity_seats": 4, "capacity_wheelchair": 0,
     "speed_factor": 1.05, "cost_per_km": 0.55},
]


def synthesize_vehicles(n, depot_idx, heterogeneity, rng):
    """Heterogeneity 0 → all identical minibus_16. 1 → maximally mixed."""
    vehicles = []
    for i in range(n):
        if heterogeneity < 0.001 or rng.random() > heterogeneity:
            template = VEHICLE_TYPES[0]
        else:
            template = rng.choice(VEHICLE_TYPES)
        shift_end = 960 if rng.random() > 0.2 else rng.randint(480, 900)
        vehicles.append({
            "id": f"v-{i}",
            "type": template["type"],
            "depot_id": "depot-0",
            "capacity_seats": template["capacity_seats"],
            "capacity_wheelchair": template["capacity_wheelchair"],
            "speed_factor": template["speed_factor"],
            "cost_per_km": template["cost_per_km"],
            "shift_start": 0,
            "shift_end": shift_end,
        })
    return vehicles


def synthesize_requests(n, nodes, depot_idx, params, rng, horizon=960):
    """Generate n PDPTW requests over the day."""
    reqs = []
    non_depot = [n_["idx"] for n_ in nodes if n_["idx"] != depot_idx]
    if len(non_depot) < 2:
        return reqs
    tw_tightness = params["tw_tightness"]
    pdptw_ratio = params["pdptw_ratio"]

    for i in range(n):
        pu = rng.choice(non_depot)
        do = rng.choice([x for x in non_depot if x != pu])
        passengers = rng.choices([1, 2, 3, 4, 5, 6], [4, 3, 2, 1, 1, 1])[0]
        wheelchairs = (1 if rng.random() < 0.10 else 0)
        priority = rng.choices([1, 2, 3, 4, 5], [5, 3, 2, 1, 1])[0]

        # Time window. Released-at scattered through the day.
        released_at = (0 if rng.random() < 0.6
                       else rng.randint(60, horizon - 240))
        center = rng.randint(released_at + 30, horizon - 60)
        # Tighter windows under high tw_tightness
        half_width = int(60 + (1 - tw_tightness) * 240)
        e_pu = max(released_at, center - half_width)
        l_pu = min(horizon - 30, center + half_width)
        # Dropoff window starts after pickup and is also constrained
        e_do = e_pu + 10
        l_do = min(horizon, l_pu + 90 + int((1 - tw_tightness) * 120))

        kind = ("passenger"
                if rng.random() < pdptw_ratio else "passenger")  # reserved for future
        reqs.append({
            "id": f"r-{i}",
            "kind": kind,
            "pickup_node_idx": pu,
            "dropoff_node_idx": do,
            "pickup_lat": nodes[pu]["lat"], "pickup_lon": nodes[pu]["lon"],
            "dropoff_lat": nodes[do]["lat"], "dropoff_lon": nodes[do]["lon"],
            "passengers": passengers,
            "wheelchairs": wheelchairs,
            "earliest_pickup": e_pu,
            "latest_pickup": l_pu,
            "earliest_dropoff": e_do,
            "latest_dropoff": l_do,
            "service_time": 2,
            "priority": priority,
            "released_at": released_at,
        })
    return reqs


def generate_task(seed: int, difficulty: int, split: str, pois: list,
                  episode_date: str = "2026-04-25"):
    rng = random.Random(seed)
    params = difficulty_params(difficulty)

    n_nodes = min(params["n_nodes"], len(pois))
    sampled = rng.sample(pois, n_nodes)
    # Force at least one depot in the sampled set
    if not any(p["category"] == "depot" for p in sampled):
        depots = [p for p in pois if p["category"] == "depot"]
        if depots:
            sampled[-1] = rng.choice(depots)

    nodes = [{"idx": i, **p} for i, p in enumerate(sampled)]
    depot_node = next((n for n in nodes if n["category"] == "depot"), nodes[0])
    depot_idx = depot_node["idx"]

    coords = [(n["lon"], n["lat"]) for n in nodes]
    dist_km, dur_min = build_matrix(coords)

    vehicles = synthesize_vehicles(params["n_vehicles"], depot_idx,
                                   params["heterogeneity_pct"], rng)

    requests_ = synthesize_requests(params["n_requests"], nodes, depot_idx,
                                    params, rng)

    weather = synthetic_weather(params["weather_severity"])

    # Traffic
    n_traffic = max(0, int(params["traffic_density"] * 12))
    traffic_events = synthesize_traffic_events(nodes, dist_km, n_traffic, rng)

    # Dynamic events: breakdowns, capacity drops, late requests
    dynamic_events = synthesize_dynamic_events(
        vehicles, requests_, params["n_dynamic_events"], 960, rng)
    # Resolve placeholder request_ids → actual late requests
    late_idx = 0
    for ev in dynamic_events:
        if ev["type"] == "new_request":
            new_id = f"r-late-{late_idx}"; late_idx += 1
            late_req = synthesize_requests(1, nodes, depot_idx, params, rng)
            if late_req:
                lr = late_req[0]
                lr["id"] = new_id
                lr["released_at"] = ev["t"]
                lr["earliest_pickup"] = max(lr["earliest_pickup"], ev["t"])
                requests_.append(lr)
                ev["request_id"] = new_id

    task = {
        "id": f"london-routing-{seed:05d}",
        "difficulty": difficulty,
        "split": split,
        "episode_date": episode_date,
        "horizon_minutes": 960,
        "tick_minutes": 15,
        "depots": [{"id": "depot-0", "name": depot_node["name"],
                    "lat": depot_node["lat"], "lon": depot_node["lon"],
                    "node_idx": depot_idx}],
        "vehicles": vehicles,
        "requests": requests_,
        "nodes": nodes,
        "distance_matrix_km": dist_km,
        "duration_matrix_min": dur_min,
        "weather_timeline": weather,
        "traffic_events": traffic_events,
        "dynamic_events": dynamic_events,
    }

    # Solver baseline
    out = solve_baseline(task, time_limit_s=8)
    task["or_tools_baseline_cost"] = out["total_cost_km"]
    task["or_tools_baseline_unserved"] = out["n_unserved"]
    task["or_tools_baseline_served"] = out["n_served"]
    return task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="generate 5 tutorial tasks only (smoke)")
    ap.add_argument("--out", default="data/tasks.parquet")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    pois = json.loads(Path("data/london_zones_1_4_pois.json").read_text())
    print(f"Loaded {len(pois)} POIs")

    plan = []
    for i in range(5):
        plan.append(("tutorial", 10_000 + i, i + 1))
    if not args.quick:
        for i in range(70):
            d = 1 + (i * 79) // 69
            plan.append(("train", 20_000 + i, d))
        for i in range(25):
            d = 50 + (i * 50) // 24
            plan.append(("test", 30_000 + i, d))

    print(f"Generating {len(plan)} tasks...")
    tasks = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(generate_task, seed, d, split, pois): (split, seed, d)
                   for split, seed, d in plan}
        for i, fut in enumerate(as_completed(futures)):
            split, seed, d = futures[fut]
            try:
                t = fut.result()
                tasks.append(t)
                print(f"  [{i+1}/{len(plan)}] {t['id']} d={d} "
                      f"served={t['or_tools_baseline_served']}/"
                      f"{len(t['requests'])} cost={t['or_tools_baseline_cost']:.1f}")
            except Exception as e:
                print(f"  FAIL {split} seed={seed} d={d}: {e}")

    tasks.sort(key=lambda t: (t["split"], t["id"]))
    df = pd.DataFrame(tasks)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"\nWrote {len(tasks)} tasks → {args.out}")
    print(f"  tutorial: {(df['split']=='tutorial').sum()}")
    print(f"  train:    {(df['split']=='train').sum()}")
    print(f"  test:     {(df['split']=='test').sum()}")

if __name__ == "__main__":
    main()
```

Validation:

```python
# tests/test_generation.py
import pandas as pd

def test_full_dataset_shape():
    df = pd.read_parquet("data/tasks.parquet")
    assert len(df) == 100
    assert (df["split"] == "tutorial").sum() == 5
    assert (df["split"] == "train").sum() == 70
    assert (df["split"] == "test").sum() == 25

def test_difficulty_monotonic_signals():
    df = pd.read_parquet("data/tasks.parquet")
    d_low = df[df["difficulty"] <= 10]
    d_high = df[df["difficulty"] >= 80]
    avg_low_reqs  = d_low["requests"].apply(len).mean()
    avg_high_reqs = d_high["requests"].apply(len).mean()
    assert avg_high_reqs > 2 * avg_low_reqs

def test_solver_baseline_present():
    df = pd.read_parquet("data/tasks.parquet")
    assert (df["or_tools_baseline_cost"] < float("inf")).sum() >= 90
    # Tutorial split must always solve fully
    tut = df[df["split"] == "tutorial"]
    assert (tut["or_tools_baseline_unserved"] == 0).all()

def test_task_ids_unique():
    df = pd.read_parquet("data/tasks.parquet")
    assert df["id"].is_unique
```

### 5.8 Sub-task A7: HF Hub publishing — `scripts/publish_to_hf.py`

```python
# scripts/publish_to_hf.py
import os
from huggingface_hub import HfApi
from pathlib import Path

REPO_ID = os.environ.get("HF_DATASET_REPO", "EnvCommons/london-dynamic-routing")

def main():
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj="data/tasks.parquet",
        path_in_repo="tasks.parquet",
        repo_id=REPO_ID, repo_type="dataset",
        commit_message="Add 100-task London dynamic routing dataset")
    # Upload dataset card
    card = Path("README_dataset.md")
    if card.exists():
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                        repo_id=REPO_ID, repo_type="dataset",
                        commit_message="Add dataset card")
    print(f"Published → https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()
```

Acceptance: dataset visible at the HF Hub URL with a README documenting the schema.

### 5.9 Kosi's checklist

- [ ] `data/london_zones_1_4_pois.json` ≥ 80 entries, all 4 zones, ≥ 4 categories
- [ ] `tests/test_distance_matrix.py` 5/5 passing locally with `OSRM_BASE_URL` set
- [ ] `tests/test_solver.py` passes (trivial + infeasibility cases)
- [ ] `tests/test_events.py` passes
- [ ] `python scripts/generate_tasks.py --quick` produces 5 tutorial tasks in `data/tasks.parquet`
- [ ] `python scripts/generate_tasks.py` produces full 100 tasks
- [ ] `tests/test_generation.py` passes
- [ ] `python scripts/publish_to_hf.py` succeeds

---

## 6. Stream B — Environment core (Daniel)

### 6.1 Files owned

```
src/
├── __init__.py
├── state.py            # EpisodeState, Stop, all bookkeeping
├── feasibility.py      # all constraint checks, marginal cost
├── reward.py           # reward function (single source of truth)
├── time_engine.py      # tick(), event queue, ETA recomputation
└── server.py           # ORS Environment class + 14 tools
tests/
├── test_state.py
├── test_feasibility.py
├── test_reward.py
└── test_time_engine.py
```

### 6.2 State machine — `src/state.py`

The episode state is fully captured by a single dataclass. All fields are JSON-serializable so the env supports introspection.

```python
# src/state.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Stop:
    request_id: str
    kind: str        # 'pickup' | 'dropoff'
    node_idx: int
    eta_minutes: float    # when vehicle is expected to arrive
    completed: bool = False

@dataclass
class EpisodeState:
    task: dict                          # frozen TaskSpec

    # Time
    current_time: int = 0               # episode minutes from t=0

    # Routes: vehicle_id → ordered list of Stops still to do
    routes: dict = field(default_factory=dict)

    # Vehicle status: 'available' | 'on_route' | 'broken' | 'inactive'
    vehicle_status: dict = field(default_factory=dict)

    # Capacity overrides (from capacity_drop events)
    vehicle_capacity_override: dict = field(default_factory=dict)

    # Request status: 'unreleased' | 'pending' | 'assigned' | 'in_vehicle'
    #                 'completed' | 'deferred' | 'cancelled'
    request_status: dict = field(default_factory=dict)
    request_assigned_to: dict = field(default_factory=dict)  # req_id → veh_id

    # Revealed traffic (a, b) → speed_factor
    revealed_traffic: dict = field(default_factory=dict)

    # Bookkeeping
    invalid_action_count: int = 0
    total_actions: int = 0
    realized_cost_km: float = 0.0
    served_requests: set = field(default_factory=set)

    def vehicle(self, vid: str) -> dict:
        v = next(v for v in self.task["vehicles"] if v["id"] == vid)
        if vid in self.vehicle_capacity_override:
            v = {**v, **self.vehicle_capacity_override[vid]}
        return v

    def request(self, rid: str) -> dict:
        return next(r for r in self.task["requests"] if r["id"] == rid)

    def initialize(self):
        for v in self.task["vehicles"]:
            self.vehicle_status[v["id"]] = "available"
            self.routes[v["id"]] = []
        for r in self.task["requests"]:
            self.request_status[r["id"]] = ("pending"
                                            if r.get("released_at", 0) == 0
                                            else "unreleased")

    def pending_request_ids(self):
        return [rid for rid, s in self.request_status.items() if s == "pending"]

    def is_assigned(self, rid: str) -> bool:
        return self.request_status.get(rid) in ("assigned", "in_vehicle")

    def route_load_at_position(self, vid: str, pos: int) -> int:
        """Number of seats occupied just before stop at position `pos`."""
        load = 0
        for i, stop in enumerate(self.routes[vid][:pos]):
            r = self.request(stop.request_id)
            if stop.kind == "pickup":   load += r["passengers"]
            elif stop.kind == "dropoff": load -= r["passengers"]
        return load
```

### 6.3 Feasibility checks — `src/feasibility.py`

Single source of truth for whether a proposed `assign(...)` is legal.

```python
# src/feasibility.py
from .state import EpisodeState, Stop

def get_edge_duration(state: EpisodeState, a: int, b: int) -> float:
    """Driving duration in minutes a→b, accounting for revealed traffic
    AND current weather."""
    base = state.task["duration_matrix_min"][a][b]
    sf = state.revealed_traffic.get((a, b), 1.0)
    # Weather speed factor (look up current hour's weather)
    from scripts.fetch_weather import weather_at, weather_speed_factor
    w = weather_at(state.task["weather_timeline"], state.current_time)
    sf *= weather_speed_factor(w)
    return base / max(sf, 0.1)


def get_edge_distance(state: EpisodeState, a: int, b: int) -> float:
    return state.task["distance_matrix_km"][a][b]


def check_feasibility(state, request_id, vehicle_id, pickup_pos, dropoff_pos):
    """Returns (ok: bool, reason: str, marginal_km: float).
    Does NOT mutate state."""
    if state.request_status.get(request_id) not in ("pending",):
        return False, f"request not pending (status: "\
                      f"{state.request_status.get(request_id)})", 0.0
    if vehicle_id not in state.routes:
        return False, "unknown vehicle", 0.0
    if state.vehicle_status[vehicle_id] in ("broken", "inactive"):
        return False, f"vehicle {state.vehicle_status[vehicle_id]}", 0.0

    veh = state.vehicle(vehicle_id)
    req = state.request(request_id)

    if req.get("wheelchairs", 0) > veh.get("capacity_wheelchair", 0):
        return False, "vehicle lacks wheelchair capacity", 0.0

    route = state.routes[vehicle_id]
    n = len(route)
    if not (0 <= pickup_pos <= n) or not (pickup_pos < dropoff_pos <= n + 1):
        return False, "invalid positions (need 0 <= p_pos < d_pos)", 0.0

    # Build prospective new route
    pickup_stop = Stop(request_id=request_id, kind="pickup",
                       node_idx=req["pickup_node_idx"], eta_minutes=0)
    dropoff_stop = Stop(request_id=request_id, kind="dropoff",
                        node_idx=req["dropoff_node_idx"], eta_minutes=0)
    new_route = (route[:pickup_pos] + [pickup_stop] +
                 route[pickup_pos:dropoff_pos-1] + [dropoff_stop] +
                 route[dropoff_pos-1:])

    # Simulate the route from current time, depot start
    depot_idx = state.task["depots"][0]["node_idx"]
    capacity = veh["capacity_seats"]
    load = 0
    t = float(state.current_time)
    prev_node = depot_idx if not route else route[-1].node_idx

    # Need to start from depot if route is empty AND vehicle hasn't moved
    if not route:
        prev_node = depot_idx

    total_dist_km = 0.0
    for stop in new_route:
        # travel from prev_node to stop.node_idx
        t += get_edge_duration(state, prev_node, stop.node_idx)
        total_dist_km += get_edge_distance(state, prev_node, stop.node_idx)
        r = state.request(stop.request_id)
        if stop.kind == "pickup":
            if t > r["latest_pickup"]:
                return False, f"misses pickup window for "\
                              f"{stop.request_id} (t={t:.0f} > "\
                              f"latest={r['latest_pickup']})", 0.0
            t = max(t, r["earliest_pickup"]) + r.get("service_time", 0)
            load += r["passengers"]
            if load > capacity:
                return False, f"capacity exceeded ({load}>{capacity}) "\
                              f"after picking up {stop.request_id}", 0.0
        else:
            if t > r["latest_dropoff"]:
                return False, f"misses dropoff window for "\
                              f"{stop.request_id} (t={t:.0f} > "\
                              f"latest={r['latest_dropoff']})", 0.0
            t = max(t, r["earliest_dropoff"]) + r.get("service_time", 0)
            load -= r["passengers"]

        prev_node = stop.node_idx

    # Return to depot, must finish before shift end
    t += get_edge_duration(state, prev_node, depot_idx)
    if t > veh["shift_end"]:
        return False, f"return-to-depot at {t:.0f} > shift_end "\
                      f"{veh['shift_end']}", 0.0

    # Compute baseline cost (current route only)
    baseline_dist = _route_distance(state, vehicle_id, route)
    marginal = total_dist_km - baseline_dist
    return True, "ok", marginal


def _route_distance(state, vehicle_id, route):
    depot_idx = state.task["depots"][0]["node_idx"]
    if not route:
        return 0.0
    total = 0.0
    prev = depot_idx
    for stop in route:
        total += get_edge_distance(state, prev, stop.node_idx)
        prev = stop.node_idx
    total += get_edge_distance(state, prev, depot_idx)
    return total
```

Validation:

```python
# tests/test_feasibility.py
import json
from src.state import EpisodeState
from src.feasibility import check_feasibility

def make_env(fixture):
    task = json.load(open(f"tests/fixtures/{fixture}"))
    s = EpisodeState(task=task); s.initialize()
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
```

### 6.4 Reward function — `src/reward.py`

The single source of truth for all rewards.

```python
# src/reward.py
"""All shaping constants and the terminal reward live here.

Per-step rewards are paid by the tool that generated the action.
Per-completion rewards are paid by tick() when a dropoff stop is realized.
Terminal reward is paid when the episode ends (submit_plan or horizon)."""

# ─── Per-step shaping (continuous feedback) ───────────────────────────────
SHAPE_VALID_ASSIGN     =  0.05    # × (1 - marginal/max_marginal)
SHAPE_INVALID_ACTION   = -0.02
SHAPE_DUPLICATE        = -0.05
SHAPE_REASSIGN_BENEFIT =  0.03    # if cost decreased
SHAPE_REASSIGN_NEUTRAL = -0.01
SHAPE_DEFER            =  0.0
SHAPE_CANCEL_PER_PRIO  = -0.10
SHAPE_ADD_VEHICLE      = -0.50
SHAPE_SWAP_BROKEN      =  0.20
SHAPE_QUERY_SPAM       = -0.001
SHAPE_PER_ACTION_DECAY = -0.005   # ⭐ trajectory-length penalty (incentivize speed)

# ─── Per-completion (paid at dropoff) ─────────────────────────────────────
COMPLETION_BASE = 0.10

def completion_reward(request, on_time: bool) -> float:
    base = COMPLETION_BASE * request["priority"]
    return base if on_time else 0.5 * base


# ─── Terminal ─────────────────────────────────────────────────────────────
def terminal_reward(state, task) -> float:
    n_total = len(task["requests"])
    n_served = len(state.served_requests)
    coverage = n_served / max(1, n_total)

    # Cost-efficiency vs OR-Tools baseline
    agent_km = state.realized_cost_km
    optimal_km = task.get("or_tools_baseline_cost", float("inf"))
    if agent_km <= 0 or optimal_km == float("inf") or optimal_km <= 0:
        cost_ratio = 0.0
    else:
        cost_ratio = min(1.0, optimal_km / agent_km)

    # Speed bonus: finishing in fewer than 5 actions per request earns up to +0.2
    budget = max(20, 5 * n_total)
    speed_bonus = max(0.0, 0.2 * (1 - state.total_actions / budget))

    if n_served == 0:
        return -1.0

    coverage_term = 1.0 * coverage
    efficiency_term = 0.5 * cost_ratio
    return coverage_term + efficiency_term + speed_bonus
```

Tests:

```python
# tests/test_reward.py
from src.reward import (terminal_reward, completion_reward,
                        SHAPE_PER_ACTION_DECAY)

class _S: pass

def test_completion_priority_scaling():
    r1 = completion_reward({"priority": 1}, True)
    r5 = completion_reward({"priority": 5}, True)
    assert r5 == 5 * r1

def test_completion_late_partial():
    r_on  = completion_reward({"priority": 3}, True)
    r_off = completion_reward({"priority": 3}, False)
    assert r_off == 0.5 * r_on

def test_terminal_zero_served_negative():
    s = _S(); s.served_requests = set(); s.realized_cost_km = 0
    s.total_actions = 5
    task = {"requests": [{}, {}], "or_tools_baseline_cost": 100}
    assert terminal_reward(s, task) == -1.0

def test_terminal_full_coverage_positive():
    s = _S(); s.served_requests = {"r-0", "r-1"}
    s.realized_cost_km = 100; s.total_actions = 8
    task = {"requests": [{"id": "r-0"}, {"id": "r-1"}],
            "or_tools_baseline_cost": 100}
    r = terminal_reward(s, task)
    assert r > 1.0  # coverage 1 + efficiency 0.5 + speed_bonus
```


### 6.5 Time engine — `src/time_engine.py`

The heart of dynamic re-optimization. `apply_tick(minutes)` does:

1. Advance simulated time
2. Process completed stops (pickups, dropoffs)
3. Reveal traffic events with `t_reveal ≤ current_time`
4. Apply weather (already in `feasibility.get_edge_duration`)
5. Apply dynamic events: breakdowns, new requests, capacity drops
6. Recompute ETAs for all in-progress routes given new traffic/weather

```python
# src/time_engine.py
from .state import EpisodeState, Stop
from .reward import completion_reward
from .feasibility import (get_edge_duration, get_edge_distance,
                          _route_distance)


def recompute_etas(state: EpisodeState):
    """Walk each vehicle's remaining route from current_time + current
    location, updating ETA on each stop."""
    depot_idx = state.task["depots"][0]["node_idx"]
    for vid, route in state.routes.items():
        if state.vehicle_status[vid] in ("broken", "inactive"):
            continue
        if not route:
            continue
        prev_node = depot_idx  # for now we assume vehicle returns to depot
                               # between routes. Realistic enough for a hackathon.
        t = float(state.current_time)
        for stop in route:
            if stop.completed:
                continue
            t += get_edge_duration(state, prev_node, stop.node_idx)
            r = state.request(stop.request_id)
            if stop.kind == "pickup":
                t = max(t, r["earliest_pickup"]) + r.get("service_time", 0)
            else:
                t = max(t, r["earliest_dropoff"]) + r.get("service_time", 0)
            stop.eta_minutes = t
            prev_node = stop.node_idx


def handle_breakdown(state: EpisodeState, vehicle_id: str):
    state.vehicle_status[vehicle_id] = "broken"
    # All pending stops on this vehicle's route → unassigned
    for stop in state.routes[vehicle_id]:
        if not stop.completed:
            rid = stop.request_id
            # Only return to pending if not yet picked up
            if stop.kind == "pickup":
                state.request_status[rid] = "pending"
                state.request_assigned_to.pop(rid, None)
    state.routes[vehicle_id] = []


def handle_capacity_overflow(state: EpisodeState, vehicle_id: str):
    """If new capacity is less than current load on the route, bump
    the lowest-priority requests back to pending until under capacity."""
    veh = state.vehicle(vehicle_id)
    new_cap = veh["capacity_seats"]
    # Find peak load on the route
    peak_load = 0
    load = 0
    affected = []
    for stop in state.routes[vehicle_id]:
        if stop.completed: continue
        r = state.request(stop.request_id)
        if stop.kind == "pickup":
            load += r["passengers"]
            affected.append((stop, r))
        else:
            load -= r["passengers"]
        peak_load = max(peak_load, load)

    if peak_load <= new_cap:
        return

    # Bump requests in priority order (low priority first)
    affected.sort(key=lambda x: x[1]["priority"])
    for stop, r in affected:
        if peak_load <= new_cap: break
        # Remove pickup + dropoff for this request
        state.routes[vehicle_id] = [s for s in state.routes[vehicle_id]
                                    if s.request_id != r["id"]]
        state.request_status[r["id"]] = "pending"
        state.request_assigned_to.pop(r["id"], None)
        peak_load -= r["passengers"]


def apply_tick(state: EpisodeState, minutes: int) -> tuple[float, list[str]]:
    """Advance simulated time by `minutes`. Returns (delta_reward, log)."""
    t_old = state.current_time
    t_new = state.current_time + minutes
    state.current_time = t_new
    delta_reward = 0.0
    log = []

    # ─── (1) Process completed stops ───────────────────────────────────
    for vid, route in list(state.routes.items()):
        new_route = []
        for stop in route:
            if stop.completed:
                continue
            if stop.eta_minutes <= t_new:
                stop.completed = True
                r = state.request(stop.request_id)
                if stop.kind == "pickup":
                    state.request_status[r["id"]] = "in_vehicle"
                    log.append(f"t={int(stop.eta_minutes)}: picked up {r['id']}")
                else:
                    on_time = stop.eta_minutes <= r["latest_dropoff"]
                    rew = completion_reward(r, on_time)
                    delta_reward += rew
                    state.served_requests.add(r["id"])
                    state.request_status[r["id"]] = "completed"
                    log.append(f"t={int(stop.eta_minutes)}: delivered "
                               f"{r['id']} (+{rew:.3f}, "
                               f"{'on time' if on_time else 'late'})")
            else:
                new_route.append(stop)
        state.routes[vid] = new_route

        # Update realized_cost_km incrementally (sum of completed-leg distances)
        # We approximate by recomputing total traversed.
    state.realized_cost_km = _compute_realized_cost(state, t_new)

    # ─── (2) Reveal traffic events ─────────────────────────────────────
    for ev in state.task["traffic_events"]:
        if t_old < ev["t_reveal"] <= t_new:
            state.revealed_traffic[(ev["node_a"], ev["node_b"])] = ev["speed_factor"]
            state.revealed_traffic[(ev["node_b"], ev["node_a"])] = ev["speed_factor"]
            log.append(f"⚠ t={ev['t_reveal']}: {ev['reason']}")

    # ─── (3) Dynamic events ────────────────────────────────────────────
    for ev in state.task["dynamic_events"]:
        if t_old < ev["t"] <= t_new:
            if ev["type"] == "vehicle_breakdown":
                vid = ev["vehicle_id"]
                handle_breakdown(state, vid)
                log.append(f"⚠ t={ev['t']}: BREAKDOWN of {vid}")
            elif ev["type"] == "new_request":
                rid = ev["request_id"]
                if rid in state.request_status:
                    state.request_status[rid] = "pending"
                    log.append(f"NEW request released: {rid}")
            elif ev["type"] == "capacity_drop":
                vid = ev["vehicle_id"]
                state.vehicle_capacity_override[vid] = {
                    "capacity_seats": ev["new_capacity_seats"]
                }
                handle_capacity_overflow(state, vid)
                log.append(f"⚠ t={ev['t']}: capacity drop {vid} → "
                           f"{ev['new_capacity_seats']} seats")

    # ─── (4) Vehicle shifts ────────────────────────────────────────────
    for v in state.task["vehicles"]:
        if (state.current_time >= v["shift_end"] and
                state.vehicle_status[v["id"]] == "available"):
            state.vehicle_status[v["id"]] = "inactive"
            log.append(f"Shift ended: {v['id']}")

    # ─── (5) Recompute ETAs ────────────────────────────────────────────
    recompute_etas(state)

    return delta_reward, log


def _compute_realized_cost(state: EpisodeState, t_now: int) -> float:
    """Sum distance traversed so far across all vehicles."""
    depot_idx = state.task["depots"][0]["node_idx"]
    total = 0.0
    for vid, route in state.routes.items():
        # Walk from depot through completed stops
        prev = depot_idx
        for stop in route:
            if stop.completed:
                total += get_edge_distance(state, prev, stop.node_idx)
                prev = stop.node_idx
    return total
```

Tests:

```python
# tests/test_time_engine.py
import json
from src.state import EpisodeState
from src.time_engine import apply_tick, recompute_etas, handle_breakdown
from src.feasibility import check_feasibility

def make_env(fixture):
    task = json.load(open(f"tests/fixtures/{fixture}"))
    s = EpisodeState(task=task); s.initialize()
    return s

def test_tick_reveals_traffic():
    s = make_env("traffic_at_60.json")
    apply_tick(s, 30)
    assert s.revealed_traffic == {}
    apply_tick(s, 40)  # now t=70 > 60
    assert s.revealed_traffic, s.revealed_traffic

def test_tick_processes_completion():
    s = make_env("trivial_task.json")
    # Manually assign
    from src.state import Stop
    pu = Stop(request_id="r-0", kind="pickup",
              node_idx=s.task["requests"][0]["pickup_node_idx"], eta_minutes=20)
    do = Stop(request_id="r-0", kind="dropoff",
              node_idx=s.task["requests"][0]["dropoff_node_idx"], eta_minutes=40)
    s.routes["v-0"] = [pu, do]
    s.request_status["r-0"] = "assigned"
    delta, log = apply_tick(s, 50)
    assert "r-0" in s.served_requests
    assert delta > 0

def test_breakdown_returns_to_pending():
    s = make_env("trivial_task.json")
    from src.state import Stop
    pu = Stop(request_id="r-0", kind="pickup",
              node_idx=1, eta_minutes=200)
    do = Stop(request_id="r-0", kind="dropoff",
              node_idx=2, eta_minutes=300)
    s.routes["v-0"] = [pu, do]
    s.request_status["r-0"] = "assigned"
    handle_breakdown(s, "v-0")
    assert s.vehicle_status["v-0"] == "broken"
    assert s.request_status["r-0"] == "pending"
    assert s.routes["v-0"] == []

def test_new_request_released():
    s = make_env("late_request_task.json")
    assert s.request_status["r-late-0"] == "unreleased"
    apply_tick(s, 200)
    assert s.request_status["r-late-0"] == "pending"
```

### 6.6 The ORS server — `src/server.py`

This is the central entry point. All 14 tools live here.

```python
# src/server.py
import os
from typing import Optional
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field
from openreward.environments import (Environment, JSONObject, Server, Split,
                                     TextBlock, ToolOutput, tool)

from .state import EpisodeState, Stop
from .feasibility import (check_feasibility, get_edge_duration,
                          get_edge_distance, _route_distance)
from .reward import (terminal_reward, SHAPE_VALID_ASSIGN, SHAPE_INVALID_ACTION,
                     SHAPE_PER_ACTION_DECAY, SHAPE_REASSIGN_BENEFIT,
                     SHAPE_REASSIGN_NEUTRAL, SHAPE_CANCEL_PER_PRIO,
                     SHAPE_ADD_VEHICLE, SHAPE_SWAP_BROKEN, SHAPE_QUERY_SPAM,
                     SHAPE_DEFER)
from .time_engine import apply_tick, recompute_etas


# ─── Task loading ─────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("ORWD_DATA_DIR", "/orwd_data")
_TASKS_PATH = Path(DATA_DIR) / "tasks.parquet"
if _TASKS_PATH.exists():
    ALL_TASKS = pd.read_parquet(_TASKS_PATH).to_dict(orient="records")
else:
    ALL_TASKS = []  # tests can monkeypatch
ALL_TASKS.sort(key=lambda t: (t["split"], t["id"]))


# ─── Tool param schemas ───────────────────────────────────────────────────
class AssignParams(BaseModel):
    request_id: str
    vehicle_id: str
    pickup_position: int = Field(..., ge=0)
    dropoff_position: int = Field(..., ge=1)

class ReassignParams(BaseModel):
    request_id: str
    new_vehicle_id: str
    pickup_position: int = Field(..., ge=0)
    dropoff_position: int = Field(..., ge=1)

class DeferParams(BaseModel):
    request_id: str
    until_minutes: int = Field(..., ge=0)

class CancelParams(BaseModel):
    request_id: str

class AddVehicleParams(BaseModel):
    type: str  # one of: minibus_16 | accessible_van_8 | minibus_24 | small_van_4

class SwapParams(BaseModel):
    vehicle_id_a: str
    vehicle_id_b: str

class QueryEdgeParams(BaseModel):
    node_a: int
    node_b: int

class QueryWeatherParams(BaseModel):
    at_minute: int = Field(..., ge=0)

class QueryEtaParams(BaseModel):
    node_a: int
    node_b: int

class TickParams(BaseModel):
    minutes: int = Field(..., ge=1, le=480)

class EmptyParams(BaseModel):
    pass


# ─── The Environment ──────────────────────────────────────────────────────
class LondonDynamicRouting(Environment):
    """Dynamic, multi-horizon, weather/traffic-aware vehicle routing on
    the real London road network."""

    def __init__(self, task_spec: JSONObject = None, secrets: dict = None):
        super().__init__(task_spec or {})
        if not task_spec:
            return
        self.task = task_spec
        self.state = EpisodeState(task=task_spec)
        self.state.initialize()
        # Pre-compute max marginal for normalization
        max_d = max(max(row) for row in task_spec["distance_matrix_km"])
        self.max_marginal_km = max(1.0, max_d * 2)
        self.add_vehicle_used = False
        self.action_budget = max(40, 8 * len(task_spec["requests"]))

    # ─── ORS metadata ────────────────────────────────────────────────
    @classmethod
    def list_splits(cls):
        return [Split(name="tutorial", type="train"),
                Split(name="train",    type="train"),
                Split(name="test",     type="test")]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        return [t for t in ALL_TASKS if t["split"] == split]

    def get_prompt(self) -> list[TextBlock]:
        t = self.task
        n_initial = sum(1 for r in t["requests"] if r.get("released_at", 0) == 0)
        text = (
            f"You are the dispatcher for a heterogeneous fleet operating in "
            f"London Zones 1–4 on {t['episode_date']}.\n\n"
            f"OPERATIONAL DAY: {t['horizon_minutes']} minutes (starting "
            f"06:00). Decision tick: {t['tick_minutes']} min.\n\n"
            f"FLEET ({len(t['vehicles'])} vehicles):\n"
        )
        for v in t["vehicles"]:
            text += (f"  • {v['id']} ({v['type']}): "
                     f"{v['capacity_seats']} seats, "
                     f"{v['capacity_wheelchair']} wheelchair, "
                     f"shift [{v['shift_start']},{v['shift_end']}]\n")
        text += (f"\nTotal requests today: {len(t['requests'])} "
                 f"({n_initial} pending at t=0, "
                 f"{len(t['requests'])-n_initial} released later).\n\n"
                 f"INITIAL PENDING REQUESTS (first 5 shown; use "
                 f"`list_pending_requests` for all):\n")
        for r in [r for r in t["requests"]
                  if r.get("released_at", 0) == 0][:5]:
            text += (f"  • {r['id']}: {r['passengers']}p "
                     f"node {r['pickup_node_idx']}→{r['dropoff_node_idx']} "
                     f"pickup window [{r['earliest_pickup']},"
                     f"{r['latest_pickup']}] priority={r['priority']}\n")
        text += (
            "\nMore requests will be revealed during the day. Vehicles may "
            "break down. Traffic and weather will change.\n\n"
            "GOAL: serve as many requests as possible, on time, with minimum "
            "total kilometres driven.\n\n"
            "WORKFLOW:\n"
            "  1. Inspect with `list_pending_requests` and `get_state`.\n"
            "  2. Assign with `assign(request_id, vehicle_id, pickup_position, "
            "dropoff_position)`. Positions 0=start, 1=after first stop, etc. "
            "Pickup position must be < dropoff position.\n"
            "  3. Advance time with `tick(minutes)`. When you tick, completed "
            "deliveries pay reward and new events may fire.\n"
            "  4. React to events with `reassign`, `swap_vehicles`, `cancel`.\n"
            "  5. Submit with `submit_plan` when done or near horizon.\n\n"
            "Each action carries a small efficiency penalty (~0.005). "
            "Avoid spamming queries.")
        return [TextBlock(type="text", text=text)]

    # ─── Helpers ─────────────────────────────────────────────────────
    def _shape(self, base: float) -> float:
        self.state.total_actions += 1
        return base + SHAPE_PER_ACTION_DECAY

    def _check_terminal(self) -> Optional[ToolOutput]:
        if self.state.current_time >= self.task["horizon_minutes"]:
            return self._finalize("Horizon reached")
        if self.state.total_actions >= self.action_budget:
            return self._finalize("Action budget exhausted")
        return None

    def _finalize(self, reason: str) -> ToolOutput:
        # Auto-complete in-flight stops by ticking to end of horizon
        remaining = max(0, self.task["horizon_minutes"] - self.state.current_time)
        if remaining > 0:
            apply_tick(self.state, remaining)
        r = terminal_reward(self.state, self.task)
        served = len(self.state.served_requests)
        total = len(self.task["requests"])
        msg = (f"EPISODE END ({reason}). Served {served}/{total} requests. "
               f"Realized cost: {self.state.realized_cost_km:.1f} km. "
               f"OR-Tools baseline: {self.task['or_tools_baseline_cost']:.1f} km. "
               f"Total actions: {self.state.total_actions}. "
               f"Terminal reward: {r:+.3f}")
        return ToolOutput(blocks=[TextBlock(type="text", text=msg)],
                          reward=r, finished=True)

    # ─── Tools ───────────────────────────────────────────────────────
    @tool
    def assign(self, params: AssignParams) -> ToolOutput:
        """Insert pickup and dropoff stops into a vehicle's route at the
        given positions. Both positions are 0-indexed; pickup must be
        before dropoff. Returns reward proportional to insertion efficiency."""
        ok, reason, marg = check_feasibility(
            self.state, params.request_id, params.vehicle_id,
            params.pickup_position, params.dropoff_position)
        if not ok:
            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"INVALID: {reason}")],
                reward=self._shape(SHAPE_INVALID_ACTION), finished=False)
        # Apply
        req = self.state.request(params.request_id)
        pu = Stop(request_id=req["id"], kind="pickup",
                  node_idx=req["pickup_node_idx"], eta_minutes=0)
        do = Stop(request_id=req["id"], kind="dropoff",
                  node_idx=req["dropoff_node_idx"], eta_minutes=0)
        route = self.state.routes[params.vehicle_id]
        route.insert(params.pickup_position, pu)
        route.insert(params.dropoff_position, do)
        self.state.request_status[req["id"]] = "assigned"
        self.state.request_assigned_to[req["id"]] = params.vehicle_id
        recompute_etas(self.state)

        shaping = SHAPE_VALID_ASSIGN * max(0, 1 - marg / self.max_marginal_km)
        return ToolOutput(
            blocks=[TextBlock(type="text",
                text=f"Assigned {req['id']} → {params.vehicle_id}. "
                     f"Marginal: {marg:.2f} km. Pickup ETA: "
                     f"{pu.eta_minutes:.0f}, Dropoff ETA: {do.eta_minutes:.0f}")],
            reward=self._shape(shaping), finished=False)

    @tool
    def reassign(self, params: ReassignParams) -> ToolOutput:
        """Move an already-assigned request to a different vehicle.
        Only allowed if pickup hasn't started yet."""
        rid = params.request_id
        if not self.state.is_assigned(rid):
            return ToolOutput(
                blocks=[TextBlock(type="text",
                    text=f"INVALID: {rid} is not assigned")],
                reward=self._shape(SHAPE_INVALID_ACTION), finished=False)
        if self.state.request_status[rid] == "in_vehicle":
            return ToolOutput(
                blocks=[TextBlock(type="text",
                    text=f"INVALID: {rid} already in vehicle")],
                reward=self._shape(SHAPE_INVALID_ACTION), finished=False)
        # Save state, remove from old vehicle, attempt new assignment
        old_vid = self.state.request_assigned_to[rid]
        old_route = self.state.routes[old_vid][:]
        self.state.routes[old_vid] = [s for s in old_route if s.request_id != rid]
        self.state.request_status[rid] = "pending"
        self.state.request_assigned_to.pop(rid, None)

        old_cost = _route_distance(self.state, old_vid, old_route)
        ok, reason, marg = check_feasibility(
            self.state, rid, params.new_vehicle_id,
            params.pickup_position, params.dropoff_position)
        if not ok:
            # Rollback
            self.state.routes[old_vid] = old_route
            self.state.request_status[rid] = "assigned"
            self.state.request_assigned_to[rid] = old_vid
            return ToolOutput(
                blocks=[TextBlock(type="text",
                    text=f"INVALID reassign: {reason} (rolled back)")],
                reward=self._shape(SHAPE_INVALID_ACTION), finished=False)
        # Apply on new vehicle
        req = self.state.request(rid)
        pu = Stop(request_id=rid, kind="pickup",
                  node_idx=req["pickup_node_idx"], eta_minutes=0)
        do = Stop(request_id=rid, kind="dropoff",
                  node_idx=req["dropoff_node_idx"], eta_minutes=0)
        self.state.routes[params.new_vehicle_id].insert(params.pickup_position, pu)
        self.state.routes[params.new_vehicle_id].insert(params.dropoff_position, do)
        self.state.request_status[rid] = "assigned"
        self.state.request_assigned_to[rid] = params.new_vehicle_id
        recompute_etas(self.state)

        new_old_cost = _route_distance(self.state, old_vid,
                                       self.state.routes[old_vid])
        cost_saved = old_cost - new_old_cost - marg
        shaping = (SHAPE_REASSIGN_BENEFIT if cost_saved > 0
                   else SHAPE_REASSIGN_NEUTRAL)
        return ToolOutput(
            blocks=[TextBlock(type="text",
                text=f"Reassigned {rid}: {old_vid}→{params.new_vehicle_id}. "
                     f"Net cost change: {-cost_saved:+.2f} km")],
            reward=self._shape(shaping), finished=False)

    @tool
    def defer(self, params: DeferParams) -> ToolOutput:
        """Postpone a pending request to a later decision tick. The request
        stays in the deferred state until `until_minutes` then returns to
        pending."""
        rid = params.request_id
        if self.state.request_status.get(rid) != "pending":
            return ToolOutput(
                blocks=[TextBlock(type="text",
                    text=f"INVALID: {rid} is not pending")],
                reward=self._shape(SHAPE_INVALID_ACTION), finished=False)
        self.state.request_status[rid] = "deferred"
        # Schedule release: append to dynamic_events as a synthetic new_request
        self.task["dynamic_events"].append({
            "t": params.until_minutes, "type": "new_request",
            "request_id": rid})
        self.task["dynamic_events"].sort(key=lambda e: e["t"])
        return ToolOutput(
            blocks=[TextBlock(type="text",
                text=f"Deferred {rid} until t={params.until_minutes}")],
            reward=self._shape(SHAPE_DEFER), finished=False)

    @tool
    def cancel(self, params: CancelParams) -> ToolOutput:
        """Refuse a request entirely. Penalty proportional to priority."""
        rid = params.request_id
        if self.state.request_status.get(rid) in ("completed", "cancelled"):
            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"INVALID: {rid} not active")],
                reward=self._shape(SHAPE_INVALID_ACTION), finished=False)
        # Remove from any vehicle route
        for vid in list(self.state.routes.keys()):
            self.state.routes[vid] = [s for s in self.state.routes[vid]
                                      if s.request_id != rid]
        req = self.state.request(rid)
        self.state.request_status[rid] = "cancelled"
        self.state.request_assigned_to.pop(rid, None)
        return ToolOutput(
            blocks=[TextBlock(type="text", text=f"Cancelled {rid}")],
            reward=self._shape(SHAPE_CANCEL_PER_PRIO * req["priority"]),
            finished=False)

    @tool
    def add_vehicle(self, params: AddVehicleParams) -> ToolOutput:
        """Spin up a reserve vehicle (one-time per task, large activation cost)."""
        if self.add_vehicle_used:
            return ToolOutput(
                blocks=[TextBlock(type="text",
                    text="INVALID: add_vehicle already used this episode")],
                reward=self._shape(SHAPE_INVALID_ACTION), finished=False)
        from scripts.generate_tasks import VEHICLE_TYPES
        template = next((t for t in VEHICLE_TYPES if t["type"] == params.type), None)
        if not template:
            return ToolOutput(
                blocks=[TextBlock(type="text",
                    text=f"INVALID: unknown vehicle type {params.type}")],
                reward=self._shape(SHAPE_INVALID_ACTION), finished=False)
        new_vid = f"v-add-{len(self.task['vehicles'])}"
        new_vehicle = {
            "id": new_vid, "type": template["type"], "depot_id": "depot-0",
            "capacity_seats": template["capacity_seats"],
            "capacity_wheelchair": template["capacity_wheelchair"],
            "speed_factor": template["speed_factor"],
            "cost_per_km": template["cost_per_km"],
            "shift_start": self.state.current_time,
            "shift_end": self.task["horizon_minutes"],
        }
        self.task["vehicles"].append(new_vehicle)
        self.state.routes[new_vid] = []
        self.state.vehicle_status[new_vid] = "available"
        self.add_vehicle_used = True
        return ToolOutput(
            blocks=[TextBlock(type="text",
                text=f"Activated {new_vid} ({params.type}). "
                     f"Available from t={self.state.current_time}.")],
            reward=self._shape(SHAPE_ADD_VEHICLE), finished=False)

    @tool
    def swap_vehicles(self, params: SwapParams) -> ToolOutput:
        """Transfer remaining route from vehicle_a to vehicle_b. Useful after
        vehicle_a breakdown or shift end. Validates b can do the route."""
        a, b = params.vehicle_id_a, params.vehicle_id_b
        if a not in self.state.routes or b not in self.state.routes:
            return ToolOutput(
                blocks=[TextBlock(type="text", text="INVALID: unknown vehicle")],
                reward=self._shape(SHAPE_INVALID_ACTION), finished=False)
        if self.state.vehicle_status[b] in ("broken", "inactive"):
            return ToolOutput(
                blocks=[TextBlock(type="text",
                    text=f"INVALID: target {b} not available")],
                reward=self._shape(SHAPE_INVALID_ACTION), finished=False)
        a_broken = self.state.vehicle_status[a] == "broken"
        # Move all unassigned request_ids from a's old route back to pending,
        # and try to greedily reassign onto b
        moved = []
        for stop in self.state.routes[a]:
            if stop.kind == "pickup" and not stop.completed:
                rid = stop.request_id
                self.state.request_status[rid] = "pending"
                self.state.request_assigned_to.pop(rid, None)
                moved.append(rid)
        self.state.routes[a] = []
        # Greedy insertion of moved requests into b
        succeeded = []
        for rid in moved:
            n = len(self.state.routes[b])
            ok, _, _ = check_feasibility(self.state, rid, b, n, n + 1)
            if ok:
                req = self.state.request(rid)
                pu = Stop(request_id=rid, kind="pickup",
                          node_idx=req["pickup_node_idx"], eta_minutes=0)
                do = Stop(request_id=rid, kind="dropoff",
                          node_idx=req["dropoff_node_idx"], eta_minutes=0)
                self.state.routes[b].extend([pu, do])
                self.state.request_status[rid] = "assigned"
                self.state.request_assigned_to[rid] = b
                succeeded.append(rid)
        recompute_etas(self.state)
        shaping = SHAPE_SWAP_BROKEN if a_broken else 0.0
        return ToolOutput(
            blocks=[TextBlock(type="text",
                text=f"Swap {a}→{b}: moved {len(moved)} requests, "
                     f"successfully reassigned {len(succeeded)}")],
            reward=self._shape(shaping), finished=False)

    @tool
    def query_traffic(self, params: QueryEdgeParams) -> ToolOutput:
        """Get currently-revealed speed factor for an edge. Default 1.0."""
        sf = self.state.revealed_traffic.get((params.node_a, params.node_b), 1.0)
        return ToolOutput(
            blocks=[TextBlock(type="text",
                text=f"Edge {params.node_a}→{params.node_b}: "
                     f"speed_factor={sf:.2f}")],
            reward=self._shape(SHAPE_QUERY_SPAM), finished=False)

    @tool
    def query_weather(self, params: QueryWeatherParams) -> ToolOutput:
        """Get weather conditions at a future timestep."""
        from scripts.fetch_weather import weather_at, weather_speed_factor
        w = weather_at(self.task["weather_timeline"], params.at_minute)
        sf = weather_speed_factor(w)
        return ToolOutput(
            blocks=[TextBlock(type="text",
                text=f"At t={params.at_minute}: precip={w['precip_mm']}mm, "
                     f"vis={w['visibility_km']:.1f}km, "
                     f"speed factor={sf:.2f}")],
            reward=self._shape(SHAPE_QUERY_SPAM), finished=False)

    @tool
    def get_distance(self, params: QueryEdgeParams) -> ToolOutput:
        """Lookup nominal distance and duration between two nodes."""
        d = self.task["distance_matrix_km"][params.node_a][params.node_b]
        t = self.task["duration_matrix_min"][params.node_a][params.node_b]
        return ToolOutput(
            blocks=[TextBlock(type="text",
                text=f"{params.node_a}→{params.node_b}: "
                     f"{d:.2f} km, {t:.1f} min nominal")],
            reward=self._shape(0.0), finished=False)

    @tool
    def get_eta(self, params: QueryEtaParams) -> ToolOutput:
        """Adjusted duration accounting for revealed traffic + current weather."""
        adj = get_edge_duration(self.state, params.node_a, params.node_b)
        return ToolOutput(
            blocks=[TextBlock(type="text",
                text=f"{params.node_a}→{params.node_b}: "
                     f"{adj:.1f} min (with traffic & weather)")],
            reward=self._shape(0.0), finished=False)

    @tool
    def get_state(self, params: EmptyParams) -> ToolOutput:
        """Compact dump of routes, vehicle status, request status."""
        s = self.state
        lines = [f"t={s.current_time}/{self.task['horizon_minutes']}",
                 f"served={len(s.served_requests)}/"
                 f"{len(self.task['requests'])}",
                 f"realized_cost={s.realized_cost_km:.1f} km", "Vehicles:"]
        for vid, status in s.vehicle_status.items():
            route_summary = " → ".join(
                f"{stop.kind[0]}({stop.request_id})" for stop in s.routes[vid])
            lines.append(f"  {vid} [{status}]: [{route_summary}]")
        n_pending = sum(1 for st in s.request_status.values() if st == "pending")
        lines.append(f"Pending: {n_pending}")
        return ToolOutput(
            blocks=[TextBlock(type="text", text="\n".join(lines))],
            reward=self._shape(0.0), finished=False)

    @tool
    def list_pending_requests(self, params: EmptyParams) -> ToolOutput:
        """All currently revealed, unassigned requests."""
        s = self.state
        pending = [self.state.request(rid)
                   for rid in s.pending_request_ids()]
        if not pending:
            text = "No pending requests."
        else:
            lines = []
            for r in pending:
                lines.append(
                    f"{r['id']}: {r['passengers']}p "
                    f"node {r['pickup_node_idx']}→{r['dropoff_node_idx']} "
                    f"window pickup [{r['earliest_pickup']},"
                    f"{r['latest_pickup']}] "
                    f"dropoff [{r['earliest_dropoff']},"
                    f"{r['latest_dropoff']}] "
                    f"prio={r['priority']}"
                    + (f" wheelchair={r['wheelchairs']}"
                       if r.get("wheelchairs") else ""))
            text = f"{len(pending)} pending:\n" + "\n".join(lines)
        return ToolOutput(blocks=[TextBlock(type="text", text=text)],
                          reward=self._shape(0.0), finished=False)

    @tool
    def tick(self, params: TickParams) -> ToolOutput:
        """Advance simulated time by `minutes`. Reveals new requests, traffic,
        weather effects; processes completed stops; pays completion rewards."""
        delta_reward, log = apply_tick(self.state, params.minutes)
        terminal = self._check_terminal()
        if terminal:
            terminal.reward += delta_reward
            return terminal
        text = (f"t={self.state.current_time}/{self.task['horizon_minutes']}. "
                f"Δreward: {delta_reward:+.3f}.\n" + "\n".join(log[:15]))
        return ToolOutput(blocks=[TextBlock(type="text", text=text)],
                          reward=self._shape(delta_reward), finished=False)

    @tool
    def submit_plan(self, params: EmptyParams) -> ToolOutput:
        """Terminate the episode and grade the final plan."""
        return self._finalize("Submitted by agent")


if __name__ == "__main__":
    Server([LondonDynamicRouting]).run()
```

### 6.7 Tool reference table

| # | Tool | Signature | Reward |
|---|------|-----------|--------|
| 1 | `assign` | `(request_id, vehicle_id, pickup_position, dropoff_position)` | +0.05 × (1 − marg/max). Invalid: −0.02 |
| 2 | `reassign` | `(request_id, new_vehicle_id, pickup_position, dropoff_position)` | +0.03 if cost ↓ else −0.01 |
| 3 | `defer` | `(request_id, until_minutes)` | 0.0 |
| 4 | `cancel` | `(request_id)` | −0.10 × priority |
| 5 | `add_vehicle` | `(type)` | −0.50 (one-time) |
| 6 | `swap_vehicles` | `(vehicle_id_a, vehicle_id_b)` | +0.20 if a is broken else 0 |
| 7 | `query_traffic` | `(node_a, node_b)` | −0.001 |
| 8 | `query_weather` | `(at_minute)` | −0.001 |
| 9 | `get_distance` | `(node_a, node_b)` | 0.0 |
| 10 | `get_eta` | `(node_a, node_b)` | 0.0 |
| 11 | `get_state` | `()` | 0.0 |
| 12 | `list_pending_requests` | `()` | 0.0 |
| 13 | `tick` | `(minutes)` | Σ completion rewards in interval |
| 14 | `submit_plan` | `()` | Terminal reward |

Every action also pays `SHAPE_PER_ACTION_DECAY = -0.005` (the trajectory length penalty).

### 6.8 Daniel's checklist

- [ ] `tests/test_state.py` passes (5 tests)
- [ ] `tests/test_feasibility.py` passes (4 tests + edge cases)
- [ ] `tests/test_reward.py` passes (4 tests)
- [ ] `tests/test_time_engine.py` passes (4 tests)
- [ ] `python -c "from src.server import LondonDynamicRouting; print('ok')"` works
- [ ] `python src/server.py` starts ORS server on :8000
- [ ] All 14 tools callable without crashes (smoke test against trivial fixture)

---

## 7. Stream C — Rollout, infra & deployment (Kayode)

### 7.1 Files owned

```
Dockerfile
requirements.txt
README.md                          # environment card (top-level)
README_dataset.md                  # HF dataset card
scripts/
└── run_rollout.py                 # ⭐ end-to-end rollout against OpenAI
tests/
├── conftest.py
├── fixtures/
│   ├── trivial_task.json          # 3 nodes, 1 vehicle, 1 request
│   ├── medium_task.json           # 15 nodes, 3 vehicles, 8 requests
│   ├── tight_window_task.json     # request with impossible window
│   ├── oversize_request.json      # request bigger than vehicle
│   ├── traffic_at_60.json         # traffic event at t=60
│   ├── breakdown_at_180.json      # vehicle breakdown at t=180
│   └── late_request_task.json     # request released at t=200
├── test_integration.py            # full env smoke
└── test_rollout.py                # rollout script smoke
```

### 7.2 The rollout script — `scripts/run_rollout.py`

This is the demo. It MUST work end-to-end with `OPENAI_API_KEY` and `OPENREWARD_API_KEY` set.

```python
#!/usr/bin/env python3
"""End-to-end rollout against the LondonDynamicRouting environment.

Required env vars:
  OPENREWARD_API_KEY   — for hitting the OpenReward env
  OPENAI_API_KEY       — for executing the agent

Usage:
  # Local server (Daniel's dev loop)
  python scripts/run_rollout.py --local --task-idx 0
  # Deployed env (demo)
  python scripts/run_rollout.py --task-idx 0
  # Multiple tasks, save trace
  python scripts/run_rollout.py --n-tasks 3 --record traces/
"""
import argparse, asyncio, json, os, sys, time
from pathlib import Path

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

DEFAULT_MODEL = os.environ.get("ROLLOUT_MODEL", "gpt-5.4")
ENV_NAMESPACE = os.environ.get("OR_ENV", "EnvCommons/LondonDynamicRouting")
LOCAL_BASE_URL = "http://localhost:8080"
LOCAL_NAME = "LondonDynamicRouting"
MAX_TURNS = 80

SYSTEM_PROMPT = """You are an expert transit fleet dispatcher operating in \
London. You manage a heterogeneous fleet across one operational day. Your goal \
is to serve as many passenger and parcel requests as possible, on time, with \
minimum total kilometres driven, while reacting to traffic, weather, and \
unexpected breakdowns or capacity changes.

Strategy:
  1. Begin by calling `list_pending_requests` and `get_state` to understand \
the situation.
  2. Assign requests to vehicles using `assign(request_id, vehicle_id, \
pickup_position, dropoff_position)`. Pickup position must be strictly less \
than dropoff position. Position 0 means insert at the start of the route.
  3. Prefer cheap insertions (low marginal km) and respect time windows + \
capacity. The reward signal will tell you how good each insertion was.
  4. After assigning a batch of pending requests, call `tick(15)` or \
`tick(30)` to advance time. Many requests are revealed mid-day, so DO NOT \
try to plan everything at t=0.
  5. If a vehicle breaks down (you'll see a ⚠ breakdown event in tick \
output), use `swap_vehicles(broken_id, healthy_id)` to recover. If a \
capacity drops, use `reassign` to move bumped requests.
  6. Be concise. Each action carries a small efficiency penalty (~0.005). \
Avoid spamming queries. Prefer doing useful work to repeated state inspections.
  7. Submit your plan with `submit_plan()` when you've handled everything \
or you're near the horizon (t close to 960)."""


async def run_one_task(task, environment, oai_client, model, max_turns,
                       trace_path=None):
    tools = await environment.list_tools(format="openai")
    OR_KEY = os.environ["OPENREWARD_API_KEY"]
    trace = {"task_id": task.task_spec["id"],
             "difficulty": task.task_spec["difficulty"],
             "split": task.task_spec.get("split", "?"),
             "actions": [], "rewards": [], "model": model}

    async with environment.session(task=task,
                                   secrets={"api_key": OR_KEY}) as session:
        prompt = await session.get_prompt()
        # OpenAI Responses API: send system + user, accumulate output items
        input_list = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt[0].text},
        ]
        total_reward = 0.0
        finished = False

        for turn in range(max_turns):
            try:
                response = await oai_client.responses.create(
                    model=model, tools=tools, input=input_list)
            except Exception as e:
                print(f"  [turn {turn}] OpenAI API error: {e}")
                trace["actions"].append({"name": "_api_error",
                                         "error": str(e)})
                break

            input_list += response.output

            had_tool_call = False
            for item in response.output:
                if item.type != "function_call":
                    continue
                had_tool_call = True
                t0 = time.time()
                try:
                    args = json.loads(str(item.arguments))
                except Exception:
                    args = {}
                try:
                    tr = await session.call_tool(item.name, args)
                    text = "".join(b.text for b in (tr.blocks or [])
                                   if hasattr(b, "text"))
                    reward = tr.reward or 0.0
                    finished = bool(tr.finished)
                    total_reward += reward
                    trace["actions"].append({
                        "turn": turn, "name": item.name, "args": args,
                        "duration_s": round(time.time() - t0, 3),
                        "text_preview": text[:200]})
                    trace["rewards"].append(reward)
                    print(f"  [t={turn:02d}] {item.name}({_short(args)}) "
                          f"→ r={reward:+.3f}, finished={finished}, "
                          f"cum={total_reward:+.3f}")
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id, "output": text})
                    if finished:
                        break
                except Exception as e:
                    err = f"Tool failed: {type(e).__name__}: {e}"
                    print(f"  [t={turn:02d}] {item.name} ERROR: {e}")
                    trace["actions"].append({"turn": turn, "name": item.name,
                                             "args": args, "error": str(e)})
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": f"Tool error: {err}"})

            if finished: break
            if not had_tool_call:
                # Model produced text only without ending — nudge by force-submit
                print(f"  [t={turn:02d}] Model returned no tool call; "
                      f"forcing submit_plan.")
                tr = await session.call_tool("submit_plan", {})
                total_reward += tr.reward or 0.0
                trace["actions"].append({"turn": turn, "name": "submit_plan",
                                         "args": {}, "forced": True})
                trace["rewards"].append(tr.reward or 0.0)
                finished = True
                break

        if not finished:
            print(f"  Hit max_turns={max_turns} without termination.")

    trace["total_reward"] = total_reward
    trace["finished"] = finished
    trace["n_actions"] = len(trace["actions"])
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        Path(trace_path).write_text(json.dumps(trace, indent=2, default=str))
        print(f"  Trace saved → {trace_path}")
    return total_reward, finished, trace


def _short(args):
    if not args: return ""
    return ", ".join(f"{k}={v}" for k, v in list(args.items())[:3])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="tutorial",
                    choices=["tutorial", "train", "test"])
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--local", action="store_true",
                    help="hit localhost:8080 instead of deployed env")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--record", default=None,
                    help="directory to write trace JSONs")
    args = ap.parse_args()

    if not os.environ.get("OPENREWARD_API_KEY"):
        print("ERROR: OPENREWARD_API_KEY not set", file=sys.stderr)
        return 2
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        return 2

    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI()

    if args.local:
        environment = or_client.environments.get(name=LOCAL_NAME,
                                                 base_url=LOCAL_BASE_URL)
    else:
        environment = or_client.environments.get(name=ENV_NAMESPACE)

    tasks = await environment.list_tasks(split=args.split)
    print(f"Got {len(tasks)} tasks in split '{args.split}'.")
    selected = tasks[args.task_idx : args.task_idx + args.n_tasks]
    if not selected:
        print(f"No tasks at index {args.task_idx}")
        return 1

    results = []
    for i, task in enumerate(selected):
        print(f"\n=== Task {args.task_idx + i}: {task.task_spec['id']} "
              f"(difficulty={task.task_spec['difficulty']}) ===")
        trace_path = None
        if args.record:
            trace_path = (f"{args.record.rstrip('/')}/"
                          f"{task.task_spec['id']}.json")
        r, f, _ = await run_one_task(task, environment, oai_client,
                                     args.model, args.max_turns, trace_path)
        results.append((task.task_spec["id"],
                        task.task_spec["difficulty"], r, f))
        print(f"=== Total reward: {r:+.3f} | finished: {f} ===")

    print("\n--- SUMMARY ---")
    for tid, d, r, f in results:
        print(f"  {tid} (d={d}): reward={r:+.3f} finished={f}")
    avg = sum(r for _, _, r, _ in results) / len(results)
    print(f"\nMean reward across {len(results)} tasks: {avg:+.3f}")
    return 0 if all(f for _, _, _, f in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

### 7.3 Test fixtures

These are checked-in JSON files matching the `TaskSpec` schema. Kayode writes them by hand in the first 20 minutes so Daniel's tests have something to load.

**`tests/fixtures/trivial_task.json`** — a 3-node task that is solvable by hand:

```json
{
  "id": "trivial_task",
  "difficulty": 1,
  "split": "tutorial",
  "episode_date": "2026-04-25",
  "horizon_minutes": 480,
  "tick_minutes": 15,
  "depots": [{"id": "depot-0", "name": "Test Depot",
              "lat": 51.50, "lon": -0.12, "node_idx": 0}],
  "vehicles": [{"id": "v-0", "type": "minibus_16",
                "depot_id": "depot-0",
                "capacity_seats": 16, "capacity_wheelchair": 1,
                "speed_factor": 1.0, "cost_per_km": 0.85,
                "shift_start": 0, "shift_end": 480}],
  "requests": [{"id": "r-0", "kind": "passenger",
                "pickup_node_idx": 1, "dropoff_node_idx": 2,
                "pickup_lat": 51.51, "pickup_lon": -0.10,
                "dropoff_lat": 51.52, "dropoff_lon": -0.16,
                "passengers": 3, "wheelchairs": 0,
                "earliest_pickup": 0, "latest_pickup": 200,
                "earliest_dropoff": 0, "latest_dropoff": 300,
                "service_time": 2, "priority": 1, "released_at": 0}],
  "nodes": [
    {"idx": 0, "name": "Depot", "lat": 51.50, "lon": -0.12,
     "zone": 1, "category": "depot"},
    {"idx": 1, "name": "Pickup A", "lat": 51.51, "lon": -0.10,
     "zone": 1, "category": "landmark"},
    {"idx": 2, "name": "Dropoff A", "lat": 51.52, "lon": -0.16,
     "zone": 1, "category": "landmark"}
  ],
  "distance_matrix_km":  [[0,2,3],[2,0,1],[3,1,0]],
  "duration_matrix_min": [[0,5,8],[5,0,3],[8,3,0]],
  "weather_timeline": [
    {"t": 0, "precip_mm": 0, "wind_kph": 5, "visibility_km": 10, "temp_c": 15}
  ],
  "traffic_events": [],
  "dynamic_events": [],
  "or_tools_baseline_cost": 8.0,
  "or_tools_baseline_unserved": 0,
  "or_tools_baseline_served": 1
}
```

The other fixtures follow the same template with one variation each:
- `oversize_request.json`: same as trivial but with `r-big` requesting `passengers: 99` against a `v-small` with `capacity_seats: 4`
- `tight_window_task.json`: request with `latest_pickup: 1` (impossible)
- `traffic_at_60.json`: adds `{"t_reveal": 60, "node_a": 1, "node_b": 2, "speed_factor": 0.3, "reason": "test"}` to traffic_events
- `breakdown_at_180.json`: adds `{"t": 180, "type": "vehicle_breakdown", "vehicle_id": "v-2"}` to dynamic_events
- `late_request_task.json`: adds a request with `released_at: 200` and an event `{"t": 200, "type": "new_request", "request_id": "r-late-0"}`
- `medium_task.json`: 15 nodes, 3 vehicles, 8 requests, full distance matrix (just write a quick generator script for this one)

### 7.4 Dockerfile

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/__init__.py scripts/__init__.py
COPY scripts/fetch_weather.py scripts/fetch_weather.py
COPY scripts/generate_tasks.py scripts/generate_tasks.py

EXPOSE 8000
CMD ["python", "-m", "src.server"]
```

(`scripts/fetch_weather.py` is needed by `feasibility.py` for the weather speed factor; `generate_tasks.py` exports `VEHICLE_TYPES` referenced by `add_vehicle`.)

### 7.5 `requirements.txt`

```
fastapi>=0.115.12
openreward
pandas
pyarrow
pydantic>=2.0
uvicorn>=0.34.3
ortools>=9.10
requests>=2.31
huggingface-hub>=0.20.0
openai>=1.40
```

### 7.6 Integration tests

```python
# tests/test_integration.py
"""End-to-end smoke tests against a real running server."""
import json, os, signal, subprocess, time
import pytest
from pathlib import Path
from openreward import OpenReward


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Spin up server.py with fixtures dir as ORWD_DATA_DIR. Need to
    construct a tasks.parquet from fixtures first."""
    import pandas as pd
    fixtures = list(Path("tests/fixtures").glob("*.json"))
    rows = [json.loads(f.read_text()) for f in fixtures]
    # Re-tag splits so list_tasks works
    for r in rows:
        r["split"] = "tutorial"
    data_dir = tmp_path_factory.mktemp("orwd_data")
    pd.DataFrame(rows).to_parquet(data_dir / "tasks.parquet")

    env = {**os.environ, "ORWD_DATA_DIR": str(data_dir)}
    proc = subprocess.Popen(
        ["python", "-m", "src.server"], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(4)
    assert proc.poll() is None, ("server failed to start: "
                                 + proc.stderr.read(1000).decode())
    yield proc
    proc.send_signal(signal.SIGTERM)
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired: proc.kill()


def test_server_lists_splits(server):
    c = OpenReward()
    env = c.environments.get(name="LondonDynamicRouting",
                             base_url="http://localhost:8080")
    splits = env.list_splits()
    names = {s.name for s in splits}
    assert names == {"tutorial", "train", "test"}


def test_server_lists_tutorial_tasks(server):
    c = OpenReward()
    env = c.environments.get(name="LondonDynamicRouting",
                             base_url="http://localhost:8080")
    tasks = env.list_tasks(split="tutorial")
    assert len(tasks) >= 1


def test_full_episode_trivial(server):
    c = OpenReward()
    env = c.environments.get(name="LondonDynamicRouting",
                             base_url="http://localhost:8080")
    tasks = env.list_tasks(split="tutorial")
    # find the trivial fixture
    trivial = next(t for t in tasks if t.task_spec["id"] == "trivial_task")
    with env.session(task=trivial) as s:
        s.get_prompt()
        out = s.call_tool("list_pending_requests", {})
        assert "r-0" in out.blocks[0].text
        out = s.call_tool("assign", {"request_id": "r-0",
                                     "vehicle_id": "v-0",
                                     "pickup_position": 0,
                                     "dropoff_position": 1})
        assert out.reward > 0
        out = s.call_tool("tick", {"minutes": 60})
        assert "r-0" in out.blocks[0].text or out.reward > 0
        out = s.call_tool("submit_plan", {})
        assert out.finished
```

```python
# tests/test_rollout.py
"""Smoke test the rollout script end-to-end."""
import os, subprocess, pytest

@pytest.mark.skipif(
    not (os.environ.get("OPENREWARD_API_KEY") and
         os.environ.get("OPENAI_API_KEY")),
    reason="API keys not set")
def test_rollout_local_one_tutorial_task(server):
    """Run the rollout script against the local fixture-based server."""
    env = {**os.environ, "ORWD_DATA_DIR": os.environ.get("ORWD_DATA_DIR", "/tmp")}
    proc = subprocess.run(
        ["python", "scripts/run_rollout.py", "--local",
         "--split", "tutorial", "--task-idx", "0", "--max-turns", "30"],
        capture_output=True, text=True, timeout=300, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "Total reward" in proc.stdout or "Mean reward" in proc.stdout
```

### 7.7 README.md (the environment card)

A short, opinionated env card. Includes:
- One-paragraph "what is this"
- Quickstart (3 commands: install, set keys, run rollout)
- Action space table (copied from §6.7)
- Reward structure summary
- Splits & difficulty table (from §3.2)
- Known limitations (OSRM rate limits, Haversine fallback, weather only at gen time)
- Citation: TfL/OSM data licence note (we use OSM data via OSRM)

### 7.8 Kayode's checklist

- [ ] All 7 fixtures in `tests/fixtures/*.json` exist and validate against schema
- [ ] `Dockerfile` builds: `docker build -t london-routing .`
- [ ] Container starts and serves `:8000`
- [ ] `tests/test_integration.py` passes against running server (3 tests)
- [ ] `python scripts/run_rollout.py --local --task-idx 0` completes successfully
- [ ] Deployment to OpenReward succeeds via GitHub integration
- [ ] `python scripts/run_rollout.py --task-idx 0` against deployed env works
- [ ] `README.md` and `README_dataset.md` committed

---

## 8. Dynamic re-optimization over multi-time horizons

This section explicitly addresses the spec requirement for **dynamic re-optimization** as conditions change.

The mechanism is the **time engine** in §6.5. Concretely:

| Change type | Mechanism | Agent must respond by |
|---|---|---|
| **New passenger demand** mid-day | `dynamic_events: [{t: 240, type: "new_request", request_id: "r-37"}]`. At `tick`, request status flips `unreleased → pending`. | Calling `assign` for the new request; possibly `reassign` to make room. |
| **Vehicle breakdown** | `{t: 180, type: "vehicle_breakdown", vehicle_id: "v-2"}`. Vehicle status `→ broken`; un-started stops returned to pending. | `swap_vehicles` to transfer route, or per-request `reassign`. |
| **Capacity drop** (e.g. wheelchair passenger boards needing extra space) | `{t: 480, type: "capacity_drop", vehicle_id: "v-0", new_capacity_seats: 8, ...}`. If new capacity < peak load, lowest-priority requests bumped to pending. | Reassigning bumped requests; possibly `add_vehicle`. |
| **Driver shift end** | Modeled via `shift_end` per vehicle; `tick` past `shift_end` sets status `→ inactive`. | Plan within shift; or `swap_vehicles` to a healthy vehicle whose shift is still open. |
| **Traffic congestion revealed** | `traffic_events: [{t_reveal: 120, ...}]`. Speed factor on edge becomes known via `revealed_traffic`; ETAs recomputed. | Possibly `reassign` if a route is now infeasible. |
| **Weather degradation** | Weather timeline applies an additional speed multiplier when `precip_mm > 2` (×0.90) or `visibility_km < 2` (×0.85). Both stack. | Same as traffic: `reassign`, or accept higher cost. |

### 8.1 Multi-time-horizon framing

The 16-hour day is divided into **64 fifteen-minute decision intervals**. The agent does not plan all 64 at once. The intended pattern:

1. **Initial planning phase** (t=0): plan first 60–90 minutes against currently-pending requests.
2. **`tick(15)` or `tick(30)`** to the next decision point.
3. **React** to whatever was revealed — new requests, breakdowns, traffic.
4. **Re-plan locally** (e.g. reassign the last 2 stops of an affected vehicle).
5. Loop until horizon.

This decomposition is **what makes it multi-step + multi-horizon**. The agent must trade off between:
- Greedy-now: "assign this request to the cheapest slot right now."
- Lookahead: "save vehicle v-1 capacity for the wheelchair request appearing at t=300."

Difficulty scales the volume and severity of these mid-episode events. At difficulty 100, expect ~10 events spread across the day, including at least one breakdown and one surge (5+ requests appearing in a single 30-min window).

---

## 9. End-to-end timeline (the 3-hour plan)

| Time | Kosi (Data) | Daniel (Env) | Kayode (Infra) |
|---|---|---|---|
| **0:00–0:20** | Curate `data/london_zones_1_4_pois.json` (~80 hand-curated POIs is sufficient for hackathon scope; skip Nominatim if it's flaky) | Skeleton `state.py`, `feasibility.py` against TaskSpec schema | Build all 7 fixtures (`tests/fixtures/*.json`); freeze schema doc |
| **0:20–0:40** | OSRM matrix builder + cache; smoke test on 3 nodes | Implement `assign`, `tick`, `submit_plan`, `get_state`, `list_pending_requests` | `Dockerfile`, `requirements.txt`; first build |
| **0:40–1:10** | Weather (Open-Meteo) + event synthesis; OR-Tools solver wrapper; test on 5-node task | Implement `reassign`, `defer`, `cancel`, query tools; reward function | `tests/test_integration.py`; verify server boots against fixtures |
| **1:10–1:40** | Run `generate_tasks.py --quick` for 5 tutorial tasks; verify schema | Implement breakdown / capacity-drop / new-request handlers; validate against `breakdown_at_180.json` and `late_request_task.json` fixtures | Wire `run_rollout.py`; test with one OpenAI call manually |
| **1:40–2:10** | Run full 100-task generation in parallel (4 workers); fix edge cases | Final tools (`add_vehicle`, `swap_vehicles`, `get_eta`); end-to-end on `medium_task.json` | Run `run_rollout.py --local --split tutorial --task-idx 0` end-to-end |
| **2:10–2:40** | Publish to HF Hub; finalize dataset card | Bug fix in response to rollout failures (always something) | Deploy to OpenReward via GitHub; rollout against deployed env |
| **2:40–3:00** | Demo prep: capture sample trace JSON, prepare slides | Code cleanup; ensure all unit tests pass | Record final demo rollout; ensure trace JSON is shareable |

### 9.1 Cuts to make if behind

In strict order of preference (drop top-of-list first):
1. **Drop wheelchair capacity dimension.** Falls back to single seat-count. Cuts ~30 lines in feasibility.
2. **Drop weather speed modifiers.** Generate weather but no-op in `feasibility.get_edge_duration`. Reduces test surface.
3. **Drop the `add_vehicle` tool.** One fewer recovery option for the agent.
4. **Drop the `defer` tool.** Agent has to choose assign-or-cancel only.
5. **Reduce to 50 tasks** (still 5 tutorial + 30 train + 15 test). Halves OR-Tools wallclock during generation.
6. **Drop heterogeneous vehicles entirely** — all minibus_16. Removes one full feature dimension.

### 9.2 What NOT to cut, even under time pressure

- **The rollout script working end-to-end.** This is the demo.
- **At least 5 tutorial tasks.** The deployed env must demonstrate something visible.
- **The OR-Tools baseline.** Without it, terminal reward is meaningless.
- **`tick()` and at least one type of dynamic event.** Without these, this is just CVRPTW from the previous spec.
- **At least one `breakdown` task** in the tutorial split. This is the "wow, look, multi-horizon!" moment.

---

## 10. Acceptance criteria

The project is complete when:

1. **`pytest tests/`** passes locally with at least 90% green (allowing flakiness on tests that hit external services).
2. **`python scripts/generate_tasks.py`** produces `data/tasks.parquet` with exactly 100 rows, distributed 5/70/25 across splits.
3. **All tutorial tasks** have `or_tools_baseline_unserved == 0` — i.e. they are *guaranteed solvable*. This is what makes the difficulty curve land — a strong agent should solve at least one.
4. **`docker build .`** succeeds and the resulting image starts with `python -m src.server` listening on `:8000`.
5. **`python scripts/run_rollout.py --local --split tutorial --task-idx 0`** completes with a non-trivial trace (≥ 5 actions, terminal reached, total reward in `[-1, 2]`).
6. **Deployment to OpenReward** succeeds; the same rollout against the deployed env (no `--local`) also completes.
7. **The dataset is published** on Hugging Face Hub at `EnvCommons/london-dynamic-routing` (or fallback under the team's username).
8. **Environment card** (README.md) is committed and accurate.
9. **A strong agent (GPT-5/Claude Opus) achieves ≥ 50% coverage on at least one tutorial task.** This is the demo claim; we should verify this in 7.8 before declaring success.

---

## 11. Quick reference: critical implementation gotchas

1. **Task ordering must be deterministic.** `list_tasks` returns tasks sorted by `(split, id)` — see top of `src/server.py`.
2. **`/orwd_data` mount.** When deployed, files uploaded via the OpenReward UI's Files tab are mounted at `/orwd_data`. Use `os.environ.get("ORWD_DATA_DIR", "/orwd_data")` so local dev with a different path works too.
3. **`task_spec` is JSON-serializable.** The whole TaskSpec must round-trip through JSON cleanly. No tuples, no `datetime` objects — convert to ISO strings and `[a, b]` lists. Parquet preserves nested lists/dicts but deserializes some as numpy arrays — convert in the env constructor if needed: `task_spec["distance_matrix_km"] = [list(r) for r in task_spec["distance_matrix_km"]]`.
4. **`ToolOutput` shape.** Always returns `blocks=[TextBlock(type="text", text="...")]`, plus `reward: float`, `finished: bool`. Errors are still successful tool calls — don't raise; return a `ToolOutput` with the error text and a small negative reward.
5. **Pydantic v2.** All param classes inherit `BaseModel`. For tools that take no params, define `class EmptyParams(BaseModel): pass` — don't try to omit the param.
6. **OpenAI Responses API.** When listing tools for the agent, pass `format="openai"` to get the correct schema. The model env var is `ROLLOUT_MODEL` (default `gpt-5.4`).
7. **OSRM rate limits.** The public demo at `router.project-osrm.org` is best-effort. If you hit 5xx during generation, the builder retries with exponential backoff up to 3 times, then falls back to Haversine × 1.4 (note this in the README and in any task with `cache[key]["fallback"] == True`).
8. **Determinism.** Every random choice in generation is seeded by the task seed. The same seed always produces the same task. The same task always grades identically given the same actions.
9. **Pickup must precede dropoff.** Both in feasibility (we check `pickup_pos < dropoff_pos`) and in OR-Tools (`AddPickupAndDelivery` + `time_dim.CumulVar(p) <= time_dim.CumulVar(d)`).

---

## 12. Appendix — TaskSpec JSON schema (formal)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["id", "difficulty", "split", "episode_date",
               "horizon_minutes", "tick_minutes", "depots", "vehicles",
               "requests", "nodes", "distance_matrix_km",
               "duration_matrix_min", "weather_timeline",
               "traffic_events", "dynamic_events",
               "or_tools_baseline_cost", "or_tools_baseline_unserved",
               "or_tools_baseline_served"],
  "properties": {
    "id":                {"type": "string", "pattern": "^london-routing-.+$"},
    "difficulty":        {"type": "integer", "minimum": 1, "maximum": 100},
    "split":             {"type": "string", "enum": ["tutorial","train","test"]},
    "episode_date":      {"type": "string", "format": "date"},
    "horizon_minutes":   {"type": "integer", "minimum": 60},
    "tick_minutes":      {"type": "integer", "enum": [5, 10, 15, 30]},
    "depots":            {"type": "array", "minItems": 1,
                          "items": {"type": "object",
                                    "required":["id","name","lat","lon","node_idx"]}},
    "vehicles": {
      "type": "array", "minItems": 1,
      "items": {
        "type": "object",
        "required": ["id","type","capacity_seats","capacity_wheelchair",
                     "speed_factor","cost_per_km","shift_start","shift_end"]
      }
    },
    "requests": {
      "type": "array", "minItems": 1,
      "items": {
        "type": "object",
        "required": ["id","kind","pickup_node_idx","dropoff_node_idx",
                     "passengers","earliest_pickup","latest_pickup",
                     "earliest_dropoff","latest_dropoff","priority","released_at"]
      }
    },
    "nodes":             {"type": "array", "minItems": 2,
                          "items": {"type": "object",
                                    "required":["idx","name","lat","lon",
                                                "zone","category"]}},
    "distance_matrix_km":{"type": "array", "items": {"type": "array"}},
    "duration_matrix_min":{"type": "array", "items": {"type": "array"}},
    "weather_timeline":  {"type": "array", "minItems": 16, "maxItems": 24},
    "traffic_events":    {"type": "array"},
    "dynamic_events":    {"type": "array"},
    "or_tools_baseline_cost":     {"type": "number"},
    "or_tools_baseline_unserved": {"type": "integer", "minimum": 0},
    "or_tools_baseline_served":   {"type": "integer", "minimum": 0}
  }
}
```

This schema is the single contract between Kosi (generation) and Daniel (consumption). Both sides should validate against it during development.

---

## 13. Appendix — File tree

```
london-dynamic-routing-env/
├── Dockerfile                              [Kayode]
├── requirements.txt                        [Kayode]
├── README.md                               [Kayode]
├── README_dataset.md                       [Kayode]
├── data/
│   ├── london_zones_1_4_pois.json          [Kosi]
│   ├── osrm_cache.json                     [Kosi, generated]
│   └── tasks.parquet                       [Kosi, generated]
├── src/
│   ├── __init__.py                         [Daniel]
│   ├── state.py                            [Daniel]
│   ├── feasibility.py                      [Daniel]
│   ├── reward.py                           [Daniel]
│   ├── time_engine.py                      [Daniel]
│   └── server.py                           [Daniel]
├── scripts/
│   ├── __init__.py                         [Kosi]
│   ├── seed_pois.py                        [Kosi, run-once]
│   ├── build_distance_matrix.py            [Kosi]
│   ├── fetch_weather.py                    [Kosi]
│   ├── synthesize_events.py                [Kosi]
│   ├── solver.py                           [Kosi]
│   ├── generate_tasks.py                   [Kosi]
│   ├── publish_to_hf.py                    [Kosi]
│   └── run_rollout.py                      [Kayode]
└── tests/
    ├── conftest.py                         [Kayode]
    ├── fixtures/
    │   ├── trivial_task.json               [Kayode]
    │   ├── medium_task.json                [Kayode]
    │   ├── tight_window_task.json          [Kayode]
    │   ├── oversize_request.json           [Kayode]
    │   ├── traffic_at_60.json              [Kayode]
    │   ├── breakdown_at_180.json           [Kayode]
    │   └── late_request_task.json          [Kayode]
    ├── test_pois.py                        [Kosi]
    ├── test_distance_matrix.py             [Kosi]
    ├── test_weather.py                     [Kosi]
    ├── test_events.py                      [Kosi]
    ├── test_solver.py                      [Kosi]
    ├── test_generation.py                  [Kosi]
    ├── test_state.py                       [Daniel]
    ├── test_feasibility.py                 [Daniel]
    ├── test_reward.py                      [Daniel]
    ├── test_time_engine.py                 [Daniel]
    ├── test_integration.py                 [Kayode]
    └── test_rollout.py                     [Kayode]
```

**Final tally** — files per developer:

| Dev | Source files | Test files | Total |
|-----|-------------|-----------|-------|
| Kosi | 8 | 6 | 14 |
| Daniel | 6 | 4 | 10 |
| Kayode | 11 (incl. fixtures) | 3 | 14 |

Roughly balanced. Daniel writes the largest amount of dense logic; Kosi has the most files but several are short scripts; Kayode owns the rollout demo and infra which is the "make it visible" critical path.