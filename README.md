# LondonDynamicRouting

An OpenReward environment for **dynamic, multi-horizon, weather- and
traffic-aware vehicle routing on the real London road network.** The agent
acts as a transit fleet dispatcher serving passenger and parcel requests
across one operational day, reacting to traffic events, weather, vehicle
breakdowns, capacity drops, and late-arriving demand.

Built for the OpenReward × EnvCommons hackathon.

## Project layout

```
scripts/                  
  seed_pois.py            geocode London POIs via Nominatim
  build_distance_matrix.py OSRM (opt-in) / Haversine distance & duration matrix
  fetch_weather.py        Open-Meteo hourly timeline + speed-factor model
  synthesize_events.py    traffic + dynamic event generators
  solver.py               OR-Tools CVRPTW / PDPTW baseline (180 s cap)
  generate_tasks.py       orchestrator → tasks.jsonl + tasks.parquet
  publish_to_hf.py        push the dataset to Hugging Face
  run_rollout.py          stream C — drive an LLM agent against a task

src/                      
  server.py               FastAPI tool server (assign, tick, submit_plan, …)
  state.py                episode state machine
  time_engine.py          16 h horizon, 15-min ticks
  feasibility.py          assignment validity checks
  reward.py               per-step shaping + terminal reward
  __main__.py             python -m src.server entry point

tests/                    parametrised + fixture-backed test suites
data/                     committed POI list and locations text file
```

## Quickstart

```bash
# 1. install deps
pip install -r requirements.txt

# 2. set api keys
export OPENREWARD_API_KEY=...
export OPENAI_API_KEY=...

# 3. run a rollout against a tutorial task
python scripts/run_rollout.py --task-idx 0
```

For local development against a server you've started yourself:

```bash
python -m src.server                # in one terminal
python scripts/run_rollout.py --local --task-idx 0
```

## Action space

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

Every action also pays a `−0.005` trajectory-length penalty.

## Reward structure

- **Per-step shaping**: small ± rewards on `assign` / `reassign` to give the
  agent immediate feedback on insertion quality (cheaper insertion → larger
  positive reward).
- **Per-completion**: positive reward when a request's drop-off is served
  on time, scaled by priority.
- **Terminal**: paid at `submit_plan`. Compares total km driven against the
  `or_tools_baseline_cost` and rewards higher coverage of requests served.
  This is the dominant signal — shaping just smooths the path.
- **Trajectory penalty**: `−0.005` per action, so spamming queries hurts.

## Splits & difficulty

| Split | Tasks | Difficulty | Purpose |
|---|---|---|---|
| `tutorial` | 5 | 1–5 | Easy, no dynamic events. Strong agents should hit ≥ 50% coverage on at least one task. |
| `train` | 70 | 1–80 | Bulk RL training. Smooth difficulty ramp. |
| `test` | 25 | 50–100 | Evaluation. Includes the hardest cases. |

Total: **100 tasks**. Difficulty `d ∈ [1, 100]` deterministically scales node
count, fleet size, request volume, time-window tightness, weather severity,
traffic density, number of mid-episode dynamic events, fleet heterogeneity,
and PDPTW share. CVRPTW and PDPTW tasks are interleaved across the difficulty
ramp.

## Multi-horizon dynamics

Episodes run 16 hours (06:00–22:00) on 15-minute decision intervals. The
agent does not plan all 64 intervals at once. The intended pattern:

1. Initial planning at `t=0` against currently pending requests.
2. `tick(15)` or `tick(30)` to the next decision point.
3. React to whatever was revealed — new requests, breakdowns, traffic,
   road disruptions.
4. Re-plan locally (often a `reassign` on the affected vehicle).
5. Loop until horizon, then `submit_plan()`.

Difficulty 100 includes ~10 mid-episode events, including at least one
breakdown and one surge of 5+ requests appearing in a 30-min window.

## Dataset generation pipeline

The dataset under `data/tasks.jsonl` (and the derived `tasks.parquet`) is
produced by the scripts under `scripts/`. Each step is independently runnable
and idempotent.

```bash
# 1. geocode POIs via Nominatim — RUN ONCE, commit data/london_pois.json
python scripts/seed_pois.py

# 2. generate the full 100-task dataset (Haversine distances, parallel solve)
PYTHONPATH=. python scripts/generate_tasks.py --workers 8

# 2b. or use real OSRM road-network distances (slower; rate-limited)
USE_OSRM=1 PYTHONPATH=. python scripts/generate_tasks.py --workers 4

# 2c. quick smoke run (10 tutorial tasks)
PYTHONPATH=. python scripts/generate_tasks.py --quick

# 3. publish to Hugging Face (writes jsonl + parquet + dataset card)
HF_TOKEN=... python scripts/publish_to_hf.py \
    --repo kosiasuzu/london-cvrptw-dynamic-optimization-rl
```

### Pipeline stages

| Stage | Module | Notes |
|---|---|---|
| POI seed | `scripts/seed_pois.py` | Nominatim queries → `data/london_pois.json` (220 unique POIs, zones 1–4). No hardcoded fallbacks: the script fails loudly if Nominatim is unreachable. |
| Distance matrix | `scripts/build_distance_matrix.py` | Default: Haversine × 1.4 at 30 km/h average (fast, deterministic). Set `USE_OSRM=1` to call `router.project-osrm.org` first with retries; falls back to Haversine on failure and tags edges with `"osrm_fallback": true`. Cached in `data/osrm_cache.json` under a thread-safe lock. |
| Weather | `scripts/fetch_weather.py` | Open-Meteo hourly archive when online; `synthetic_weather()` for offline determinism. Speed factor combines precipitation, wind and visibility. |
| Event synthesis | `scripts/synthesize_events.py` | Traffic events (edge slowdowns revealed mid-episode) and dynamic events (new requests, breakdowns, capacity drops, road disruptions / construction). |
| Baseline solver | `scripts/solver.py` | OR-Tools CVRPTW + PDPTW. Time-limit capped at 180 s. Each request is gated by a per-`SetRange` guard so a single bad window degrades to a dropped request rather than aborting the solve. |
| Orchestrator | `scripts/generate_tasks.py` | Difficulty → params → unique-node sampling → solve → write `tasks.jsonl` + `tasks.parquet`. |
| Publish | `scripts/publish_to_hf.py` | Uploads both files plus `README_dataset.md` to the configured HF repo. |

### Output schema

Each line of `tasks.jsonl` is a fully-specified, deterministic episode.
The full schema is documented in [`README_dataset.md`](README_dataset.md).
Every random choice is seeded by the task id, so the same id always produces
the same task and the same task always grades identically given the same
sequence of actions.

## Known limitations

- **OSRM is opt-in.** The default `USE_OSRM=0` uses Haversine × 1.4 so
  generation never stalls on the public demo's rate limits. Set `USE_OSRM=1`
  for real road-network distances; affected edges are flagged
  `"osrm_fallback": true` whenever an OSRM request fails after retries.
- **Weather is fixed at task-generation time.** It does not re-fetch
  during rollout; the timeline embedded in the task is authoritative.
- **POI list is geocoded, not exhaustive.** 220 POIs across zones 1–4
  pulled live from Nominatim. Not a complete TfL stop list.
- **Determinism.** Every random choice in generation is seeded by the task
  seed. The same seed produces the same task; the same task grades
  identically given the same actions.

## Data & licensing

Road-network distances and durations come from
[OSRM](http://project-osrm.org/) over OpenStreetMap data
(© OpenStreetMap contributors, ODbL). Weather snapshots come from
[Open-Meteo](https://open-meteo.com/). London POI selection is informed by
[Transport for London open data](https://tfl.gov.uk/info-for/open-data-users/)
and OSM tags resolved via Nominatim.

This environment is released under the same hackathon-friendly terms as the
rest of the OpenReward EnvCommons.
