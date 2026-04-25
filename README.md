# LondonDynamicRouting

An OpenReward environment for **dynamic, multi-horizon, weather- and
traffic-aware vehicle routing on the real London road network.** The agent
acts as a transit fleet dispatcher serving passenger and parcel requests
across one operational day, reacting to traffic events, weather, vehicle
breakdowns, capacity drops, and late-arriving demand.

Built for the OpenReward × EnvCommons hackathon.

## Quickstart

```bash
# 1. install deps
pip install -r requirements.txt

# 2. set API keys
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

Difficulty `d ∈ [1, 100]` deterministically scales node count, fleet size,
request volume, time-window tightness, weather severity, traffic density,
number of mid-episode dynamic events, fleet heterogeneity, and PDPTW share.

## Multi-horizon dynamics

Episodes run 16 hours (06:00–22:00) on 15-minute decision intervals. The
agent does not plan all 64 intervals at once. The intended pattern:

1. Initial planning at `t=0` against currently pending requests.
2. `tick(15)` or `tick(30)` to the next decision point.
3. React to whatever was revealed — new requests, breakdowns, traffic.
4. Re-plan locally (often a `reassign` on the affected vehicle).
5. Loop until horizon, then `submit_plan()`.

Difficulty 100 includes ~10 mid-episode events, including at least one
breakdown and one surge of 5+ requests appearing in a 30-min window.

## Known limitations

- **OSRM rate limits.** The public `router.project-osrm.org` demo is
  best-effort. Distance-matrix builds retry with exponential backoff up to
  3 times, then fall back to Haversine × 1.4. Tasks built from the fallback
  carry a `cache[key]["fallback"] == True` marker.
- **Weather is fixed at task-generation time.** It does not re-fetch
  during rollout; the timeline embedded in the task is authoritative.
- **POI list is hand-curated.** ~80 London POIs in zones 1–4. Not a complete
  TfL stop list.
- **Determinism.** Every random choice in generation is seeded by the task
  seed. The same seed produces the same task; the same task grades
  identically given the same actions.

## Data & licensing

Road-network distances and durations come from
[OSRM](http://project-osrm.org/) over OpenStreetMap data
(© OpenStreetMap contributors, ODbL). Weather snapshots come from
[Open-Meteo](https://open-meteo.com/). London POI selection is informed by
[Transport for London open data](https://tfl.gov.uk/info-for/open-data-users/).

This environment is released under the same hackathon-friendly terms as the
rest of the OpenReward EnvCommons.
