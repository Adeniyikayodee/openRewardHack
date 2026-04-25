---
license: odbl
task_categories:
  - reinforcement-learning
  - other
tags:
  - vehicle-routing
  - pdptw
  - dynamic-routing
  - london
  - openreward
  - envcommons
pretty_name: London Dynamic Routing
size_categories:
  - n<1K
---

# london-dynamic-routing

Task dataset for the **LondonDynamicRouting** OpenReward environment:
dynamic, multi-horizon, weather- and traffic-aware vehicle routing on the
real London road network.

100 tasks of monotonically increasing difficulty (1..100), partitioned into
`tutorial` (5), `train` (70), `test` (25) splits.

## Format

A single `tasks.parquet` file. Each row is a fully-specified, deterministic
episode. The columns match the **TaskSpec** schema documented in the
[environment repo](../README.md). Key columns:

| Column | Type | Notes |
|---|---|---|
| `id` | string | `london-routing-NNN` |
| `difficulty` | int | 1..100 |
| `split` | string | `tutorial` \| `train` \| `test` |
| `episode_date` | string | ISO date used to sample weather |
| `horizon_minutes` | int | typically 960 (16 h) |
| `tick_minutes` | int | decision interval, typically 15 |
| `depots` | list[obj] | 1–3 garages with `lat`, `lon`, `node_idx` |
| `vehicles` | list[obj] | heterogeneous fleet, capacities, shifts |
| `requests` | list[obj] | passenger / parcel, time windows, priority |
| `nodes` | list[obj] | geocoded London locations |
| `distance_matrix_km` | list[list[float]] | N×N OSRM distances |
| `duration_matrix_min` | list[list[float]] | N×N OSRM durations |
| `weather_timeline` | list[obj] | hourly precipitation, wind, visibility |
| `traffic_events` | list[obj] | edge slowdowns revealed over time |
| `dynamic_events` | list[obj] | new requests, breakdowns, capacity drops |
| `or_tools_baseline_cost` | float | km driven by OR-Tools baseline |
| `or_tools_baseline_unserved` | int | requests OR-Tools could not serve |
| `or_tools_baseline_served` | int | requests OR-Tools served |

## Splits

| Split | Tasks | Difficulty | Purpose |
|---|---|---|---|
| `tutorial` | 5 | 1–5 | Easy, no dynamic events. Smoke tests + demos. |
| `train` | 70 | 1–80 | Bulk RL training. Smooth difficulty ramp. |
| `test` | 25 | 50–100 | Evaluation. Includes hardest cases. |

All `tutorial` tasks have `or_tools_baseline_unserved == 0` — they are
guaranteed solvable by the baseline and a strong LLM agent should hit
≥ 50% coverage on at least one.

## Loading

```python
import pandas as pd
df = pd.read_parquet("tasks.parquet")
print(df["split"].value_counts())
trivial = df[df["id"] == "london-routing-001"].iloc[0].to_dict()
```

Or via the OpenReward client:

```python
from openreward import OpenReward
env = OpenReward().environments.get(name="EnvCommons/LondonDynamicRouting")
tasks = env.list_tasks(split="tutorial")
```

## Determinism

Every random choice in task generation is seeded by the task's id. The
same id always produces the same task; the same task always grades
identically given the same sequence of actions.

## Sources & licensing

- Road-network distances/durations: by default the builder uses
  Haversine × 1.4 at 30 km/h average for fast, rate-limit-free generation.
  Setting `USE_OSRM=1` switches to live [OSRM](http://project-osrm.org/)
  queries over OpenStreetMap (© OpenStreetMap contributors, ODbL); edges
  that fall back after OSRM failure are flagged `"osrm_fallback": true`.
- Weather: [Open-Meteo](https://open-meteo.com/) historical hourly archive.
- POIs: 220 unique London locations across zones 1–4, geocoded live via
  [Nominatim](https://nominatim.org/) over
  [Transport for London open data](https://tfl.gov.uk/info-for/open-data-users/)
  and OSM tags. No hardcoded fallbacks; see `scripts/seed_pois.py`.

Released under ODbL to match upstream OpenStreetMap/TfL terms.

## Citation

```
@misc{london-dynamic-routing-2026,
  title = {LondonDynamicRouting: a multi-horizon vehicle-routing
           OpenReward environment},
  author = {Kosi and Daniel and Kayode},
  year = 2026,
  howpublished = {OpenReward EnvCommons hackathon}
}
```
