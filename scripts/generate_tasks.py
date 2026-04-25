"""generate the london-dynamic-routing dataset.

outputs data/tasks.jsonl (primary, one json per line) and data/tasks.parquet (derived).

usage:
    python scripts/generate_tasks.py                  # full 100 tasks
    python scripts/generate_tasks.py --quick          # 10 tutorial tasks only
    python scripts/generate_tasks.py --n-tasks 50     # custom count
    python scripts/generate_tasks.py --workers 4      # override worker count

task distribution (100 tasks):
    5  tutorial  (difficulty 1-5, cycling)
    70 train     (difficulty 1-80, linear ramp)
    25 test      (difficulty 50-100, linear ramp)
"""
import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from scripts.build_distance_matrix import build_matrix, is_fallback
from scripts.fetch_weather import synthetic_weather
from scripts.synthesize_events import synthesize_dynamic_events, synthesize_traffic_events
from scripts.solver import MAX_SOLVER_SECONDS, solve_baseline

POI_FILE = Path("data/london_pois.json")

VEHICLE_TYPES = [
    {"type": "minibus_16",      "capacity_seats": 16, "capacity_wheelchair": 1,
     "speed_factor": 1.00, "cost_per_km": 0.85},
    {"type": "accessible_van_8","capacity_seats":  8, "capacity_wheelchair": 2,
     "speed_factor": 0.95, "cost_per_km": 0.70},
    {"type": "minibus_24",      "capacity_seats": 24, "capacity_wheelchair": 1,
     "speed_factor": 0.90, "cost_per_km": 1.05},
    {"type": "small_van_4",     "capacity_seats":  4, "capacity_wheelchair": 0,
     "speed_factor": 1.05, "cost_per_km": 0.55},
]


# ---------------------------------------------------------------------------
# difficulty → parameter mapping
# ---------------------------------------------------------------------------

def difficulty_params(d: int) -> dict:
    # node and request counts are sized so every task is solvable within
    # MAX_SOLVER_SECONDS (3 min). empirical limits:
    #   cvrptw: up to 50 nodes / 40 requests within 3 min
    #   pdptw:  up to 40 nodes / 20 pairs within 3 min (each pair needs 2 nodes)
    return {
        "n_nodes":          min(50, int(8 + d * 0.42)),   # 8 → 50
        "n_vehicles":       2 + d // 25,                   # 2 → 6
        "n_requests":       min(40, int(5 + d * 0.35)),    # 5 → 40
        "weather_severity": min(1.0, d / 100),
        "traffic_density":  min(1.0, d / 80),
        "n_dynamic_events": d // 10,                       # 0 → 10
        "heterogeneity":    min(1.0, d / 50),
        "tw_tightness":     0.3 + 0.5 * (d / 100),        # 0.30 → 0.80
    }


def task_type_for(d: int, rng: random.Random) -> str:
    """low difficulty → cvrptw; high → pdptw; blend in between."""
    pdptw_prob = max(0.0, min(1.0, (d - 30) / 40))
    return "pdptw" if rng.random() < pdptw_prob else "cvrptw"


# ---------------------------------------------------------------------------
# synthesizers
# ---------------------------------------------------------------------------

def synthesize_vehicles(n: int, depot_id: str, heterogeneity: float,
                        rng: random.Random) -> list[dict]:
    vehicles = []
    for i in range(n):
        if heterogeneity < 0.001 or rng.random() > heterogeneity:
            tmpl = VEHICLE_TYPES[0]
        else:
            tmpl = rng.choice(VEHICLE_TYPES)
        shift_end = 960 if rng.random() > 0.2 else rng.randint(480, 900)
        vehicles.append({
            "id": f"v-{i}",
            "type": tmpl["type"],
            "depot_id": depot_id,
            "capacity_seats": tmpl["capacity_seats"],
            "capacity_wheelchair": tmpl["capacity_wheelchair"],
            "speed_factor": tmpl["speed_factor"],
            "cost_per_km": tmpl["cost_per_km"],
            "shift_start": 0,
            "shift_end": shift_end,
        })
    return vehicles


def synthesize_cvrptw_requests(n: int, nodes: list[dict], depot_idx: int,
                                params: dict, rng: random.Random,
                                horizon: int = 960) -> list[dict]:
    """cvrptw: depot → dropoff. modelled in the unified TaskSpec as a
    pickup-at-depot + dropoff request so the runtime treats every request
    identically.

    each request gets a unique dropoff node so the solver can apply distinct
    time windows without conflicts on a shared CumulVar.
    """
    reqs = []
    by_idx = {nd["idx"]: nd for nd in nodes}
    depot_nd = by_idx[depot_idx]
    non_depot = [nd["idx"] for nd in nodes if nd["idx"] != depot_idx]
    if not non_depot:
        return reqs
    tw_tightness = params["tw_tightness"]
    # sample without replacement; cap at available unique nodes
    n = min(n, len(non_depot))
    chosen = rng.sample(non_depot, n)

    for i in range(n):
        do = chosen[i]
        passengers  = rng.choices([1, 2, 3, 4, 5, 6], [4, 3, 2, 1, 1, 1])[0]
        wheelchairs = 1 if rng.random() < 0.10 else 0
        priority    = rng.choices([1, 2, 3, 4, 5], [5, 3, 2, 1, 1])[0]
        released_at = 0 if rng.random() < 0.6 else rng.randint(60, horizon - 240)
        center      = rng.randint(released_at + 30, horizon - 60)
        half_width  = int(60 + (1 - tw_tightness) * 240)
        e_do = max(released_at, center - half_width)
        l_do = min(horizon, center + half_width)
        # pickup happens at the depot; window is wide so the dropoff window
        # is the binding constraint.
        e_pu = released_at
        l_pu = l_do
        reqs.append({
            "id": f"r-{i}",
            "kind": "parcel",
            "pickup_node_idx":  depot_idx,
            "dropoff_node_idx": do,
            "pickup_lat":  depot_nd["lat"], "pickup_lon":  depot_nd["lon"],
            "dropoff_lat": by_idx[do]["lat"],
            "dropoff_lon": by_idx[do]["lon"],
            "passengers":  passengers,
            "wheelchairs": wheelchairs,
            "earliest_pickup":  e_pu, "latest_pickup":  l_pu,
            "earliest_dropoff": e_do, "latest_dropoff": l_do,
            "service_time": 2,
            "priority":    priority,
            "released_at": released_at,
        })
    return reqs


def synthesize_pdptw_requests(n: int, nodes: list[dict], depot_idx: int,
                               params: dict, rng: random.Random,
                               horizon: int = 960) -> list[dict]:
    """pdptw: explicit pickup + dropoff pair.

    each request gets a unique pickup node and a unique dropoff node, all
    mutually disjoint, so the solver can apply distinct time windows without
    conflicts on a shared CumulVar.
    """
    reqs = []
    by_idx = {nd["idx"]: nd for nd in nodes}
    non_depot = [nd["idx"] for nd in nodes if nd["idx"] != depot_idx]
    if len(non_depot) < 2:
        return reqs
    tw_tightness = params["tw_tightness"]
    # each pair consumes 2 unique non-depot nodes
    max_pairs = len(non_depot) // 2
    n = min(n, max_pairs)
    pool = rng.sample(non_depot, 2 * n)

    for i in range(n):
        pu = pool[2 * i]
        do = pool[2 * i + 1]
        passengers  = rng.choices([1, 2, 3, 4, 5, 6], [4, 3, 2, 1, 1, 1])[0]
        wheelchairs = 1 if rng.random() < 0.10 else 0
        priority    = rng.choices([1, 2, 3, 4, 5], [5, 3, 2, 1, 1])[0]
        released_at = 0 if rng.random() < 0.6 else rng.randint(60, horizon - 240)
        center      = rng.randint(released_at + 30, horizon - 60)
        half_width  = int(60 + (1 - tw_tightness) * 240)
        e_pu = max(released_at, center - half_width)
        l_pu = min(horizon - 30, center + half_width)
        e_do = e_pu + 10
        l_do = min(horizon, l_pu + 90 + int((1 - tw_tightness) * 120))
        reqs.append({
            "id": f"r-{i}",
            "kind": "passenger",
            "pickup_node_idx":  pu,
            "dropoff_node_idx": do,
            "pickup_lat":  by_idx[pu]["lat"], "pickup_lon":  by_idx[pu]["lon"],
            "dropoff_lat": by_idx[do]["lat"], "dropoff_lon": by_idx[do]["lon"],
            "passengers":  passengers,
            "wheelchairs": wheelchairs,
            "earliest_pickup":  e_pu, "latest_pickup":  l_pu,
            "earliest_dropoff": e_do, "latest_dropoff": l_do,
            "service_time": 2,
            "priority":    priority,
            "released_at": released_at,
        })
    return reqs


# ---------------------------------------------------------------------------
# single task generator
# ---------------------------------------------------------------------------

def generate_task(seed: int, difficulty: int, split: str,
                  pois: list[dict], episode_date: str = "2026-04-25",
                  solver_time_limit: int = MAX_SOLVER_SECONDS) -> dict:
    rng    = random.Random(seed)
    params = difficulty_params(difficulty)
    ttype  = task_type_for(difficulty, rng)

    n_nodes = min(params["n_nodes"], len(pois))
    sampled = rng.sample(pois, n_nodes)
    # ensure at least one depot category entry
    if not any(p["category"] == "depot" for p in sampled):
        depots = [p for p in pois if p["category"] == "depot"]
        if depots:
            sampled[-1] = rng.choice(depots)

    nodes     = [{"idx": i, **p} for i, p in enumerate(sampled)]
    depot_nd  = next((nd for nd in nodes if nd["category"] == "depot"), nodes[0])
    depot_idx = depot_nd["idx"]

    coords   = [(nd["lon"], nd["lat"]) for nd in nodes]
    dist_km, dur_min = build_matrix(coords)
    fallback = is_fallback(coords)

    vehicles = synthesize_vehicles(
        params["n_vehicles"], "depot-0", params["heterogeneity"], rng)

    n_req = params["n_requests"]
    n_dyn = params["n_dynamic_events"]
    # reserve nodes for late requests so they don't collide with initial dropoffs
    if ttype == "pdptw":
        # each pair needs 2 unique non-depot nodes; cap so the problem stays
        # within the 3-minute solver budget
        max_pairs = (n_nodes - 1) // 2
        n_req = min(n_req, max(0, max_pairs - n_dyn), 20)
        requests = synthesize_pdptw_requests(n_req, nodes, depot_idx, params, rng)
    else:
        n_req = min(n_req, max(0, (n_nodes - 1) - n_dyn))
        requests = synthesize_cvrptw_requests(n_req, nodes, depot_idx, params, rng)

    weather = synthetic_weather(params["weather_severity"])

    n_traffic = max(0, int(params["traffic_density"] * 12))
    traffic_events = synthesize_traffic_events(nodes, dist_km, n_traffic, rng)

    dynamic_events = synthesize_dynamic_events(
        vehicles, requests, params["n_dynamic_events"], 960, rng, nodes=nodes)

    # resolve late-request placeholder ids using only nodes not already taken
    used_nodes: set[int] = set()
    for r in requests:
        if "pickup_node_idx" in r:
            used_nodes.add(r["pickup_node_idx"])
        used_nodes.add(r["dropoff_node_idx"])

    late_idx = 0
    for ev in dynamic_events:
        if ev["type"] != "new_request":
            continue
        free_nodes = [nd for nd in nodes
                      if nd["idx"] != depot_idx and nd["idx"] not in used_nodes]
        needed = 2 if ttype == "pdptw" else 1
        if len(free_nodes) < needed:
            continue  # no room for another unique-node request

        new_id = f"r-late-{late_idx}"
        late_idx += 1
        if ttype == "cvrptw":
            late = synthesize_cvrptw_requests(1, free_nodes, depot_idx, params, rng)
        else:
            late = synthesize_pdptw_requests(1, free_nodes, depot_idx, params, rng)
        if not late:
            continue
        lr = late[0]
        lr["id"] = new_id
        lr["released_at"] = ev["t"]
        horizon_m = 960
        # rebuild a valid forward-looking window starting at ev["t"]
        if ttype == "cvrptw":
            base_e = max(int(lr["earliest_dropoff"]), int(ev["t"]))
            base_l = max(int(lr["latest_dropoff"]), base_e + 60)
            lr["earliest_dropoff"] = min(base_e, horizon_m - 30)
            lr["latest_dropoff"]   = min(base_l, horizon_m)
            used_nodes.add(lr["dropoff_node_idx"])
        else:
            base_pe = max(int(lr["earliest_pickup"]), int(ev["t"]))
            base_pl = max(int(lr["latest_pickup"]),   base_pe + 60)
            base_de = max(int(lr["earliest_dropoff"]), base_pe + 10)
            base_dl = max(int(lr["latest_dropoff"]),   base_de + 60)
            lr["earliest_pickup"]  = min(base_pe, horizon_m - 90)
            lr["latest_pickup"]    = min(base_pl, horizon_m - 30)
            lr["earliest_dropoff"] = min(base_de, horizon_m - 60)
            lr["latest_dropoff"]   = min(base_dl, horizon_m)
            used_nodes.add(lr["pickup_node_idx"])
            used_nodes.add(lr["dropoff_node_idx"])
        requests.append(lr)
        ev["request_id"] = new_id

    task: dict = {
        "id": f"london-routing-{seed:05d}",
        "task_type": ttype,
        "difficulty": difficulty,
        "split": split,
        "episode_date": episode_date,
        "horizon_minutes": 960,
        "tick_minutes": 15,
        "depots": [{
            "id": "depot-0",
            "name": depot_nd["name"],
            "lat": depot_nd["lat"],
            "lon": depot_nd["lon"],
            "node_idx": depot_idx,
        }],
        "vehicles": vehicles,
        "requests": requests,
        "nodes": nodes,
        "distance_matrix_km":  dist_km,
        "duration_matrix_min": dur_min,
        "weather_timeline": weather,
        "traffic_events": traffic_events,
        "dynamic_events": dynamic_events,
        "osrm_fallback": fallback,
    }

    try:
        baseline = solve_baseline(task, time_limit_s=solver_time_limit)
    except Exception:
        baseline = {"total_cost_km": -1.0, "n_unserved": len(requests), "n_served": 0}
    task["or_tools_baseline_cost"]     = baseline["total_cost_km"]
    task["or_tools_baseline_unserved"] = baseline["n_unserved"]
    task["or_tools_baseline_served"]   = baseline["n_served"]
    return task


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

def build_plan(quick: bool, n_tasks: int | None = None) -> list[tuple[str, int, int]]:
    plan: list[tuple[str, int, int]] = []
    if quick:
        for i in range(10):
            plan.append(("tutorial", 10_000 + i, i % 5 + 1))
        return plan
    for i in range(5):
        plan.append(("tutorial", 10_000 + i, i % 5 + 1))
    for i in range(70):
        plan.append(("train", 20_000 + i, 1 + (i * 79) // 69))
    for i in range(25):
        plan.append(("test", 30_000 + i, 50 + (i * 50) // 24))
    if n_tasks is not None:
        plan = plan[:n_tasks]
    return plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="generate 10 tutorial tasks only (smoke test)")
    ap.add_argument("--n-tasks", type=int, default=None,
                    help="limit total tasks generated (default: full 100)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-jsonl",   default="data/tasks.jsonl")
    ap.add_argument("--out-parquet", default="data/tasks.parquet")
    ap.add_argument("--episode-date", default="2026-04-25")
    ap.add_argument("--solver-time-limit", type=int, default=MAX_SOLVER_SECONDS,
                    help=f"or-tools time budget per task in seconds (default: {MAX_SOLVER_SECONDS})")
    args = ap.parse_args()

    pois = json.loads(POI_FILE.read_text())
    print(f"loaded {len(pois)} pois from {POI_FILE}")

    plan  = build_plan(args.quick, args.n_tasks)
    total = len(plan)
    print(f"generating {total} tasks with {args.workers} workers...")

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    failed    = 0

    with open(args.out_jsonl, "w") as out_f, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:

        futures = {
            ex.submit(generate_task, seed, d, split, pois,
                      args.episode_date, args.solver_time_limit): (split, seed, d)
            for split, seed, d in plan
        }

        for fut in as_completed(futures):
            split, seed, d = futures[fut]
            try:
                task = fut.result()
                out_f.write(json.dumps(task) + "\n")
                completed += 1
                print(f"  [{completed}/{total}] {task['id']} type={task['task_type']} "
                      f"d={d} served={task['or_tools_baseline_served']}/"
                      f"{len(task['requests'])} "
                      f"{'(fallback)' if task['osrm_fallback'] else ''}")
            except Exception as e:
                failed += 1
                print(f"  fail {split} seed={seed} d={d}: {e}", file=sys.stderr)

    print(f"\ndone: {completed} ok, {failed} failed → {args.out_jsonl}")

    # derive parquet
    tasks = [json.loads(line) for line in Path(args.out_jsonl).open()]
    tasks.sort(key=lambda t: (t["split"], t["id"]))
    pd.DataFrame(tasks).to_parquet(args.out_parquet)
    df = pd.DataFrame(tasks)
    print(f"wrote {len(df)} rows → {args.out_parquet}")
    print(f"  tutorial: {(df['split'] == 'tutorial').sum()}")
    print(f"  train:    {(df['split'] == 'train').sum()}")
    print(f"  test:     {(df['split'] == 'test').sum()}")


if __name__ == "__main__":
    main()
