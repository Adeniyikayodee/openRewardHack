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
        prev_node = depot_idx  # assume vehicle returns to depot between routes
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
    for stop in state.routes[vehicle_id]:
        if not stop.completed:
            rid = stop.request_id
            if stop.kind == "pickup":
                state.request_status[rid] = "pending"
                state.request_assigned_to.pop(rid, None)
    state.routes[vehicle_id] = []


def handle_capacity_overflow(state: EpisodeState, vehicle_id: str):
    """If new capacity is less than current load, bump lowest-priority
    requests back to pending until under capacity."""
    veh = state.vehicle(vehicle_id)
    new_cap = veh["capacity_seats"]
    load = 0
    peak_load = 0
    affected = []
    for stop in state.routes[vehicle_id]:
        if stop.completed:
            continue
        r = state.request(stop.request_id)
        if stop.kind == "pickup":
            load += r["passengers"]
            affected.append((stop, r))
        else:
            load -= r["passengers"]
        peak_load = max(peak_load, load)

    if peak_load <= new_cap:
        return

    affected.sort(key=lambda x: x[1]["priority"])
    for stop, r in affected:
        if peak_load <= new_cap:
            break
        state.routes[vehicle_id] = [s for s in state.routes[vehicle_id]
                                    if s.request_id != r["id"]]
        state.request_status[r["id"]] = "pending"
        state.request_assigned_to.pop(r["id"], None)
        peak_load -= r["passengers"]


def apply_tick(state: EpisodeState, minutes: int) -> tuple:
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

    state.realized_cost_km = _compute_realized_cost(state, t_new)

    # ─── (2) Reveal traffic events ─────────────────────────────────────
    for ev in state.task["traffic_events"]:
        if t_old < ev["t_reveal"] <= t_new:
            state.revealed_traffic[(ev["node_a"], ev["node_b"])] = ev["speed_factor"]
            state.revealed_traffic[(ev["node_b"], ev["node_a"])] = ev["speed_factor"]
            log.append(f"t={ev['t_reveal']}: {ev['reason']}")

    # ─── (3) Dynamic events ────────────────────────────────────────────
    for ev in state.task["dynamic_events"]:
        if t_old < ev["t"] <= t_new:
            if ev["type"] == "vehicle_breakdown":
                vid = ev["vehicle_id"]
                handle_breakdown(state, vid)
                log.append(f"t={ev['t']}: BREAKDOWN of {vid}")
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
                log.append(f"t={ev['t']}: capacity drop {vid} → "
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
        prev = depot_idx
        for stop in route:
            if stop.completed:
                total += get_edge_distance(state, prev, stop.node_idx)
                prev = stop.node_idx
    return total
