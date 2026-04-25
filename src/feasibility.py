from .state import EpisodeState, Stop


def get_edge_duration(state: EpisodeState, a: int, b: int) -> float:
    """Driving duration in minutes a→b, accounting for revealed traffic
    AND current weather."""
    base = state.task["duration_matrix_min"][a][b]
    sf = state.revealed_traffic.get((a, b), 1.0)
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
        return False, (f"request not pending (status: "
                       f"{state.request_status.get(request_id)})"), 0.0
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
                 route[pickup_pos:dropoff_pos - 1] + [dropoff_stop] +
                 route[dropoff_pos - 1:])

    # Simulate the route from current time, depot start
    depot_idx = state.task["depots"][0]["node_idx"]
    capacity = veh["capacity_seats"]
    load = 0
    t = float(state.current_time)
    prev_node = depot_idx if not route else route[-1].node_idx

    if not route:
        prev_node = depot_idx

    total_dist_km = 0.0
    for stop in new_route:
        t += get_edge_duration(state, prev_node, stop.node_idx)
        total_dist_km += get_edge_distance(state, prev_node, stop.node_idx)
        r = state.request(stop.request_id)
        if stop.kind == "pickup":
            if t > r["latest_pickup"]:
                return False, (f"misses pickup window for "
                               f"{stop.request_id} (t={t:.0f} > "
                               f"latest={r['latest_pickup']})"), 0.0
            t = max(t, r["earliest_pickup"]) + r.get("service_time", 0)
            load += r["passengers"]
            if load > capacity:
                return False, (f"capacity exceeded ({load}>{capacity}) "
                               f"after picking up {stop.request_id}"), 0.0
        else:
            if t > r["latest_dropoff"]:
                return False, (f"misses dropoff window for "
                               f"{stop.request_id} (t={t:.0f} > "
                               f"latest={r['latest_dropoff']})"), 0.0
            t = max(t, r["earliest_dropoff"]) + r.get("service_time", 0)
            load -= r["passengers"]

        prev_node = stop.node_idx

    # Return to depot, must finish before shift end
    t += get_edge_duration(state, prev_node, depot_idx)
    if t > veh["shift_end"]:
        return False, (f"return-to-depot at {t:.0f} > shift_end "
                       f"{veh['shift_end']}"), 0.0

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
