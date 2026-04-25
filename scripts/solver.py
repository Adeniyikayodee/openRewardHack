"""or-tools baseline solver — dual mode: cvrptw and pdptw.

dispatches on task["task_type"]:
  "cvrptw" — depot-origin deliveries, no explicit pickup node.
  "pdptw"  — arbitrary pickup+dropoff pairs (pickup-and-delivery with time windows).

both modes return:
    {"total_cost_km": float, "n_unserved": int, "n_served": int, "routes": list}

note: pywrapcp is not thread-safe for concurrent solve calls. _SOLVER_LOCK
serialises access when generate_tasks.py uses a thread pool.
"""
import threading

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

_SOLVER_LOCK = threading.Lock()

DISJUNCTION_PENALTY = 100_000
# maximum wall-clock seconds the solver is allowed per problem.
# all tasks are sized so a good solution is reachable within this budget.
MAX_SOLVER_SECONDS = 180


def solve_baseline(task: dict, time_limit_s: int = MAX_SOLVER_SECONDS) -> dict:
    with _SOLVER_LOCK:
        if task["task_type"] == "cvrptw":
            return _solve_cvrptw(task, time_limit_s)
        return _solve_pdptw(task, time_limit_s)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _adjusted_durations(task: dict) -> list[list[float]]:
    """apply traffic events known at t=0 to the base duration matrix."""
    dur = [row[:] for row in task["duration_matrix_min"]]
    for ev in task.get("traffic_events", []):
        if ev["t_reveal"] == 0:
            a, b = ev["node_a"], ev["node_b"]
            sf = ev["speed_factor"]
            if dur[a][b] > 0:
                dur[a][b] /= sf
            if dur[b][a] > 0:
                dur[b][a] /= sf
    return dur


def _dist_cb_factory(task: dict, manager):
    dist = task["distance_matrix_km"]

    def cb(i, j):
        return int(dist[manager.IndexToNode(i)][manager.IndexToNode(j)] * 1000)

    return cb


def _search_params(time_limit_s: int):
    params = pywrapcp.DefaultRoutingSearchParameters()
    # path_cheapest_arc is reliable across both cvrptw and pdptw problem sizes
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = time_limit_s
    return params


def _extract_result(task: dict, routing, manager, sol) -> dict:
    if not sol:
        return {
            "total_cost_km": float("inf"),
            "n_unserved": len(task["requests"]),
            "n_served": 0,
            "routes": [],
        }

    dist = task["distance_matrix_km"]
    n_vehicles = len(task["vehicles"])
    pickup_nodes = {r["pickup_node_idx"]: r["id"] for r in task["requests"]
                   if "pickup_node_idx" in r}
    # for cvrptw requests use dropoff_node_idx as the served node
    dropoff_nodes = {r["dropoff_node_idx"]: r["id"] for r in task["requests"]
                    if "pickup_node_idx" not in r}

    served: set[str] = set()
    routes: list[list[int]] = []
    total_dist_m = 0

    for v in range(n_vehicles):
        idx = routing.Start(v)
        route_nodes: list[int] = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route_nodes.append(node)
            next_idx = sol.Value(routing.NextVar(idx))
            if not routing.IsEnd(next_idx):
                ni = manager.IndexToNode(next_idx)
                total_dist_m += int(dist[node][ni] * 1000)
            idx = next_idx
        routes.append(route_nodes)
        for node in route_nodes:
            if node in pickup_nodes:
                served.add(pickup_nodes[node])
            if node in dropoff_nodes:
                served.add(dropoff_nodes[node])

    n_served = len(served)
    n_unserved = len(task["requests"]) - n_served
    return {
        "total_cost_km": total_dist_m / 1000.0,
        "n_unserved": n_unserved,
        "n_served": n_served,
        "routes": routes,
    }


# ---------------------------------------------------------------------------
# cvrptw: depot-origin deliveries
# ---------------------------------------------------------------------------

def _solve_cvrptw(task: dict, time_limit_s: int) -> dict:
    n_nodes    = len(task["nodes"])
    n_vehicles = len(task["vehicles"])
    depot_idx  = task["depots"][0]["node_idx"]
    horizon    = task["horizon_minutes"]
    dur        = _adjusted_durations(task)

    manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)

    dist_cb = _dist_cb_factory(task, manager)
    transit_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # capacity dimension — unary demand at dropoff
    demands = [0] * n_nodes
    for r in task["requests"]:
        demands[r["dropoff_node_idx"]] += r.get("passengers", 1)

    def demand_cb(i):
        return demands[manager.IndexToNode(i)]

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    capacities = [v["capacity_seats"] for v in task["vehicles"]]
    routing.AddDimensionWithVehicleCapacity(demand_cb_idx, 0, capacities, True, "Capacity")

    # time dimension — window on delivery node only
    def time_cb(i, j):
        a = manager.IndexToNode(i)
        b = manager.IndexToNode(j)
        return max(1, int(dur[a][b] + 0.5))

    time_cb_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_cb_idx, horizon, horizon, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for r in task["requests"]:
        d_idx = manager.NodeToIndex(r["dropoff_node_idx"])
        e, l = int(r["earliest_dropoff"]), int(r["latest_dropoff"])
        if e > l:
            continue  # invalid window — leave node unconstrained, disjunction will drop it
        try:
            time_dim.CumulVar(d_idx).SetRange(e, l)
        except Exception:
            continue
        routing.AddDisjunction([d_idx], DISJUNCTION_PENALTY)

    sol = routing.SolveWithParameters(_search_params(time_limit_s))
    return _extract_result(task, routing, manager, sol)


# ---------------------------------------------------------------------------
# pdptw: pickup-and-delivery with time windows
# ---------------------------------------------------------------------------

def _solve_pdptw(task: dict, time_limit_s: int) -> dict:
    n_nodes    = len(task["nodes"])
    n_vehicles = len(task["vehicles"])
    depot_idx  = task["depots"][0]["node_idx"]
    horizon    = task["horizon_minutes"]
    dur        = _adjusted_durations(task)

    manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)

    dist_cb = _dist_cb_factory(task, manager)
    transit_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # capacity dimension — pickup adds, dropoff removes
    demands = [0] * n_nodes
    for r in task["requests"]:
        demands[r["pickup_node_idx"]]  += r.get("passengers", 1)
        demands[r["dropoff_node_idx"]] -= r.get("passengers", 1)

    def demand_cb(i):
        return demands[manager.IndexToNode(i)]

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    capacities = [v["capacity_seats"] for v in task["vehicles"]]
    routing.AddDimensionWithVehicleCapacity(demand_cb_idx, 0, capacities, True, "Capacity")

    # time dimension
    def time_cb(i, j):
        a = manager.IndexToNode(i)
        b = manager.IndexToNode(j)
        return max(1, int(dur[a][b] + 0.5))

    time_cb_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_cb_idx, horizon, horizon, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    # pickup-and-delivery pairs
    for r in task["requests"]:
        e_p, l_p = int(r["earliest_pickup"]),  int(r["latest_pickup"])
        e_d, l_d = int(r["earliest_dropoff"]), int(r["latest_dropoff"])
        if e_p > l_p or e_d > l_d:
            continue  # invalid window — skip this request
        p = manager.NodeToIndex(r["pickup_node_idx"])
        d = manager.NodeToIndex(r["dropoff_node_idx"])
        try:
            time_dim.CumulVar(p).SetRange(e_p, l_p)
            time_dim.CumulVar(d).SetRange(e_d, l_d)
        except Exception:
            continue
        routing.AddPickupAndDelivery(p, d)
        routing.solver().Add(routing.VehicleVar(p) == routing.VehicleVar(d))
        routing.solver().Add(time_dim.CumulVar(p) <= time_dim.CumulVar(d))
        # separate disjunctions so the pair is skipped as a unit via AddPickupAndDelivery
        routing.AddDisjunction([p], DISJUNCTION_PENALTY)
        routing.AddDisjunction([d], DISJUNCTION_PENALTY)

    sol = routing.SolveWithParameters(_search_params(time_limit_s))
    return _extract_result(task, routing, manager, sol)
