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


def _to_py(obj):
    """recursively convert numpy types from parquet roundtrip back to plain
    python primitives so downstream consumers (state, feasibility, the
    openreward serializer) see real lists / dicts."""
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        return _to_py(obj.tolist())
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    return obj


if _TASKS_PATH.exists():
    ALL_TASKS = [_to_py(t)
                 for t in pd.read_parquet(_TASKS_PATH).to_dict(orient="records")]
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
                 f"{len(t['requests']) - n_initial} released later).\n\n"
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
            self.state.routes[old_vid] = old_route
            self.state.request_status[rid] = "assigned"
            self.state.request_assigned_to[rid] = old_vid
            return ToolOutput(
                blocks=[TextBlock(type="text",
                    text=f"INVALID reassign: {reason} (rolled back)")],
                reward=self._shape(SHAPE_INVALID_ACTION), finished=False)
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
        moved = []
        for stop in self.state.routes[a]:
            if stop.kind == "pickup" and not stop.completed:
                rid = stop.request_id
                self.state.request_status[rid] = "pending"
                self.state.request_assigned_to.pop(rid, None)
                moved.append(rid)
        self.state.routes[a] = []
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
        pending = [self.state.request(rid) for rid in s.pending_request_ids()]
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
