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
