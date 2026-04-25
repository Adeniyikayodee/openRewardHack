"""Smoke test: instantiate the environment from the trivial fixture and
call all 14 tools at least once without crashing."""
import json
import pytest
from src.server import LondonDynamicRouting
from src.state import Stop


def make_env(fixture="trivial_task.json"):
    task = json.load(open(f"tests/fixtures/{fixture}"))
    return LondonDynamicRouting(task_spec=task)


# ── helpers ──────────────────────────────────────────────────────────────────

def call(env, tool_name, **kwargs):
    tool_fn = getattr(env, tool_name)
    # Each @tool method accepts a pydantic params model; the Environment base
    # class also exposes a dict-based call_tool path, but direct invocation
    # via the param class is simpler for unit tests.
    param_cls = tool_fn.__func__.__annotations__.get("params") or tool_fn.__annotations__.get("params")
    # Fall back: build the param model from kwargs
    import inspect
    sig = inspect.signature(tool_fn)
    param_cls = list(sig.parameters.values())[0].annotation
    params = param_cls(**kwargs)
    return tool_fn(params)


# ── tool-by-tool smoke ────────────────────────────────────────────────────────

def test_list_pending_requests():
    env = make_env()
    out = env.list_pending_requests(env.list_pending_requests.__func__.__annotations__
                                    .get("params", type("E", (), {}))())
    # simpler: just call via the param classes directly
    from src.server import EmptyParams
    out = env.list_pending_requests(EmptyParams())
    assert "r-0" in out.blocks[0].text


def test_get_state():
    from src.server import EmptyParams
    env = make_env()
    out = env.get_state(EmptyParams())
    assert "v-0" in out.blocks[0].text


def test_get_distance():
    from src.server import QueryEdgeParams
    env = make_env()
    out = env.get_distance(QueryEdgeParams(node_a=0, node_b=1))
    assert "km" in out.blocks[0].text


def test_get_eta():
    from src.server import QueryEtaParams
    env = make_env()
    out = env.get_eta(QueryEtaParams(node_a=0, node_b=1))
    assert "min" in out.blocks[0].text


def test_query_traffic():
    from src.server import QueryEdgeParams
    env = make_env()
    out = env.query_traffic(QueryEdgeParams(node_a=0, node_b=1))
    assert "speed_factor" in out.blocks[0].text


def test_query_weather():
    from src.server import QueryWeatherParams
    env = make_env()
    out = env.query_weather(QueryWeatherParams(at_minute=60))
    assert "precip" in out.blocks[0].text


def test_assign_and_tick():
    from src.server import AssignParams, TickParams
    env = make_env()
    out = env.assign(AssignParams(request_id="r-0", vehicle_id="v-0",
                                  pickup_position=0, dropoff_position=1))
    assert "r-0" in out.blocks[0].text
    assert out.reward > -1  # valid assign
    assert not out.finished

    out = env.tick(TickParams(minutes=60))
    assert not out.finished


def test_assign_invalid_unknown_request():
    from src.server import AssignParams
    env = make_env()
    out = env.assign(AssignParams(request_id="r-nope", vehicle_id="v-0",
                                  pickup_position=0, dropoff_position=1))
    assert "INVALID" in out.blocks[0].text


def test_defer():
    from src.server import DeferParams
    env = make_env()
    out = env.defer(DeferParams(request_id="r-0", until_minutes=60))
    assert "Deferred" in out.blocks[0].text
    assert env.state.request_status["r-0"] == "deferred"


def test_cancel():
    from src.server import CancelParams
    env = make_env()
    out = env.cancel(CancelParams(request_id="r-0"))
    assert "Cancelled" in out.blocks[0].text
    assert env.state.request_status["r-0"] == "cancelled"
    assert out.reward < 0


def test_add_vehicle():
    from src.server import AddVehicleParams
    env = make_env()
    out = env.add_vehicle(AddVehicleParams(type="small_van_4"))
    assert "Activated" in out.blocks[0].text
    assert out.reward < 0  # large penalty

    # second call blocked
    out2 = env.add_vehicle(AddVehicleParams(type="small_van_4"))
    assert "INVALID" in out2.blocks[0].text


def test_reassign():
    from src.server import AssignParams, ReassignParams, AddVehicleParams
    env = make_env()
    # Need two vehicles: add one first
    env.add_vehicle(AddVehicleParams(type="small_van_4"))
    new_vid = "v-add-1"
    # Assign to v-0 first
    env.assign(AssignParams(request_id="r-0", vehicle_id="v-0",
                            pickup_position=0, dropoff_position=1))
    out = env.reassign(ReassignParams(request_id="r-0", new_vehicle_id=new_vid,
                                      pickup_position=0, dropoff_position=1))
    # May succeed or fail depending on capacity, but should not crash
    assert out.blocks[0].text  # non-empty response


def test_swap_vehicles():
    from src.server import AssignParams, SwapParams, AddVehicleParams
    env = make_env()
    env.add_vehicle(AddVehicleParams(type="small_van_4"))
    new_vid = "v-add-1"
    env.assign(AssignParams(request_id="r-0", vehicle_id="v-0",
                            pickup_position=0, dropoff_position=1))
    # Break v-0
    env.state.vehicle_status["v-0"] = "broken"
    out = env.swap_vehicles(SwapParams(vehicle_id_a="v-0", vehicle_id_b=new_vid))
    assert "Swap" in out.blocks[0].text


def test_submit_plan():
    from src.server import EmptyParams
    env = make_env()
    out = env.submit_plan(EmptyParams())
    assert out.finished


def test_full_episode_assign_tick_submit():
    """Complete happy-path: assign, tick past delivery, submit."""
    from src.server import AssignParams, TickParams, EmptyParams
    env = make_env()

    env.assign(AssignParams(request_id="r-0", vehicle_id="v-0",
                            pickup_position=0, dropoff_position=1))
    # Tick enough to complete the route (depot→node1→node2→depot = ~40 min)
    env.tick(TickParams(minutes=60))
    out = env.submit_plan(EmptyParams())
    assert out.finished
    assert "r-0" in env.state.served_requests


def test_get_prompt():
    env = make_env()
    blocks = env.get_prompt()
    assert len(blocks) == 1
    assert "dispatcher" in blocks[0].text.lower()
