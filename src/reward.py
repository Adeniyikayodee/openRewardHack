"""All shaping constants and the terminal reward live here.

Per-step rewards are paid by the tool that generated the action.
Per-completion rewards are paid by tick() when a dropoff stop is realized.
Terminal reward is paid when the episode ends (submit_plan or horizon)."""

# ─── Per-step shaping (continuous feedback) ───────────────────────────────
SHAPE_VALID_ASSIGN     =  0.05    # × (1 - marginal/max_marginal)
SHAPE_INVALID_ACTION   = -0.02
SHAPE_DUPLICATE        = -0.05
SHAPE_REASSIGN_BENEFIT =  0.03    # if cost decreased
SHAPE_REASSIGN_NEUTRAL = -0.01
SHAPE_DEFER            =  0.0
SHAPE_CANCEL_PER_PRIO  = -0.10
SHAPE_ADD_VEHICLE      = -0.50
SHAPE_SWAP_BROKEN      =  0.20
SHAPE_QUERY_SPAM       = -0.001
SHAPE_PER_ACTION_DECAY = -0.005   # trajectory-length penalty (incentivize speed)

# ─── Per-completion (paid at dropoff) ─────────────────────────────────────
COMPLETION_BASE = 0.10


def completion_reward(request, on_time: bool) -> float:
    base = COMPLETION_BASE * request["priority"]
    return base if on_time else 0.5 * base


# ─── Terminal ─────────────────────────────────────────────────────────────
def terminal_reward(state, task) -> float:
    n_total = len(task["requests"])
    n_served = len(state.served_requests)
    coverage = n_served / max(1, n_total)

    # Cost-efficiency vs OR-Tools baseline
    agent_km = state.realized_cost_km
    optimal_km = task.get("or_tools_baseline_cost", float("inf"))
    if agent_km <= 0 or optimal_km == float("inf") or optimal_km <= 0:
        cost_ratio = 0.0
    else:
        cost_ratio = min(1.0, optimal_km / agent_km)

    # Speed bonus: finishing in fewer than 5 actions per request earns up to +0.2
    budget = max(20, 5 * n_total)
    speed_bonus = max(0.0, 0.2 * (1 - state.total_actions / budget))

    if n_served == 0:
        return -1.0

    coverage_term = 1.0 * coverage
    efficiency_term = 0.5 * cost_ratio
    return coverage_term + efficiency_term + speed_bonus
