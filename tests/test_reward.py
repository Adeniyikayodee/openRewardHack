from src.reward import (terminal_reward, completion_reward,
                        SHAPE_PER_ACTION_DECAY)


class _S:
    pass


def test_completion_priority_scaling():
    r1 = completion_reward({"priority": 1}, True)
    r5 = completion_reward({"priority": 5}, True)
    assert r5 == 5 * r1


def test_completion_late_partial():
    r_on  = completion_reward({"priority": 3}, True)
    r_off = completion_reward({"priority": 3}, False)
    assert r_off == 0.5 * r_on


def test_terminal_zero_served_negative():
    s = _S()
    s.served_requests = set()
    s.realized_cost_km = 0
    s.total_actions = 5
    task = {"requests": [{}, {}], "or_tools_baseline_cost": 100}
    assert terminal_reward(s, task) == -1.0


def test_terminal_full_coverage_positive():
    s = _S()
    s.served_requests = {"r-0", "r-1"}
    s.realized_cost_km = 100
    s.total_actions = 8
    task = {"requests": [{"id": "r-0"}, {"id": "r-1"}],
            "or_tools_baseline_cost": 100}
    r = terminal_reward(s, task)
    assert r > 1.0  # coverage 1 + efficiency 0.5 + speed_bonus


def test_shape_per_action_decay_negative():
    assert SHAPE_PER_ACTION_DECAY < 0
