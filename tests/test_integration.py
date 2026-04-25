"""End-to-end smoke tests against a real running server."""
from openreward import OpenReward


def _client():
    c = OpenReward()
    return c.environments.get(name="LondonDynamicRouting",
                              base_url="http://localhost:8080")


def test_server_lists_splits(server):
    env = _client()
    splits = env.list_splits()
    names = {s.name for s in splits}
    assert names == {"tutorial", "train", "test"}


def test_server_lists_tutorial_tasks(server):
    env = _client()
    tasks = env.list_tasks(split="tutorial")
    assert len(tasks) >= 1


def test_full_episode_trivial(server):
    env = _client()
    tasks = env.list_tasks(split="tutorial")
    trivial = next(t for t in tasks if t.task_spec["id"] == "trivial_task")
    with env.session(task=trivial) as s:
        s.get_prompt()
        out = s.call_tool("list_pending_requests", {})
        assert "r-0" in out.blocks[0].text
        out = s.call_tool("assign", {"request_id": "r-0",
                                     "vehicle_id": "v-0",
                                     "pickup_position": 0,
                                     "dropoff_position": 1})
        assert out.reward > 0
        out = s.call_tool("tick", {"minutes": 60})
        assert "r-0" in out.blocks[0].text or out.reward > 0
        out = s.call_tool("submit_plan", {})
        assert out.finished
