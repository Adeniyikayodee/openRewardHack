import pytest
import numpy as np

from mobileenv import MobileEnv, StepInput, ObserveInput


def _make_task(scenario="small", seed=0):
    """Helper to create a task spec."""
    configs = {
        "small": {"num_stations": 3, "num_users": 5},
        "medium": {"num_stations": 7, "num_users": 15},
        "large": {"num_stations": 13, "num_users": 30},
    }
    c = configs[scenario]
    return {
        "id": f"{scenario}_seed{seed}",
        "scenario": scenario,
        "seed": seed,
        "num_stations": c["num_stations"],
        "num_users": c["num_users"],
        "max_timesteps": 100,
    }


# --- Initialization tests ---


@pytest.mark.parametrize(
    "scenario,num_stations,num_users",
    [
        ("small", 3, 5),
        ("medium", 7, 15),
        ("large", 13, 30),
    ],
)
def test_initialization(scenario, num_stations, num_users):
    """Environment initializes correctly for each scenario."""
    env = MobileEnv(task_spec=_make_task(scenario))
    assert env.num_stations == num_stations
    assert env.num_users == num_users
    assert env.current_step == 0
    assert env.cumulative_reward == 0.0
    assert not env.finished
    # Observation should be the right size
    expected_obs_size = num_users * (2 * num_stations + 1)
    assert len(env.obs) == expected_obs_size


def test_deterministic_reset():
    """Same seed produces the same initial observation."""
    env1 = MobileEnv(task_spec=_make_task("small", seed=42))
    env2 = MobileEnv(task_spec=_make_task("small", seed=42))
    np.testing.assert_array_equal(env1.obs, env2.obs)


def test_different_seeds_differ():
    """Different seeds produce different initial observations."""
    env1 = MobileEnv(task_spec=_make_task("small", seed=0))
    env2 = MobileEnv(task_spec=_make_task("small", seed=1))
    assert not np.array_equal(env1.obs, env2.obs)


# --- Observe tests ---


@pytest.mark.asyncio
async def test_observe_no_side_effects():
    """Observe returns valid output without advancing state."""
    env = MobileEnv(task_spec=_make_task("small"))
    result = await env.observe(ObserveInput())
    assert result.reward == 0.0
    assert not result.finished
    assert env.current_step == 0
    assert "Network State" in result.blocks[0].text


@pytest.mark.asyncio
async def test_observe_multiple_times():
    """Calling observe multiple times doesn't change state."""
    env = MobileEnv(task_spec=_make_task("small"))
    r1 = await env.observe(ObserveInput())
    r2 = await env.observe(ObserveInput())
    assert r1.blocks[0].text == r2.blocks[0].text
    assert env.current_step == 0


# --- Step tests ---


@pytest.mark.asyncio
async def test_step_advances_state():
    """Step advances the timestep counter."""
    env = MobileEnv(task_spec=_make_task("small"))
    result = await env.step(StepInput(actions={}))  # All noop
    assert env.current_step == 1
    assert not result.finished
    assert "Step 1/100 completed" in result.blocks[0].text


@pytest.mark.asyncio
async def test_step_with_actions():
    """Step accepts valid actions and returns a reward."""
    env = MobileEnv(task_spec=_make_task("small"))
    result = await env.step(StepInput(actions={"0": 1, "2": 2}))
    assert env.current_step == 1
    assert result.metadata["step_reward"] is not None
    assert isinstance(result.reward, float)


@pytest.mark.asyncio
async def test_step_noop_returns_reward():
    """Even noop actions produce a reward (average utility)."""
    env = MobileEnv(task_spec=_make_task("small"))
    result = await env.step(StepInput(actions={}))
    assert isinstance(result.reward, float)


@pytest.mark.asyncio
async def test_step_cumulative_reward():
    """Cumulative reward tracks correctly across steps."""
    env = MobileEnv(task_spec=_make_task("small"))
    total = 0.0
    for _ in range(5):
        result = await env.step(StepInput(actions={}))
        total += result.metadata["step_reward"]
    assert abs(env.cumulative_reward - total) < 1e-10


# --- Validation tests ---


@pytest.mark.asyncio
async def test_invalid_ue_index():
    """Invalid UE index is rejected."""
    env = MobileEnv(task_spec=_make_task("small"))
    result = await env.step(StepInput(actions={"99": 1}))
    assert "error" in result.metadata
    assert env.current_step == 0  # State should not advance


@pytest.mark.asyncio
async def test_invalid_action_value():
    """Action value out of range is rejected."""
    env = MobileEnv(task_spec=_make_task("small"))
    result = await env.step(StepInput(actions={"0": 99}))
    assert "error" in result.metadata
    assert env.current_step == 0


@pytest.mark.asyncio
async def test_negative_action():
    """Negative action value is rejected."""
    env = MobileEnv(task_spec=_make_task("small"))
    result = await env.step(StepInput(actions={"0": -1}))
    assert "error" in result.metadata


@pytest.mark.asyncio
async def test_non_integer_ue_key():
    """Non-integer UE key is rejected."""
    env = MobileEnv(task_spec=_make_task("small"))
    result = await env.step(StepInput(actions={"abc": 1}))
    assert "error" in result.metadata


# --- Episode completion tests ---


@pytest.mark.asyncio
async def test_episode_completion():
    """Episode finishes after max_timesteps."""
    task = _make_task("small")
    task["max_timesteps"] = 10
    env = MobileEnv(task_spec=task)

    for i in range(10):
        result = await env.step(StepInput(actions={}))

    assert result.finished
    assert env.finished
    assert "EPISODE COMPLETE" in result.blocks[0].text
    assert result.metadata["finished"] is True


@pytest.mark.asyncio
async def test_step_after_finished():
    """Stepping after episode is done returns error."""
    task = _make_task("small")
    task["max_timesteps"] = 2
    env = MobileEnv(task_spec=task)

    await env.step(StepInput(actions={}))
    await env.step(StepInput(actions={}))  # Should finish
    result = await env.step(StepInput(actions={}))  # After finished

    assert "error" in result.metadata
    assert result.finished


@pytest.mark.asyncio
async def test_final_reward_is_cumulative():
    """On the final step, reward equals cumulative reward."""
    task = _make_task("small")
    task["max_timesteps"] = 5
    env = MobileEnv(task_spec=task)

    for _ in range(5):
        result = await env.step(StepInput(actions={}))

    assert result.finished
    assert abs(result.reward - env.cumulative_reward) < 1e-10


# --- Task listing tests ---


def test_list_splits():
    """list_splits returns correct splits."""
    assert MobileEnv.list_splits() == ["train"]


def test_list_tasks_count():
    """list_tasks returns 1000 tasks for train split."""
    tasks = MobileEnv.list_tasks("train")
    assert len(tasks) == 1000


def test_list_tasks_structure():
    """Each task has the required fields."""
    tasks = MobileEnv.list_tasks("train")
    for task in tasks[:10]:
        assert "id" in task
        assert "scenario" in task
        assert "seed" in task
        assert "num_stations" in task
        assert "num_users" in task
        assert "max_timesteps" in task


def test_list_tasks_unique_ids():
    """All task IDs are unique."""
    tasks = MobileEnv.list_tasks("train")
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids))


def test_list_tasks_invalid_split():
    """Invalid split raises ValueError."""
    with pytest.raises(ValueError):
        MobileEnv.list_tasks("invalid")


def test_list_tasks_scenarios():
    """Tasks cover all three scenarios."""
    tasks = MobileEnv.list_tasks("train")
    scenarios = set(t["scenario"] for t in tasks)
    assert scenarios == {"small", "medium", "large"}


# --- Prompt tests ---


@pytest.mark.asyncio
async def test_get_prompt():
    """get_prompt returns a valid prompt with initial observation."""
    env = MobileEnv(task_spec=_make_task("small"))
    prompt = await env.get_prompt()
    assert len(prompt) == 1
    text = prompt[0].text
    assert "wireless network controller" in text
    assert "3 base stations" in text
    assert "5 user equipment" in text
    assert "Network State" in text
