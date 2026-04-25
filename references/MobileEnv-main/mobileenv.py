from typing import List

import gymnasium
import mobile_env  # noqa: F401 — registers gymnasium environments
import numpy as np
from pydantic import BaseModel

from openreward.environments import Environment, JSONObject, ToolOutput, tool, TextBlock


TASK_CONFIGS = [
    {"scenario": "small", "num_stations": 3, "num_users": 5, "num_seeds": 333},
    {"scenario": "medium", "num_stations": 7, "num_users": 15, "num_seeds": 333},
    {"scenario": "large", "num_stations": 13, "num_users": 30, "num_seeds": 334},
]


class MobileEnvTaskSpec(BaseModel):
    id: str
    scenario: str
    seed: int
    num_stations: int
    num_users: int
    max_timesteps: int = 100


class ObserveInput(BaseModel, extra="forbid"):
    """Get the current network state observation."""
    pass


class StepInput(BaseModel, extra="forbid"):
    """Submit connection decisions for UEs and advance the simulation by one timestep.

    actions: A dictionary mapping UE index (as string) to action value.
      - Action 0: No operation (keep current connections)
      - Action 1 to NUM_STATIONS: Toggle connection to that base station (1-indexed).
        Action N toggles connection to BS N-1.
      UEs not included in the dictionary default to action 0 (no change).
    """
    actions: dict[str, int]


class MobileEnv(Environment):

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = MobileEnvTaskSpec.model_validate(task_spec)

        # Create the gymnasium environment with seed in config
        # (mobile-env seeds via config, not via reset())
        env_name = f"mobile-{self.validated.scenario}-central-v0"
        self.gym_env = gymnasium.make(env_name, config={"seed": self.validated.seed})

        # Reset the environment
        self.obs, self.info = self.gym_env.reset()

        # State tracking
        self.current_step = 0
        self.max_steps = self.validated.max_timesteps
        self.cumulative_reward = 0.0
        self.reward_history: List[float] = []
        self.finished = False

        # Cache scenario dimensions for observation parsing
        self.num_stations = self.validated.num_stations
        self.num_users = self.validated.num_users
        # Central handler features: connections (NUM_STATIONS) + snrs (NUM_STATIONS) + utility (1)
        self.obs_per_ue = 2 * self.num_stations + 1

    def _get_reachability(self) -> dict[int, list[bool]]:
        """Check which BS each UE can actually reach (raw SNR above threshold)."""
        core = self.gym_env.unwrapped
        reachability = {}
        # Sort by bs_id to match the ordering used by mobile-env's features()
        stations = sorted(core.stations.values(), key=lambda bs: bs.bs_id)
        for ue_id in sorted(core.users.keys()):
            ue = core.users[ue_id]
            reachability[ue_id] = [
                core.check_connectivity(bs, ue) for bs in stations
            ]
        return reachability

    def _parse_observation(self, obs: np.ndarray) -> str:
        """Convert flat observation array into human-readable text."""
        lines = []
        lines.append(f"=== Network State (Timestep {self.current_step + 1}/{self.max_steps}) ===")
        lines.append("")

        # Get actual reachability from the underlying environment
        reachability = self._get_reachability()

        for ue_idx in range(self.num_users):
            start = ue_idx * self.obs_per_ue

            # Extract sub-vectors
            connections = obs[start : start + self.num_stations]
            snrs = obs[start + self.num_stations : start + 2 * self.num_stations]
            utility = obs[start + 2 * self.num_stations]

            # Determine current connection(s)
            connected_bs = [i for i, c in enumerate(connections) if c > 0.5]
            if connected_bs:
                conn_str = ", ".join(f"BS {bs}" for bs in connected_bs)
            else:
                conn_str = "Not connected"

            # Reachable base stations (where raw SNR exceeds threshold)
            ue_reach = reachability.get(ue_idx, [False] * self.num_stations)
            reachable_bs = [i for i, r in enumerate(ue_reach) if r]
            if reachable_bs:
                reach_str = ", ".join(f"BS {bs}" for bs in reachable_bs)
            else:
                reach_str = "None (out of range)"

            lines.append(f"UE {ue_idx}:")
            lines.append(f"  Connected to: {conn_str}")
            lines.append(f"  Reachable: {reach_str}")
            lines.append(f"  Utility: {utility:.3f}")

            snr_parts = [f"BS {bs_idx}: {snrs[bs_idx]:.3f}" for bs_idx in range(self.num_stations)]
            lines.append(f"  SNR: [{', '.join(snr_parts)}]")
            lines.append("")

        # Summary statistics
        utilities = [
            obs[ue_idx * self.obs_per_ue + 2 * self.num_stations]
            for ue_idx in range(self.num_users)
        ]
        reachable_count = sum(
            1 for ue_idx in range(self.num_users)
            if any(reachability.get(ue_idx, []))
        )
        lines.append("--- Summary ---")
        lines.append(f"Average utility: {np.mean(utilities):.3f}")
        lines.append(f"Min utility: {np.min(utilities):.3f}")
        lines.append(f"Max utility: {np.max(utilities):.3f}")
        lines.append(f"UEs in range: {reachable_count}/{self.num_users}")

        return "\n".join(lines)

    @tool
    async def observe(self, params: ObserveInput) -> ToolOutput:
        """Observe the current state of the wireless network without advancing time."""
        if self.finished:
            return ToolOutput(
                metadata={"error": "Episode already finished"},
                blocks=[TextBlock(text="Error: Episode already finished.")],
                reward=0.0,
                finished=True,
            )

        obs_text = self._parse_observation(self.obs)

        return ToolOutput(
            metadata={
                "timestep": self.current_step + 1,
                "max_timesteps": self.max_steps,
                "cumulative_reward": self.cumulative_reward,
            },
            blocks=[TextBlock(text=obs_text)],
            reward=0.0,
            finished=False,
        )

    @tool
    async def step(self, params: StepInput) -> ToolOutput:
        """Submit connection decisions and advance the simulation by one timestep."""
        if self.finished:
            return ToolOutput(
                metadata={"error": "Episode already finished"},
                blocks=[TextBlock(text="Error: Episode already finished.")],
                reward=0.0,
                finished=True,
            )

        # Convert LLM actions dict to gymnasium MultiDiscrete array
        action_array = np.zeros(self.num_users, dtype=int)  # Default: noop

        for ue_key, action_val in params.actions.items():
            try:
                ue_idx = int(ue_key)
            except ValueError:
                return ToolOutput(
                    metadata={"error": f"Invalid UE index: '{ue_key}'. Must be an integer."},
                    blocks=[TextBlock(text=f"Error: UE index '{ue_key}' is not a valid integer.")],
                    reward=0.0,
                    finished=False,
                )

            if ue_idx < 0 or ue_idx >= self.num_users:
                return ToolOutput(
                    metadata={"error": f"UE index {ue_idx} out of range [0, {self.num_users - 1}]"},
                    blocks=[TextBlock(text=f"Error: UE index {ue_idx} out of range [0, {self.num_users - 1}].")],
                    reward=0.0,
                    finished=False,
                )

            if action_val < 0 or action_val > self.num_stations:
                return ToolOutput(
                    metadata={"error": f"Action {action_val} for UE {ue_idx} out of range [0, {self.num_stations}]"},
                    blocks=[TextBlock(text=f"Error: Action {action_val} for UE {ue_idx} out of range [0, {self.num_stations}].")],
                    reward=0.0,
                    finished=False,
                )

            action_array[ue_idx] = action_val

        # Step the gymnasium environment
        self.obs, reward, terminated, truncated, info = self.gym_env.step(action_array)

        self.current_step += 1
        reward = float(reward)
        self.cumulative_reward += reward
        self.reward_history.append(reward)
        self.finished = terminated or truncated or (self.current_step >= self.max_steps)

        # Build response text
        result_lines = [
            f"Step {self.current_step}/{self.max_steps} completed.",
            f"Step reward: {reward:.4f}",
            f"Cumulative reward: {self.cumulative_reward:.4f}",
            "",
        ]

        if self.finished:
            avg_reward = self.cumulative_reward / self.current_step
            result_lines.append("EPISODE COMPLETE!")
            result_lines.append(f"Total steps: {self.current_step}")
            result_lines.append(f"Final cumulative reward: {self.cumulative_reward:.4f}")
            result_lines.append(f"Average reward per step: {avg_reward:.4f}")
        else:
            result_lines.append("--- New State ---")
            result_lines.append(self._parse_observation(self.obs))

        result_text = "\n".join(result_lines)

        return ToolOutput(
            metadata={
                "step": self.current_step,
                "step_reward": reward,
                "cumulative_reward": self.cumulative_reward,
                "finished": self.finished,
            },
            blocks=[TextBlock(text=result_text)],
            reward=self.cumulative_reward if self.finished else reward,
            finished=self.finished,
        )

    async def get_prompt(self) -> List[TextBlock]:
        obs_text = self._parse_observation(self.obs)

        prompt = f"""You are a wireless network controller managing a mobile network with {self.num_stations} base stations (BS) and {self.num_users} user equipment devices (UE).

OBJECTIVE: Maximize the average Quality of Experience (utility) across all UEs by deciding which base station each UE should connect to at each timestep. The episode runs for {self.max_steps} timesteps.

NETWORK CONFIGURATION:
- Scenario: {self.validated.scenario}
- Base Stations: {self.num_stations} (indexed BS 0 to BS {self.num_stations - 1})
- User Equipment: {self.num_users} (indexed UE 0 to UE {self.num_users - 1})
- Episode Length: {self.max_steps} timesteps

OBSERVATION FORMAT:
At each timestep, you see for each UE:
- Connected to: Which BS(s) the UE is currently connected to
- Reachable: Which BS(s) the UE can actually connect to (signal above threshold).
  IMPORTANT: A UE can ONLY connect to reachable base stations. Connection attempts to
  unreachable BSs will silently fail. If a UE shows "None (out of range)", it cannot
  connect to any BS until it moves closer.
- Utility: Current utility value (range [-1, 1]; higher is better)
- SNR: Relative signal-to-noise ratio to each BS (range [-1, 1]; higher means better
  signal RELATIVE to other BSs, but does NOT guarantee the BS is reachable)

ACTION FORMAT:
Use the 'step' tool with an 'actions' dictionary mapping UE indices (as strings) to actions:
- Action 0: No operation (keep current connections)
- Action 1 to {self.num_stations}: Toggle connection to BS (action - 1).
  If the UE is not connected to that BS, it connects. If already connected, it disconnects.

Example: {{"0": 2, "3": 1}} means toggle UE 0's connection to BS 1, toggle UE 3's connection to BS 0. All other UEs keep their current connections.

STRATEGY TIPS:
- Only attempt to connect UEs to REACHABLE base stations. Unreachable connections will fail.
- UEs showing "None (out of range)" cannot connect — skip them and focus on connectable UEs.
- Among reachable BSs, higher SNR generally means better connection quality.
- Overloading a single BS with too many UEs reduces quality for all connected UEs.
- UEs move over time, so reachability and optimal connections change across timesteps.
- You can call observe() at any time to re-check the state without advancing time.

REWARD: Average utility across all UEs at each timestep. Your goal is to maximize the cumulative reward over all {self.max_steps} timesteps.

CURRENT STATE (Timestep 1/{self.max_steps}):
{obs_text}

Begin by analyzing the current state and making your first connection decision using the step tool."""

        return [TextBlock(text=prompt)]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split != "train":
            raise ValueError(f"Unknown split: {split}. Available splits: ['train']")

        tasks = []
        for config in TASK_CONFIGS:
            for seed in range(config["num_seeds"]):
                tasks.append({
                    "id": f"{config['scenario']}_seed{seed}",
                    "scenario": config["scenario"],
                    "seed": seed,
                    "num_stations": config["num_stations"],
                    "num_users": config["num_users"],
                    "max_timesteps": 100,
                })
        return tasks

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]
