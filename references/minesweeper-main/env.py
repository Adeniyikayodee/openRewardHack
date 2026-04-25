import textarena as ta
import re
from typing import List
from pydantic import BaseModel
from openreward.environments import Environment, JSONObject, ToolOutput, TextBlock, tool


class TaskSpec(BaseModel):
    id: str
    env_id: str
    seed: int
    variant: str = ""


class RevealCellParams(BaseModel, extra="forbid"):
    row: int
    column: int


class MinesweeperEnvironment(Environment):
    GAME_NAME = "Minesweeper"
    VARIANTS = [
        "Minesweeper-v0",
        "Minesweeper-v0-train",
        "Minesweeper-v0-raw",
        "Minesweeper-v0-small",
        "Minesweeper-v0-small-train",
        "Minesweeper-v0-small-raw",
        "Minesweeper-v0-medium",
        "Minesweeper-v0-medium-train",
        "Minesweeper-v0-medium-raw",
        "Minesweeper-v0-hard",
        "Minesweeper-v0-hard-train",
        "Minesweeper-v0-hard-raw",
    ]
    NUM_TASKS_PER_VARIANT = 50

    def __init__(self, task_spec, secrets={}):
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)
        self.ta_env = ta.make(env_id=self.config.env_id)
        self.game_done = False
        self.turn_count = 0

    @classmethod
    def list_splits(cls):
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split):
        tasks = []
        for variant_id in cls.VARIANTS:
            for seed_idx in range(cls.NUM_TASKS_PER_VARIANT):
                seed = seed_idx if split == "train" else seed_idx + 10000
                tasks.append({
                    "id": f"{variant_id}_seed{seed}",
                    "env_id": variant_id,
                    "seed": seed,
                    "variant": variant_id
                })
        return tasks

    def _format_observation(self, observation) -> str:
        if isinstance(observation, str):
            match = None
            for m in re.finditer(r'^\[(?!GAME\])[^\]]+\].*$', observation, re.MULTILINE):
                match = m
            if match:
                return observation[match.end():].lstrip('\n')
            return observation
        if isinstance(observation, list):
            if not observation:
                return ""
            last = observation[-1]
            if isinstance(last, tuple) and len(last) >= 2:
                return str(last[1])
            return str(last)
        return str(observation)

    def _map_reward(self, raw):
        """Map TextArena rewards (typically -1 to 1) to 0-1 range"""
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))

    async def get_prompt(self):
        self.ta_env.reset(num_players=1, seed=self.config.seed)
        _, obs = self.ta_env.get_observation()
        obs_text = self._format_observation(obs)

        prompt = f"""You are playing Minesweeper.

{obs_text}

Use the reveal_cell tool to reveal cells on the grid.
Provide row and column indices (0-indexed). Avoid mines! Numbers indicate adjacent mine count."""

        return [TextBlock(text=prompt)]

    @tool
    async def reveal_cell(self, params: RevealCellParams) -> ToolOutput:
        """Reveal a cell on the minesweeper grid at the given row and column (0-indexed). Numbers show adjacent mine count. Hitting a mine ends the game."""
        if self.game_done:
            return ToolOutput(
                blocks=[TextBlock(text="Game is already over.")],
                metadata={"error": "game_finished"},
                reward=0.0,
                finished=True
            )

        action = f"[{params.row} {params.column}]"
        done, info = self.ta_env.step(action=action)
        self.turn_count += 1

        if done:
            self.game_done = True
            rewards, game_info = self.ta_env.close()

            # Extract reward for player 0
            raw = rewards.get(0, 0.0) if isinstance(rewards, dict) else float(rewards)
            reward = self._map_reward(raw)

            # Extract game info
            reason = ""
            if isinstance(game_info, dict) and 0 in game_info:
                reason = game_info[0].get("reason", "")

            summary = f"Game Over! Reward: {reward:.2f}"
            if reason:
                summary += f"\n{reason}"

            return ToolOutput(
                blocks=[TextBlock(text=summary)],
                metadata={"turn": self.turn_count, "reward": reward},
                reward=reward,
                finished=True
            )

        # Game continues
        _, obs = self.ta_env.get_observation()
        obs_text = self._format_observation(obs)

        return ToolOutput(
            blocks=[TextBlock(text=obs_text)],
            metadata={"turn": self.turn_count},
            reward=0.0,
            finished=False
        )
