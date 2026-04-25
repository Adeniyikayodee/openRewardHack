from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pandas as pd
from datasets import Dataset, load_dataset
from pydantic import BaseModel, Field

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool


# Path handling: /orwd_data (production) vs local (dev)
if Path("/orwd_data/").exists():
    DATA_PATH = Path("/orwd_data/")
else:
    DATA_PATH = Path(__file__).parent


# Load dataset at module level
if (DATA_PATH / "data.parquet").exists():
    # Production or local dev: load from parquet
    tasks_df = pd.read_parquet(DATA_PATH / "data.parquet")
    # Critical: handle NaN values and drop index column
    if '__index_level_0__' in tasks_df.columns:
        tasks_df = tasks_df.drop('__index_level_0__', axis=1)
    tasks_df = tasks_df.fillna('')  # Prevents 500 errors from NaN in JSON
    tasks_list = tasks_df.to_dict(orient="records")
    # Convert numpy arrays to Python lists for JSON serialization
    for task in tasks_list:
        if 'answer_nodes' in task:
            task['answer_nodes'] = list(task['answer_nodes'])
else:
    # Development fallback: load from HuggingFace (requires internet)
    ds = load_dataset("openai/graphwalks", split="train")
    tasks_list = [dict(ds[i]) for i in range(len(ds))]


class SubmitAnswerInput(BaseModel):
    """Parameter schema for submit_answer tool."""

    answer: str = Field(
        ...,
        description=(
            "Your final answer in the exact format: 'Final Answer: [node1, node2, ...]' "
            "or 'Final Answer: []' for empty set. Node names must match exactly."
        )
    )


class GraphWalks(Environment):
    """GraphWalks graph traversal reasoning environment.

    Tasks involve performing graph operations on directed graphs:
    - parents: Find all nodes with edges pointing TO a target node
    - bfs: Breadth-first search from a node at a specific depth
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        # Store task data
        self.task_data = dict(task_spec)
        self.prompt_text = str(self.task_data.get("prompt", ""))
        # Convert answer_nodes to list (may be numpy array from parquet)
        answer_nodes_raw = self.task_data.get("answer_nodes", [])
        self.answer_nodes = list(answer_nodes_raw) if hasattr(answer_nodes_raw, '__iter__') else []
        self.problem_type = str(self.task_data.get("problem_type", ""))

    @classmethod
    def list_splits(cls) -> list[str]:
        """Return available splits: parents, bfs, and test (all)."""
        return ["parents", "bfs", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """Return tasks filtered by split."""

        # Filter by split
        if split == "parents":
            return [t for t in tasks_list if t.get("problem_type") == "parents"]
        elif split == "bfs":
            return [t for t in tasks_list if t.get("problem_type") == "bfs"]
        elif split == "test":
            return tasks_list  # Return all tasks
        else:
            return []

    async def get_prompt(self) -> List[TextBlock]:
        """Return the complete task prompt.

        The dataset already contains complete prompts with:
        - Instructions and rules
        - Graph edge list
        - Operation to perform
        - Format requirements
        """
        return [TextBlock(text=self.prompt_text)]

    @tool
    async def submit_answer(self, params: SubmitAnswerInput) -> ToolOutput:
        """Submit your final answer for the graph operation.

        Expected format: "Final Answer: [node1, node2, ...]"
        For empty set: "Final Answer: []"

        Args:
            params: SubmitAnswerInput with answer string

        Returns:
            ToolOutput with feedback, metadata, reward, and finished=True
        """

        try:
            # Parse the answer using regex
            # Pattern: "Final Answer: [...]" with optional whitespace
            pattern = r"Final\s+Answer:\s*\[(.*?)\]"
            match = re.search(pattern, params.answer, re.IGNORECASE | re.DOTALL)

            if not match:
                return ToolOutput(
                    blocks=[TextBlock(
                        text=(
                            "❌ Invalid answer format. Please use the exact format:\n"
                            "Final Answer: [node1, node2, ...]\n"
                            "or\n"
                            "Final Answer: []"
                        )
                    )],
                    metadata={
                        "error": "invalid_format",
                        "submitted": params.answer,
                        "problem_type": self.problem_type,
                    },
                    reward=0.0,
                    finished=True,
                )

            # Extract node list from brackets
            nodes_str = match.group(1).strip()

            # Parse nodes: split by comma, strip whitespace
            if nodes_str == "":
                submitted_nodes = []
            else:
                # Handle both "node1, node2" and "node1,node2"
                submitted_nodes = [n.strip() for n in nodes_str.split(",")]
                # Remove empty strings from extra commas
                submitted_nodes = [n for n in submitted_nodes if n]
                # Strip quotes from nodes (in case agent uses Python list format)
                submitted_nodes = [n.strip("'\"") for n in submitted_nodes]

            # Convert to sets for order-independent comparison
            submitted_set = set(submitted_nodes)
            correct_set = set(self.answer_nodes)

            # Check correctness
            correct = (submitted_set == correct_set)
            reward = 1.0 if correct else 0.0

            # Generate feedback
            if correct:
                feedback_text = (
                    f"✅ Correct! The {self.problem_type} operation returned {len(correct_set)} node(s)."
                )
            else:
                # Provide detailed feedback
                missing = correct_set - submitted_set
                extra = submitted_set - correct_set

                feedback_parts = ["❌ Incorrect."]

                if missing:
                    feedback_parts.append(
                        f"\nMissing nodes: {sorted(missing)}"
                    )
                if extra:
                    feedback_parts.append(
                        f"\nExtra nodes: {sorted(extra)}"
                    )

                feedback_parts.append(
                    f"\n\nYour answer: {sorted(submitted_nodes) if submitted_nodes else '[]'}"
                )
                feedback_parts.append(
                    f"\nCorrect answer: {sorted(self.answer_nodes) if self.answer_nodes else '[]'}"
                )

                feedback_text = "".join(feedback_parts)

            return ToolOutput(
                blocks=[TextBlock(text=feedback_text)],
                metadata={
                    "submitted_nodes": sorted(submitted_nodes) if submitted_nodes else [],
                    "correct_nodes": sorted(self.answer_nodes) if self.answer_nodes else [],
                    "correct": correct,
                    "problem_type": self.problem_type,
                    "num_submitted": len(submitted_nodes),
                    "num_correct": len(self.answer_nodes),
                },
                reward=reward,
                finished=True,
            )

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            return ToolOutput(
                blocks=[TextBlock(
                    text=f"❌ Error processing answer: {str(e)}"
                )],
                metadata={
                    "error": error_msg,
                    "submitted": params.answer,
                },
                reward=0.0,
                finished=True,
            )
