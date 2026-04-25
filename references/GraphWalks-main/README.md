# GraphWalks

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/graphwalks) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/openai/graphwalks)

## Description

GraphWalks is a single-turn environment that tests an agent's ability to perform graph operations on directed graphs presented as edge lists. Each task provides a directed graph and asks the agent to execute a specific operation -- either finding the parent nodes of a target node or performing breadth-first search (BFS) from a given node at a specified depth. The agent must return the correct set of nodes as its answer.

## Capabilities

- Graph reasoning over directed graphs
- Directed graph traversal (parent-finding and breadth-first search)
- Set-based answer validation with exact node name matching

## Compute Requirements

This is a single-turn environment with no sandbox.

## License

[MIT](https://opensource.org/licenses/MIT).

## Tasks

There are 1,150 total tasks across three splits:

- **parents** (600 tasks): Find all nodes with edges pointing TO a target node in the directed graph.
- **bfs** (550 tasks): Perform breadth-first search from a given node at a specified depth and return the set of reachable nodes.
- **test** (1,150 tasks): All tasks combined from both problem types.

Each task provides the agent with a complete prompt containing instructions, a directed graph as an edge list (e.g., `node_a -> node_b`), the operation to perform, and the expected output format.

## Reward Structure

GraphWalks uses binary deterministic reward (1.0 or 0.0). The agent's submitted set of nodes is compared against the expected answer using order-independent set comparison. If the submitted set exactly matches the correct set, the reward is 1.0; otherwise it is 0.0. No partial credit is given, and no LLM grader is used.

## Data

The dataset is sourced from the HuggingFace [`openai/graphwalks`](https://huggingface.co/datasets/openai/graphwalks) dataset and stored on the OpenReward platform as `data.parquet`. The parquet file contains the following columns:

| Column | Description |
|--------|-------------|
| `prompt` | Complete task prompt with graph edge list and operation instructions |
| `answer_nodes` | List of correct node names (ground truth) |
| `problem_type` | Either `"parents"` or `"bfs"` |
| `prompt_chars` | Character count of the prompt |

## Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `submit_answer` | `answer: str` | Submit answer in the format `"Final Answer: [node1, node2, ...]"`. For empty results, use `"Final Answer: []"`. Ends the episode. |

## Time Horizon

GraphWalks is a single-turn environment. The agent receives a graph and an operation in the prompt, reasons about the answer, and submits a single response via the `submit_answer` tool. There is exactly one tool call per task.

## Environment Difficulty

[Put environment difficulty here]

## Other Environment Requirements

There are no further environment requirements. GraphWalks works out of the box with the OpenReward endpoint without any external API keys or secrets.

## Safety

GraphWalks does not present safety concerns. The agent only processes graph structures presented as text edge lists and submits text answers. There is no access to external systems, no code execution, and no interaction with real-world data.

## Citations

```bibtex
@dataset{openai_graphwalks,
  author    = {OpenAI},
  title     = {GraphWalks},
  year      = {2025},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/datasets/openai/graphwalks}
}
```
