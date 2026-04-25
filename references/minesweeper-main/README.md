# Minesweeper

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/Minesweeper)

## Description

**Minesweeper** is an environment for evaluating agents on spatial reasoning, probabilistic inference, and strategic exploration. This environment wraps the Minesweeper implementation from [TextArena](https://github.com/LeonGuertler/TextArena), a framework for text-based game environments.

## Capabilities

- Probabilistic reasoning under uncertainty
- Spatial pattern recognition and inference
- Strategic decision-making with risk assessment
- Constraint satisfaction from partial information

## Compute Requirements

Minesweeper does not require a sandbox. It has minimal compute requirements.

## License

[MIT](https://github.com/LeonGuertler/TextArena/blob/main/LICENSE).

## Tasks

There are two splits: train (600 tasks) and test (600 tasks). Each split contains 50 tasks across each of 12 variants:

- **Minesweeper-v0**
- **Minesweeper-v0-train**
- **Minesweeper-v0-raw**
- **Minesweeper-v0-small**
- **Minesweeper-v0-small-train**
- **Minesweeper-v0-small-raw**
- **Minesweeper-v0-medium**
- **Minesweeper-v0-medium-train**
- **Minesweeper-v0-medium-raw**
- **Minesweeper-v0-hard**
- **Minesweeper-v0-hard-train**
- **Minesweeper-v0-hard-raw**

Each task is seeded for reproducibility.

## Reward Structure

This is a sparse reward environment. Rewards are mapped from TextArena's native range of {-1, 0, 1} to {0.0, 0.5, 1.0} via `(raw + 1) / 2`.

We do not use LLM graders for this environment; reward is determined programmatically.

## Data

Game state is generated procedurally by the TextArena engine using seeded randomness. No external data files are required.

## Tools

Agents are given a single tool:

- `reveal_cell(row, column)`: Reveal a cell on the minesweeper grid at the given row and column (0-indexed). Numbers show adjacent mine count. Hitting a mine ends the game.

## Time Horizon

Minesweeper is a multi-turn environment.

## Environment Difficulty

This environment ranges from moderate to very challenging depending on the variant. Board size and mine density increase from small to hard variants, requiring increasingly sophisticated probabilistic reasoning.

## Other Environment Requirements

There are no further environment requirements; Minesweeper works out of the box without any secrets or API keys.

## Safety

Agents in Minesweeper interact only with a puzzle game and have no access to external systems, the internet, or sensitive data. The environment does not present safety risks.

## Citations

```bibtex
@software{textarena2024,
  author    = {Guertler, Leon and Banting, Wilfried and Pignatelli, Eduardo},
  title     = {TextArena},
  year      = {2024},
  publisher = {GitHub},
  url       = {https://github.com/LeonGuertler/TextArena}
}
```
