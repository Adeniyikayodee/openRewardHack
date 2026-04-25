# MobileEnv

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/MobileEnv)

## Description

MobileEnv is an environment for evaluating language model agents on wireless mobile network coordination tasks. Agents act as centralized network controllers, deciding which base station each user equipment device should connect to at each timestep to maximize overall Quality of Experience (utility). The environment wraps the [mobile-env](https://github.com/stefanbschneider/mobile-env) Gymnasium simulation.

## Capabilities

- Analyzing wireless network state (connections, signal strength, utility)
- Sequential resource allocation decisions over 100 timesteps
- Load balancing across multiple base stations
- Adapting to dynamic user movement and changing signal conditions

## Compute Requirements

MobileEnv runs a lightweight procedural simulation. It does not require a sandbox and has minimal compute requirements.

## License

[MIT](https://opensource.org/license/mit).

## Tasks

There are 1000 tasks in a single `train` split across 3 scenario sizes with varying random seeds:

| Scenario | Base Stations | User Equipment | Tasks |
|----------|--------------|----------------|-------|
| Small    | 3            | 5              | 333   |
| Medium   | 7            | 15             | 333   |
| Large    | 13           | 30             | 334   |

Each task runs for 100 timesteps. At each timestep, the agent observes the network state and decides which base station each UE should connect to. Different random seeds produce different user movement trajectories and initial placements.

## Reward Structure

This is a dense, verifiable reward environment. After each timestep, the reward is the average utility across all user equipment devices:

$$\text{reward}_t = \frac{1}{N_{UE}} \sum_{i=1}^{N_{UE}} u_i$$

where $u_i \in [-1, 1]$ is the bounded log utility of UE $i$. The final reward at episode completion is the cumulative sum of all step rewards.

We do not use LLM graders for this task.

## Data

MobileEnv uses procedurally generated simulation data from the mobile-env package. No external datasets are required. Different random seeds produce different user placements and movement patterns.

## Tools

Agents are given two tools:

- `observe`: View the current network state without advancing time. Returns each UE's current connections, reachability (which BSs have sufficient signal for a connection), relative signal-to-noise ratios, and utility.
- `step`: Submit connection decisions for all UEs and advance the simulation by one timestep. Returns the step reward and new network state.

Note: SNR values shown are relative per-UE (best BS = 1.0), not absolute signal strength. The "Reachable" field indicates which BSs a UE can actually connect to based on the raw signal threshold. Connection attempts to unreachable BSs silently fail. Reachability changes over time as UEs move.

## Time Horizon

MobileEnv is a multi-turn environment with 100 sequential decisions per task (one per timestep), each requiring one `step` tool call. The agent may optionally call `observe` at any point.

## Other Environment Requirements

There are no further environment requirements; MobileEnv works out of the box without any secrets.

## Safety

Agents in MobileEnv optimize wireless network connections in a simulated environment. The environment does not present safety risks, as agents only interact with an artificial simulation with no real-world network effects.

## Citations

```bibtex
@inproceedings{schneider2022mobileenv,
  author    = {Schneider, Stefan and Werner, Stefan and Khalili, Ramin and Hecker, Artur and Karl, Holger},
  title     = {mobile-env: An Open Platform for Reinforcement Learning in Wireless Mobile Networks},
  booktitle = {IEEE/IFIP Network Operations and Management Symposium (NOMS)},
  year      = {2022},
  publisher = {IEEE}
}
```
