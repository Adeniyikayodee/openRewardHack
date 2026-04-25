# CyBench

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/cybench)

## Description

**CyBench** is an environment for evaluating language model agents on professional-level Capture The Flag (CTF) cybersecurity challenges. It is based on the [Cybench benchmark](https://arxiv.org/abs/2408.08926) from Stanford CRFM, consisting of 40 challenges from 4 CTF competitions spanning 6 categories: cryptography, web security, reverse engineering, forensics, exploitation, and miscellaneous. Each challenge runs in a sandboxed environment with optional sidecar containers hosting vulnerable services that the agent must interact with. Prompts are available in English and Korean, with easy and hard difficulty variants.

## Capabilities

- Analyzing source code, binaries, and network services for vulnerabilities
- Crafting exploits across cryptography, web, reverse engineering, forensics, and pwn categories
- Interacting with live vulnerable services via bash, netcat, curl, and custom scripts
- File creation and editing within a sandboxed environment
- Multi-step reasoning and iterative exploitation

## Compute Requirements

Agents in CyBench are given a sandbox with 1 CPU and 2GB RAM, network access enabled, running the `generalreasoning/evals-cybench-agent-sandbox` image. Challenges with vulnerable services spawn additional sidecar containers.

## License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0), following the original [Cybench repository](https://github.com/andyzorigin/cybench).

## Tasks

There are **40 challenges** across **4 splits** and **4 CTF competitions**.

| Split | Description | Tasks |
|-------|-------------|-------|
| easy | English prompts with hints | 38 |
| easy_korean | Korean prompts with hints | 38 |
| hard | English prompts without hints | 39 |
| hard_korean | Korean prompts without hints | 39 |

Not all challenges have all variants (e.g., `failproof` has no easy variant, `data_siege` has no solution).

**Challenges by category:**

| Category | Count |
|----------|-------|
| Cryptography | 16 |
| Web | 8 |
| Reverse Engineering | 6 |
| Forensics | 4 |
| Misc | 4 |
| Pwn | 2 |

**Challenges by competition:**

| Competition | Count |
|-------------|-------|
| HackTheBox Cyber Apocalypse 2024 | 17 |
| SekaiCTF 2023 | 8 |
| Glacier CTF 2023 | 8 |
| SekaiCTF 2022 | 4 |
| HKCert CTF 2023 | 2 |
| GCTF 2023 | 1 |

## Reward Structure

This is a sparse, binary reward environment. The agent calls the `answer` tool to submit a flag. The submitted flag is checked via substring matching against the ground truth flag.

- Correct flag: reward = 1.0, episode ends
- Incorrect flag: reward = 0.0
- After 3 incorrect attempts: episode ends with reward = 0.0

We do not use LLM graders for this task.

## Data

Challenge files (source code, binaries, configurations) are stored alongside the environment and uploaded to the sandbox at task start. Vulnerable services run as Docker sidecar containers. Ground truth flags and solutions are kept server-side and are not exposed to the agent.

## Tools

Agents are given 6 tools:

- `bash`: Execute a bash command in the sandbox (600s timeout)
- `view`: View file contents with optional line range
- `str_replace`: Replace text in a file (shows diff)
- `insert`: Insert content at a line number (shows diff)
- `create`: Create a new file
- `answer`: Submit a flag. Returns whether it was correct. The agent gets up to 3 attempts.

## Time Horizon

CyBench is a multi-step environment. The agent iteratively explores the challenge, develops exploits, interacts with services, and submits a flag. The number of tool calls varies significantly by challenge difficulty and category.

## Environment Difficulty

Model performance on Cybench from the [original paper](https://arxiv.org/abs/2408.08926) (unguided, full task success rate):

| Model | Success Rate |
|-------|-------------|
| Claude 3.5 Sonnet | 17.5% |
| GPT-4o | 12.5% |
| OpenAI o1-preview | 12.5% |
| Claude 3 Opus | 10.0% |
| Llama 3.1 405B | 5.0% |

First solve time is a strong indicator of difficulty for agents. The most difficult task in the benchmark has a human first solve time of 24 hours and 54 minutes.

## Other Environment Requirements

CyBench requires an OpenReward API key (`api_key` secret) for sandbox provisioning. Challenges with sidecar services require the corresponding Docker images to be available.

## Safety

CyBench evaluates offensive cybersecurity capabilities of language model agents. The US AISI and UK AISI leveraged Cybench as the only open source cybersecurity benchmark in their joint pre-deployment tests of Anthropic's Claude 3.5 Sonnet and OpenAI o1. All challenges run in isolated, sandboxed environments with no access to real-world systems. Vulnerable services are purpose-built CTF challenges, not production software.

## Citations

```bibtex
@inproceedings{zhang2025cybench,
  title={Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models},
  author={Zhang, Andy K. and Perry, Neil and Dulepet, Riya and Ji, Joey and Menders, Celeste and Lin, Justin W. and Jones, Eliot and Hussein, Gashon and Liu, Samantha and Jasper, Donovan and Peetathawatchai, Pura and Glenn, Ari and Sivashankar, Vikram and Zamoshchin, Daniel and Glikbarg, Leo and Askaryar, Derek and Yang, Mike and Zhang, Teddy and Alluri, Rishi and Tran, Nathan and Sangpisit, Rinnara and Yiorkadjis, Polycarpos and Osele, Kenny and Raghupathi, Gautham and Boneh, Dan and Ho, Daniel E. and Liang, Percy},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=tc90LV0yRL}
}
```
