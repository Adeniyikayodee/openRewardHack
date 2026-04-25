#!/usr/bin/env python3
"""End-to-end rollout against the LondonDynamicRouting environment.

Required env vars:
  OPENREWARD_API_KEY   — for hitting the OpenReward env
  OPENAI_API_KEY       — for executing the agent

Usage:
  # Local server (Daniel's dev loop)
  python scripts/run_rollout.py --local --task-idx 0
  # Deployed env (demo)
  python scripts/run_rollout.py --task-idx 0
  # Multiple tasks, save trace
  python scripts/run_rollout.py --n-tasks 3 --record traces/
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

DEFAULT_MODEL = os.environ.get("ROLLOUT_MODEL", "gpt-5.4")
ENV_NAMESPACE = os.environ.get("OR_ENV", "EnvCommons/LondonDynamicRouting")
LOCAL_BASE_URL = "http://localhost:8080"
LOCAL_NAME = "LondonDynamicRouting"
MAX_TURNS = 80

SYSTEM_PROMPT = """You are an expert transit fleet dispatcher operating in \
London. You manage a heterogeneous fleet across one operational day. Your goal \
is to serve as many passenger and parcel requests as possible, on time, with \
minimum total kilometres driven, while reacting to traffic, weather, and \
unexpected breakdowns or capacity changes.

Strategy:
  1. Begin by calling `list_pending_requests` and `get_state` to understand \
the situation.
  2. Assign requests to vehicles using `assign(request_id, vehicle_id, \
pickup_position, dropoff_position)`. Pickup position must be strictly less \
than dropoff position. Position 0 means insert at the start of the route.
  3. Prefer cheap insertions (low marginal km) and respect time windows + \
capacity. The reward signal will tell you how good each insertion was.
  4. After assigning a batch of pending requests, call `tick(15)` or \
`tick(30)` to advance time. Many requests are revealed mid-day, so DO NOT \
try to plan everything at t=0.
  5. If a vehicle breaks down (you'll see a ⚠ breakdown event in tick \
output), use `swap_vehicles(broken_id, healthy_id)` to recover. If a \
capacity drops, use `reassign` to move bumped requests.
  6. Be concise. Each action carries a small efficiency penalty (~0.005). \
Avoid spamming queries. Prefer doing useful work to repeated state inspections.
  7. Submit your plan with `submit_plan()` when you've handled everything \
or you're near the horizon (t close to 960)."""


async def run_one_task(task, environment, oai_client, model, max_turns,
                       trace_path=None):
    tools = await environment.list_tools(format="openai")
    OR_KEY = os.environ["OPENREWARD_API_KEY"]
    trace = {"task_id": task.task_spec["id"],
             "difficulty": task.task_spec["difficulty"],
             "split": task.task_spec.get("split", "?"),
             "actions": [], "rewards": [], "model": model}

    async with environment.session(task=task,
                                   secrets={"api_key": OR_KEY}) as session:
        prompt = await session.get_prompt()
        input_list = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt[0].text},
        ]
        total_reward = 0.0
        finished = False

        for turn in range(max_turns):
            try:
                response = await oai_client.responses.create(
                    model=model, tools=tools, input=input_list)
            except Exception as e:
                print(f"  [turn {turn}] OpenAI API error: {e}")
                trace["actions"].append({"name": "_api_error",
                                         "error": str(e)})
                break

            input_list += response.output

            had_tool_call = False
            for item in response.output:
                if item.type != "function_call":
                    continue
                had_tool_call = True
                t0 = time.time()
                try:
                    args = json.loads(str(item.arguments))
                except Exception:
                    args = {}
                try:
                    tr = await session.call_tool(item.name, args)
                    text = "".join(b.text for b in (tr.blocks or [])
                                   if hasattr(b, "text"))
                    reward = tr.reward or 0.0
                    finished = bool(tr.finished)
                    total_reward += reward
                    trace["actions"].append({
                        "turn": turn, "name": item.name, "args": args,
                        "duration_s": round(time.time() - t0, 3),
                        "text_preview": text[:200]})
                    trace["rewards"].append(reward)
                    print(f"  [t={turn:02d}] {item.name}({_short(args)}) "
                          f"→ r={reward:+.3f}, finished={finished}, "
                          f"cum={total_reward:+.3f}")
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id, "output": text})
                    if finished:
                        break
                except Exception as e:
                    err = f"Tool failed: {type(e).__name__}: {e}"
                    print(f"  [t={turn:02d}] {item.name} ERROR: {e}")
                    trace["actions"].append({"turn": turn, "name": item.name,
                                             "args": args, "error": str(e)})
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": f"Tool error: {err}"})

            if finished:
                break
            if not had_tool_call:
                print(f"  [t={turn:02d}] Model returned no tool call; "
                      f"forcing submit_plan.")
                tr = await session.call_tool("submit_plan", {})
                total_reward += tr.reward or 0.0
                trace["actions"].append({"turn": turn, "name": "submit_plan",
                                         "args": {}, "forced": True})
                trace["rewards"].append(tr.reward or 0.0)
                finished = True
                break

        if not finished:
            print(f"  Hit max_turns={max_turns} without termination.")

    trace["total_reward"] = total_reward
    trace["finished"] = finished
    trace["n_actions"] = len(trace["actions"])
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        Path(trace_path).write_text(json.dumps(trace, indent=2, default=str))
        print(f"  Trace saved → {trace_path}")
    return total_reward, finished, trace


def _short(args):
    if not args:
        return ""
    return ", ".join(f"{k}={v}" for k, v in list(args.items())[:3])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="tutorial",
                    choices=["tutorial", "train", "test"])
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--local", action="store_true",
                    help="hit localhost:8080 instead of deployed env")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--record", default=None,
                    help="directory to write trace JSONs")
    args = ap.parse_args()

    if not os.environ.get("OPENREWARD_API_KEY"):
        print("ERROR: OPENREWARD_API_KEY not set", file=sys.stderr)
        return 2
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        return 2

    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI()

    if args.local:
        environment = or_client.environments.get(name=LOCAL_NAME,
                                                 base_url=LOCAL_BASE_URL)
    else:
        environment = or_client.environments.get(name=ENV_NAMESPACE)

    tasks = await environment.list_tasks(split=args.split)
    print(f"Got {len(tasks)} tasks in split '{args.split}'.")
    selected = tasks[args.task_idx:args.task_idx + args.n_tasks]
    if not selected:
        print(f"No tasks at index {args.task_idx}")
        return 1

    results = []
    for i, task in enumerate(selected):
        print(f"\n=== Task {args.task_idx + i}: {task.task_spec['id']} "
              f"(difficulty={task.task_spec['difficulty']}) ===")
        trace_path = None
        if args.record:
            trace_path = (f"{args.record.rstrip('/')}/"
                          f"{task.task_spec['id']}.json")
        r, f, _ = await run_one_task(task, environment, oai_client,
                                     args.model, args.max_turns, trace_path)
        results.append((task.task_spec["id"],
                        task.task_spec["difficulty"], r, f))
        print(f"=== Total reward: {r:+.3f} | finished: {f} ===")

    print("\n--- SUMMARY ---")
    for tid, d, r, f in results:
        print(f"  {tid} (d={d}): reward={r:+.3f} finished={f}")
    avg = sum(r for _, _, r, _ in results) / len(results)
    print(f"\nMean reward across {len(results)} tasks: {avg:+.3f}")
    return 0 if all(f for _, _, _, f in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
