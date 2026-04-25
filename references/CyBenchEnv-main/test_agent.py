import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
ENV_NAME = "GeneralReasoning/cybench"
SPLIT = "easy"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENREWARD_API_KEY = os.environ["OPENREWARD_API_KEY"]

async def main() -> None:
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Connect to local server
    environment = or_client.environments.get(
        name=ENV_NAME,
        base_url="http://localhost:8080"
    )

    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks in '{SPLIT}' split")

    # Test first task
    task = tasks[0]

    rollout = or_client.rollout.create(
        run_name=ENV_NAME.split("/")[-1] + "_test",
        rollout_name="test_run",
        environment=ENV_NAME,
        split=SPLIT,
        task_spec=task.task_spec,
    )

    async with environment.session(
        task=task,
        secrets={"api_key": OPENREWARD_API_KEY}
    ) as session:
        prompt_blocks = await session.get_prompt()
        input_list = [{"role": "user", "content": prompt_blocks[0].text}]
        finished = False

        rollout.log_openai_response(message=input_list[0], is_finished=False)

        while not finished:
            response = await oai_client.responses.create(
                model=MODEL_NAME,
                tools=tools,
                input=input_list,
            )

            rollout.log_openai_response(response.output[-1])
            input_list.extend(response.output)

            for item in response.output:
                if item.type == "function_call":
                    print(f"\nTool: {item.name}")

                    tool_result = await session.call_tool(
                        item.name,
                        json.loads(str(item.arguments)),
                    )

                    reward = tool_result.reward
                    finished = tool_result.finished

                    print(f"Reward: {reward}")
                    print(f"Finished: {finished}")

                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": tool_result.blocks[0].text if tool_result.blocks else "",
                    })

                    rollout.log_openai_response(input_list[-1], reward=reward, is_finished=finished)

                    if finished:
                        print("\n=== Task completed ===")
                        break

            if not any(i.type == "function_call" for i in response.output):
                print("No tool calls, stopping")
                break

if __name__ == "__main__":
    asyncio.run(main())
