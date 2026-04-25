import asyncio
import json
import os
from openai import AsyncOpenAI
from openreward import AsyncOpenReward


async def test_with_openai():
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI()

    MODEL_NAME = "gpt-5.2"
    ENV_NAME = "minesweeper"
    SPLIT = "test"
    BASE_URL = "http://localhost:8080"

    # Manually define tool schema
    tools = [{
        "type": "function",
        "name": "reveal_cell",
        "description": "Reveal a cell on the minesweeper grid at the given row and column (0-indexed). Numbers show adjacent mine count. Hitting a mine ends the game.",
        "parameters": {
            "type": "object",
            "properties": {
                "row": {
                    "type": "integer",
                    "description": "Row index (0-indexed)"
                },
                "column": {
                    "type": "integer",
                    "description": "Column index (0-indexed)"
                }
            },
            "required": ["row", "column"],
            "additionalProperties": False
        }
    }]

    environment = or_client.environments.get(name=ENV_NAME, base_url=BASE_URL)
    tasks = await environment.list_tasks(split=SPLIT)

    print(f"Found {len(tasks)} tasks")
    print(f"Testing first task: {tasks[0]['id']}\n")

    for task in tasks[:1]:  # Test first task
        async with environment.session(task=task) as session:
            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]
            finished = False
            turn = 0
            max_turns = 50

            print(f"Initial prompt:\n{prompt[0].text}\n")
            print("="*60)

            while not finished and turn < max_turns:
                turn += 1
                print(f"\nTurn {turn}:")

                # Use responses.create(), NOT chat.completions.create()
                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    tools=tools,
                    input=input_list
                )

                # Response has 'output', NOT 'choices'
                input_list += response.output

                for item in response.output:
                    if item.type == "function_call":
                        print(f"  Tool call: {item.name}")
                        print(f"  Arguments: {item.arguments}")

                        tool_result = await session.call_tool(
                            item.name,
                            json.loads(str(item.arguments))
                        )

                        finished = tool_result.finished

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text
                        })

                        print(f"  Reward: {tool_result.reward:.3f}")
                        print(f"  Output: {tool_result.blocks[0].text[:200]}")

                        if tool_result.finished:
                            print("\n" + "="*60)
                            print('GAME FINISHED!')
                            print(f"Final reward: {tool_result.reward:.3f}")
                            print("="*60)
                            break

            if turn >= max_turns:
                print(f"\nReached maximum turns ({max_turns})")


if __name__ == "__main__":
    asyncio.run(test_with_openai())
