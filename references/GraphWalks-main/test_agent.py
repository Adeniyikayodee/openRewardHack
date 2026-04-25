import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward


MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable required")


async def test_agent():
    """Test GraphWalks environment with OpenAI agent."""

    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Connect to environment (local or deployed)
    environment = or_client.environments.get(
        name="EnvCommons/GraphWalks",
        base_url="http://localhost:8080"  # Comment out for deployed environment
    )

    # Get tasks and tools
    print("Fetching tasks and tools...")
    tasks = await environment.list_tasks(split="test")
    tools = await environment.list_tools(format="openai")

    print(f"Number of tasks: {len(tasks)}")
    print(f"Number of tools: {len(tools)}")

    # Test first task
    task = tasks[0]
    print(f"\nTesting task...")
    print(f"Starting session...\n")

    async with environment.session(task=task) as session:
        prompt = await session.get_prompt()

        # Build input list (Responses API format)
        prompt_text = prompt[0].text if isinstance(prompt, list) else str(prompt)
        print(prompt_text)
        input_list = [{"role": "user", "content": prompt_text}]
        finished = False

        print("Agent starting...\n")

        turn = 0
        while not finished and turn < 10:  # Limit to 10 turns
            turn += 1
            print(f"--- Turn {turn} ---")

            # Call model
            response = await oai_client.responses.create(
                model=MODEL_NAME,
                tools=tools,
                input=input_list,
            )

            # Add response to input history
            input_list += response.output

            # Process tool calls
            has_tool_call = False
            for item in response.output:
                if item.type == "function_call":
                    has_tool_call = True
                    print(f"Tool call: {item.name}")
                    print(f"Arguments: {item.arguments}\n")

                    # Execute tool
                    tool_result = await session.call_tool(
                        item.name,
                        json.loads(str(item.arguments))
                    )

                    result_text = tool_result.blocks[0].text if tool_result.blocks else "No output"
                    print(f"Tool result: {result_text}")
                    print(f"Reward: {tool_result.reward}")
                    print(f"Finished: {tool_result.finished}\n")

                    finished = tool_result.finished

                    # Add tool result to input (Responses API format)
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps({"result": result_text})
                    })

            # Stop if no tool call (model gave up)
            if not has_tool_call:
                print("Model stopped without tool call")
                break

        print(f"\nSession complete after {turn} turn(s)")


if __name__ == "__main__":
    asyncio.run(test_agent())
