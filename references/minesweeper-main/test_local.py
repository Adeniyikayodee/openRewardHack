import asyncio
from openreward import AsyncOpenReward


async def test_locally():
    print("Starting local test for Minesweeper environment...")
    client = AsyncOpenReward()

    # Connect to local server
    env = client.environments.get(
        name="minesweeper",
        base_url="http://localhost:8080"
    )

    # Get tasks
    print("\n1. Testing task retrieval...")
    tasks = await env.list_tasks(split="test")
    print(f"   Found {len(tasks)} tasks")

    if len(tasks) == 0:
        print("   FAILED: No tasks found")
        return

    example_task = tasks[0]
    task_dict = example_task if isinstance(example_task, dict) else example_task.__dict__
    print(f"   Example task: {task_dict.get('id', 'N/A')}")

    # Test with session
    print("\n2. Testing session...")
    async with env.session(task=example_task) as session:
        # Test prompt generation
        print("   Testing prompt generation...")
        prompt = await session.get_prompt()
        print(f"   Prompt preview: {prompt[0].text[:200]}...")

        # Test tool call - reveal cell at (0, 0)
        print("\n3. Testing reveal_cell tool...")
        result = await session.call_tool("reveal_cell", {"row": 0, "column": 0})
        print(f"   Reward: {result.reward}")
        print(f"   Finished: {result.finished}")
        print(f"   Output preview: {result.blocks[0].text[:200] if result.blocks else 'No output'}")

        # Try a few more moves if not finished
        if not result.finished:
            print("\n4. Testing additional moves...")
            for i in range(3):
                result = await session.call_tool("reveal_cell", {"row": i, "column": i})
                print(f"   Move {i+2}: reward={result.reward}, finished={result.finished}")
                if result.finished:
                    print(f"   Final output: {result.blocks[0].text}")
                    break

    print("\n" + "="*60)
    print("SMOKE TEST PASSED!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_locally())
