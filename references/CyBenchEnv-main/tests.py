import pytest

from openreward.environments import ToolOutput, JSONObject
from cybench import CyBench, AnswerParams
from utils import CHALLENGES_DIR

TASKS = CyBench.list_tasks("easy") + CyBench.list_tasks("easy_korean") + CyBench.list_tasks("hard") + CyBench.list_tasks("hard_korean")
EXAMPLE_TASK = TASKS[0]
TASKS_WITH_FILES = [t for t in TASKS if t["files"]]

@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS_WITH_FILES, ids=lambda t: t["name"])
async def test_files_uploaded(task: JSONObject):
    """Test that challenge files are uploaded to the correct locations."""
    env = CyBench(task_spec=task, secrets={"api_key": "test"})

    try:
        await env.setup()

        for local_path, upload_path in env.validated.files.items():
            test_flag = "-d" if upload_path.endswith("/") else "-f"
            _, code = await env.sandbox.run(f"test {test_flag} {upload_path}")
            assert code == 0, f"File check failed for {upload_path}"
    finally:
        await env.teardown()

@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["name"])
async def test_gold(task: JSONObject):
    env = CyBench(task_spec=task, secrets={"api_key": "test"})
    try:
        await env.setup()

        answer = env.validated.flag
        result: ToolOutput = await env.answer(AnswerParams(flag=answer))
        assert result.reward == 1.0
    finally:
        await env.teardown()

@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["name"])
async def test_gold_using_solution(task: JSONObject):
    env = CyBench(task_spec=task, secrets={"api_key": "test"})
    try:
        await env.setup()

        for local_path, upload_path in env.validated.solution_files.items():
            await env.sandbox.upload(local_path, upload_path)
        res, code = await env.sandbox.run(env.validated.solution_bash_cmd)
        assert code == 0, f"Solution command failed with exit code {code}"

        result: ToolOutput = await env.answer(AnswerParams(flag=res))
        assert result.reward == 1.0
    finally:
        await env.teardown()

@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["name"])
async def test_xfail(task: JSONObject):
    env = CyBench(task_spec=task, secrets={"api_key": "test"})
    try:
        await env.setup()

        incorrect_answer = "123456789"
        result: ToolOutput = await env.answer(AnswerParams(flag=incorrect_answer))
        assert result.reward == 0.0
    finally:
        await env.teardown()



async def main():
    task = [t for t in TASKS if t["name"] == "frog_waf"][0]
    print(task)
    env = CyBench(task_spec=task, secrets={"api_key": "test"})
    try:
        await env.setup()
        print("setup")

        for local_path, upload_path in env.validated.solution_files.items():
            await env.sandbox.upload(local_path, upload_path)
        print(f"Running {env.validated.solution_bash_cmd}")
        res, code = await env.sandbox.run(env.validated.solution_bash_cmd)
        assert code == 0, f"Command failed with exit code {code}"
        print(res)

        result: ToolOutput = await env.answer(AnswerParams(flag=res))
        print(result)
    finally:
        await env.teardown()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
