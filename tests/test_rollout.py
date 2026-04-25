"""Smoke test the rollout script end-to-end against the local server."""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


@pytest.mark.skipif(
    not (os.environ.get("OPENREWARD_API_KEY") and
         os.environ.get("OPENAI_API_KEY")),
    reason="API keys not set")
def test_rollout_local_one_tutorial_task(server, tasks_parquet_dir):
    env = {**os.environ, "ORWD_DATA_DIR": str(tasks_parquet_dir)}
    proc = subprocess.run(
        ["python", "scripts/run_rollout.py", "--local",
         "--split", "tutorial", "--task-idx", "0", "--max-turns", "30"],
        capture_output=True, text=True, timeout=300, env=env,
        cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr
    assert "Total reward" in proc.stdout or "Mean reward" in proc.stdout
