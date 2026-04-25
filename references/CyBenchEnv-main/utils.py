import re

import yaml
from pathlib import Path

from openreward.api.sandboxes.types import SandboxSidecarContainer, SandboxHostAlias

CHALLENGES_DIR = Path(__file__).parent / "challenges"


def _parse_healthcheck(config: dict, sidecar_name: str) -> str | None:
    """
    Parse healthcheck from compose config and adapt for running from main pod.
    Replaces localhost/127.0.0.1 with the sidecar's hostname.
    """
    if "healthcheck" not in config:
        return None

    healthcheck = config["healthcheck"]
    test = healthcheck.get("test")
    if not test:
        return None

    # Format is ["CMD-SHELL", "command..."]
    assert isinstance(test, list) and len(test) == 2 and test[0] == "CMD-SHELL", \
        f"Unexpected healthcheck format for {sidecar_name}: {test}"

    cmd = test[1]
    # Replace localhost/127.0.0.1 with sidecar hostname
    cmd = re.sub(r'\blocalhost\b', sidecar_name, cmd)
    cmd = re.sub(r'\b127\.0\.0\.1\b', sidecar_name, cmd)

    # Convert pgrep checks to network-based checks.
    # pgrep looks for a local process, but healthchecks run from the agent
    # sandbox where the sidecar's process doesn't exist. Use a TCP probe instead.
    if 'pgrep' in cmd:
        ports = [int(p) for p in config.get("expose", [])]
        if ports:
            cmd = f"socat TCP:{sidecar_name}:{ports[0]} - </dev/null"
        else:
            return None

    return cmd


def parse_compose_for_challenge(
    challenge_name: str,
) -> tuple[list[SandboxSidecarContainer], list[SandboxHostAlias], list[str]]:
    """
    Parse compose.yaml for a challenge and extract sidecar configurations.

    Returns (sidecars, host_aliases, healthcheck_cmds).
    """
    compose_path = CHALLENGES_DIR / challenge_name / "compose.yaml"
    assert compose_path.exists(), f"Compose file not found for challenge {challenge_name}"
    with open(compose_path, "r") as f:
        compose = yaml.safe_load(f)

    services = compose.get("services", {})
    assert services, f"No services found in compose file for challenge {challenge_name}"

    # The main service uses the agent-sandbox image
    main_service_names: set[str] = set()
    for name, config in services.items():
        image = config.get("image", "")
        if image == "generalreasoning/evals-cybench-agent-sandbox":
            main_service_names.add(name)
    assert len(main_service_names) == 1, f"Expected 1 main service in compose file for challenge {challenge_name}, found {len(main_service_names)}"

    sidecars: list[SandboxSidecarContainer] = []
    sidecar_hostnames: list[str] = []
    healthcheck_cmds: list[str] = []

    for name, config in services.items():
        if name in main_service_names:
            continue
        assert "image" in config, f"Sidecar {name} in {challenge_name} has no image"
        assert "build" not in config, f"Sidecar {name} in {challenge_name} has build (not supported)"
        assert "ports" not in config, f"Sidecar {name} in {challenge_name} has ports (use expose instead)"
        assert "command" not in config, f"Sidecar {name} in {challenge_name} has command (not supported)"

        image = config["image"]
        if ":" not in image:
            image = f"{image}:latest"

        # Extract ports from expose
        ports: list[int] = []
        if "expose" in config:
            for port_spec in config["expose"]:
                ports.append(int(port_spec))

        # Extract environment variables
        env: dict[str, str] | None = None
        if "environment" in config:
            env_config = config["environment"]
            if isinstance(env_config, dict):
                env = {k: str(v) for k, v in env_config.items()}
            elif isinstance(env_config, list):
                env = {}
                for item in env_config:
                    if "=" in item:
                        k, v = item.split("=", 1)
                        env[k] = v

        # Parse healthcheck (adapted for main pod)
        healthcheck_cmd = _parse_healthcheck(config, name)
        if healthcheck_cmd:
            healthcheck_cmds.append(healthcheck_cmd)

        sidecars.append(
            SandboxSidecarContainer(
                name=name,
                image=image,
                ports=ports if ports else None,
                env=env,
                machine_size="1:2",
            )
        )
        sidecar_hostnames.append(name)

    # Create host aliases for all sidecars
    host_aliases: list[SandboxHostAlias] = []
    if sidecar_hostnames:
        host_aliases.append(SandboxHostAlias(ip="127.0.0.1", hostnames=sidecar_hostnames))

    return sidecars, host_aliases, healthcheck_cmds
