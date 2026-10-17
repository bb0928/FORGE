"""Dependency-free launch configuration; no shell evaluation of user arguments."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / "configs/milestone.json").read_text())


def positive_int(value):
    number = int(value)
    if number < 1:
        raise ValueError("must be a positive integer")
    return number


def environment(overrides=None):
    env = os.environ.copy()
    # A launcher is a reproducible profile, not a mix of old experiment settings.
    controlled = set(PROFILE["environment"]) | {
        "RESUME_CKPT", "WRIST_TERM_POS_CM", "WRIST_TERM_ORI_DEG",
        "WRIST_TERM_GRACE_STEPS", "WRIST_TERM_WARMUP_STEPS",
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
        "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
    }
    for key in list(env):
        if key in controlled or key.startswith(("FORGE_", "TORCHELASTIC_", "WANDB_")):
            env.pop(key)
    env.update(PROFILE["environment"])
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "MUJOCO_GL": "egl", "WANDB_MODE": "disabled",
                "FORGE_LOGGER": "tensorboard"})
    env.update({str(k): str(v) for k, v in (overrides or {}).items()})
    return env


def run(command, env, dry_run=False, before_run=None):
    visible = {k: env[k] for k in sorted(set(PROFILE["environment"]) | {
        "FORGE_LOG_ROOT", "RUN_NAME", "FORGE_LOGGER", "WANDB_MODE"
    }) if k in env}
    print(json.dumps({"command": command, "environment": visible}, indent=2), flush=True)
    if dry_run:
        return 0
    # The preflight checks CUDA and imports before any training or checkpoint load.
    subprocess.run([sys.executable, str(ROOT / "scripts/check_runtime.py")],
                   cwd=ROOT, env=env, check=True)
    if before_run is not None:
        before_run()
    return subprocess.call(command, cwd=ROOT, env=env)
