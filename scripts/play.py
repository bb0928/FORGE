"""Replay a locally supplied, trusted milestone checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from common import ROOT, environment, positive_int, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=positive_int, default=300)
    parser.add_argument("--randomization", action="store_true", help="Enable training domain randomization and observation noise")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if not args.device.startswith("cuda:") or not args.device[5:].isdigit():
        parser.error("--device must be cuda:N")
    env = environment({"NUM_ENVS": 1, "ENABLE_DOMAIN_RAND": int(args.randomization),
                       "FORGE_REQUIRED_GPUS": int(args.device[5:]) + 1})
    command = [sys.executable, str(ROOT / "play_base.py"), "--checkpoint", str(checkpoint),
               "--port", str(args.port), "--device", args.device, "--num-envs", "1"]
    if args.headless:
        command += ["--headless", "--steps", str(args.steps)]
    return run(command, env, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
