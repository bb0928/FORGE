"""Launch the August 1 profile, from scratch, on one local CUDA machine."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

from common import ROOT, PROFILE, environment, positive_int, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=positive_int, default=PROFILE["gpus"])
    parser.add_argument("--num-envs", type=positive_int, default=4096, help="Environments per GPU")
    parser.add_argument("--iterations", type=positive_int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-name", default="forge-20260801-s7")
    parser.add_argument("--log-root", type=Path, default=ROOT / "logs")
    parser.add_argument("--smoke", action="store_true", help="64 envs/GPU, 2 iterations; a functional test only")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without imports, writes or GPU work")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        parser.error("--run-name must be a single directory name using letters, digits, _, - or .")
    if args.smoke:
        args.num_envs, args.iterations = 64, 2
    log_root = args.log_root.expanduser().resolve()
    log_dir = log_root / args.run_name
    if not args.dry_run and log_dir.exists() and any(log_dir.iterdir()):
        parser.error(f"Run directory is not empty: {log_dir}. Choose a new --run-name.")
    env = environment({"NUM_ENVS": args.num_envs, "MAX_ITER": args.iterations,
                       "SEED": args.seed, "RUN_NAME": args.run_name,
                       "FORGE_LOG_ROOT": log_root, "FORGE_REQUIRED_GPUS": args.gpus})
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nnodes=1", f"--nproc-per-node={args.gpus}", str(ROOT / "train_forge_base.py")]
    def record_launch():
        log_dir.mkdir(parents=True, exist_ok=True)
        metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                    "profile": PROFILE, "gpus": args.gpus, "num_envs_per_gpu": args.num_envs,
                    "iterations": args.iterations, "seed": args.seed, "smoke": args.smoke,
                    "milestone_topology": args.gpus == 4 and args.num_envs == 4096 and
                    args.iterations == 10000 and args.seed == 7,
                    "source_manifest": json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())}
        (log_dir / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return run(command, env, args.dry_run, before_run=record_launch)


if __name__ == "__main__":
    raise SystemExit(main())
