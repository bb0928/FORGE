#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$PROJECT_ROOT/legged_gym_repo:${PYTHONPATH:-}"
export PYTHONPATH="$PROJECT_ROOT/rsl_rl_repo:${PYTHONPATH:-}"

cd "$PROJECT_ROOT/legged_gym_repo"

RESUMEPATH="$PROJECT_ROOT/legged_gym_repo/logs/final-grid-search/Apr02_13-53-08_0/model_0.pt"

python legged_gym/scripts/play_res.py \
  --task g1_ee_residual \
  --max_iterations 20000 \
  --resume --resume_path "$RESUMEPATH" --num_envs 1 \
  --experiment_name "final-grid-search" \
  --headless
