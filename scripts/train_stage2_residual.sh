#!/bin/bash
set -euo pipefail

export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="$PROJECT_ROOT/legged_gym_repo:${PYTHONPATH:-}"
export PYTHONPATH="$PROJECT_ROOT/rsl_rl_repo:${PYTHONPATH:-}"

cd "$PROJECT_ROOT/legged_gym_repo"

base_model_path="$PROJECT_ROOT/model_17000.pt"

echo "base model: $base_model_path"

# Verify the active Python environment has the required packages.
# If this fails, activate your conda environment first:
#   conda activate <your_env>
python -c "import numpy, torch" 2>/dev/null || {
    echo "ERROR: numpy or torch not found. Please activate your conda environment first:"
    echo "  conda activate <your_env_name>"
    exit 1
}

python legged_gym/scripts/train.py \
    --task g1_ee_residual \
    --num_envs 4096 \
    --headless \
    --rl_device cuda:0 \
    --sim_device cuda:0 \
    --experiment_name "final-grid-search" \
    --run_name "0" \
    --max_iterations 20000 \
    --base_model_path "$base_model_path" \
    --command_sample_strategy "uniform"

echo "completed!"
