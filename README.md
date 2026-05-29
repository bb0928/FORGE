# FORGE: Force-Aware Multi-Granularity Control for Efficient Humanoid Loco-Manipulation

This repository contains the official open-source implementation of the paper **FORGE: Force-Aware Multi-Granularity Control for Efficient Humanoid Loco-Manipulation**.

---

## Installation

### 1. Create conda environment

```bash
conda create -n forge python=3.8 -y
conda activate forge
```

### 2. Install PyTorch

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install IsaacGym

IsaacGym must be downloaded manually from the [NVIDIA website](https://developer.nvidia.com/isaac-gym). After downloading and extracting:

```bash
cd isaacgym/python
pip install -e .
```

### 4. Install rsl_rl and legged_gym

```bash
pip install -e rsl_rl_repo
pip install -e legged_gym_repo
```

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Place the pretrained base model at `model_17000.pt` in the project root, then run:

```bash
bash scripts/train_stage2_residual.sh
```

Key arguments (edit the script to change):

| Argument | Default | Description |
|---|---|---|
| `--num_envs` | 4096 | Number of parallel environments |
| `--max_iterations` | 20000 | Training iterations |
| `--rl_device` | cuda:0 | RL compute device |
| `--sim_device` | cuda:0 | Simulation device |
| `--experiment_name` | final-grid-search | Experiment name for logging |

---

## Evaluation

Update `RESUMEPATH` in `scripts/test_res_pf.sh` to point to your checkpoint, then run:

```bash
bash scripts/test_res_pf.sh
```

This runs a single environment in headless mode using the `g1_ee_residual` task.

---

## TODO

- [x] RL residual policy training script (`scripts/train_stage2_residual.sh`)
- [x] RL residual policy evaluation script (`scripts/test_res_pf.sh`)
- [ ] Base policy training script (not yet open-sourced)
- [ ] MuJoCo deployment script (not yet open-sourced)
- [ ] Real-robot deployment script (not yet open-sourced)
