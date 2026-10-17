# FORGE mjlab: August 1, 2026 milestone

A standalone 29-DOF G1 base policy for walking, squatting and upper-body target
tracking, using mjlab / MuJoCo Warp and RSL-RL. This source release preserves the
August 1 milestone's training implementation and makes its launch configuration
portable.

**Scope:** the mjlab base-policy milestone. This is not the complete Isaac Gym
FORGE paper implementation, the later force-estimator experiments, or a real-robot
deployment package. No pretrained checkpoint is included. Training can start from
scratch; playback requires your own compatible checkpoint.

## Installation

Use Linux, Python 3.12 and an NVIDIA CUDA GPU. The recorded run used **4 GPUs**,
4,096 environments per GPU and 10,000 training iterations. Other GPU counts or
environment counts change the total PPO batch and are not equivalent reproductions.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The dependency pins are a **reconstructed compatibility baseline**, not an archived
pip freeze from the original training machine. The vendored PPO matches RSL-RL
5.2.0 plus the FORGE StdL2 changes, and the original G1 XML hash matches mjlab
1.4.0. Full GPU reproduction remains to be validated; see `VALIDATION.md`.
Installation uses the public package index and does not need an internal image,
scheduler, mount, W&B server or account.

## Train

Inspect the exact plan without installing simulator dependencies or starting work:

```bash
python scripts/train.py --dry-run
```

Run the milestone from scratch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py
```

Outputs go to `logs/forge-20260801-s7/`, including `launch.json`, TensorBoard logs
and checkpoints. A nonempty run directory is rejected to prevent accidental
overwrites. Choose a new name with `--run-name`. The launcher clears stale
experiment and distributed-process settings, including `RESUME_CKPT`, before
applying `configs/milestone.json`.

For a short functional smoke test on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --gpus 1 --smoke --run-name smoke
```

This uses 64 environments and two iterations. It checks execution, not policy
quality or the milestone result. To change resources explicitly:

```bash
python scripts/train.py --gpus 1 --num-envs 512 --iterations 10000 --run-name single-gpu
```

`--log-root`, `--seed`, `--num-envs`, `--iterations` and `--gpus` are recorded in
`launch.json`. Outputs and checkpoints are excluded from Git. TensorBoard is the
default logger; the supplied launchers do not upload data to W&B.

## Replay

Supply a compatible checkpoint produced by this base-policy implementation:

```bash
python scripts/play.py --checkpoint logs/forge-20260801-s7/model_9999.pt
```

Open `http://localhost:8080`. The viewer binds to localhost. For a remote GPU
machine, forward the port with `ssh -L 8080:localhost:8080 user@gpu-host`.
The viewer provides walking, height and upper-body curriculum controls, plus FK
target overlays. The initial curriculum is taken from the checkpoint and bounded
by the milestone cap of 0.6. Interactive replay disables automatic resets so the
viewer can be inspected manually; it is not a quantitative benchmark.

For a finite, headless replay check:

```bash
python scripts/play.py --checkpoint logs/smoke/model_1.pt --headless --steps 300
```

Headless replay retains environment termination/reset rules, rejects non-finite
actions/rewards and prints `HEADLESS_REPLAY_OK` on completion. Default replay
disables training randomization and observation noise; pass `--randomization` to
enable them. `--device cuda:N` selects a visible GPU. Checkpoints are loaded with
PyTorch serialization and must come from a trusted source.

## Milestone configuration

| Setting | Value |
|---|---|
| Robot | G1, 29 controlled joints; fixed Dex3-1 fingers |
| Hand asset | `dex31_payload` |
| Additional payload mass | Independently randomized 0–2 kg per hand at startup |
| Domain randomization | Enabled; conservative physical, gain, delay and observation randomization |
| Command mixture | Walking / position / force-mode masks: 0.8 / 0.1 / 0.1 |
| Upper-body curriculum | 0 → 0.6, increment 0.05 |
| Pose guard | Upper body only; weight -0.25; starts at iteration 1000 |
| Pose-guard thresholds | Warning 0.65, danger 0.80, danger scale 3.0 |
| Seed | 7 |
| Rollout | 24 steps per environment |
| PPO | 5 epochs, 4 minibatches, adaptive learning rate starting at 1e-3 |
| Symmetry | Data augmentation and mirror loss, coefficient 1e-3 |
| StdL2 | Repository-local PPO implementation, maximum coefficient 0.02 |

The force-mode mask in this base-policy command sampler does **not** establish
validated force estimation or closed-loop force control. Those experiments are
outside this release.

The actor consumes six frames of 131 values plus six 11-group masks: **852 values**.
The critic consumes one 156-value privileged frame plus a 17-group mask:
**173 values**. The actor outputs the full 29-joint action, and mjlab applies its
per-joint action scales and default offsets exactly once. There is no frozen-base
plus residual composition in this training entry point.

## Tests and provenance

```bash
python -m unittest discover -s tests -v
python scripts/check_runtime.py
sha256sum -c SHA256SUMS
```

Launcher/resource tests use the Python standard library. Neural checks require
PyTorch; XML compilation requires MuJoCo. Without these packages those tests are
explicitly skipped. For CPU-only checks on macOS or Linux, install
`torch==2.7.1` and `mujoco==3.8.0` in a separate environment; GPU execution is still
required to validate the released training/replay path.

`SOURCE_MANIFEST.json` records original and release SHA256 hashes. Rewards, the
runner, robot XML and network code were retained; the training entry point has
portable path/logging changes. See `RELEASE_NOTES.md` for the exact scope and
`VALIDATION.md` for completed checks and remaining limits.

## Layout

```text
configs/milestone.json       Recorded launch settings
scripts/train.py             Portable single-machine torchrun launcher
scripts/play.py              Interactive/headless replay launcher
scripts/check_runtime.py     Dependency, CUDA and asset preflight
train_forge_base.py          Environment, rewards and training configuration
play_base.py                 Viewer and finite replay implementation
commands.py                 Mixed commands and FK target curriculum
obs_spec.py / obs_builder.py Observation layouts and history
rl/                         Networks, runner and vendored StdL2 PPO
assets/                     Milestone robot XML and its 49 referenced meshes
tests/                      Launcher, resource and neural contract checks
```

## License and attribution

FORGE contributions in this release are provided under Apache-2.0. Third-party
components retain their own notices and terms: RSL-RL's vendored PPO is
BSD-3-Clause, and Unitree robot resources retain Unitree's BSD-3-Clause notice.
See `NOTICE`, `THIRD_PARTY_NOTICES.md` and `LICENSES/` for source versions and
license text.
