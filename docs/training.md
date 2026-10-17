# Training

[Overview](../README.md) · [Installation](installation.md)

## Default recipe

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py
```

The launcher starts one process per GPU with `torch.distributed.run`, applies
[configs/milestone.json](../configs/milestone.json), and trains from scratch.
Use this launcher to apply the full recipe.

| Setting | Value |
|---|---|
| Robot | G1, 29 controlled joints; fixed Dex3-1 fingers |
| GPUs / environments | 4 GPUs × 4,096 environments |
| Iterations / seed | 10,000 / 7 |
| Rollout | 24 steps per environment |
| Hand payloads | Independently randomized 0–2 kg per hand at startup |
| Domain randomization | Physical parameters, gains, actuator delay and observation noise |
| Upper-body curriculum | 0 → 0.6, increment 0.05 |
| Pose guard | Upper body; weight −0.25; starts at iteration 1000 |
| Pose-guard thresholds | Warning 0.65, danger 0.80, danger scale 3.0 |
| PPO | 5 epochs, 4 minibatches, adaptive learning rate starting at 1e-3 |
| Symmetry | Data augmentation and mirror loss, coefficient 1e-3 |
| StdL2 | Repository-local PPO; maximum coefficient 0.02 |

## Smoke test

Run a short training-to-playback cycle:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --gpus 1 --smoke --run-name smoke
CUDA_VISIBLE_DEVICES=0 python scripts/play.py \
  --checkpoint logs/smoke/model_1.pt --headless --steps 300
```

`--smoke` selects 64 environments per GPU and two PPO iterations. This is a quick
installation check. A completed headless replay prints `HEADLESS_REPLAY_OK`.

## Custom runs

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --gpus 1 --num-envs 512 --iterations 10000 --seed 7 --run-name g1-single-gpu
```

| Option | Default | Meaning |
|---|---|---|
| `--gpus` | 4 | Processes / visible GPUs on this machine |
| `--num-envs` | 4096 | Parallel environments per GPU |
| `--iterations` | 10000 | PPO iterations |
| `--seed` | 7 | Random seed |
| `--run-name` | `forge-20260801-s7` | Output subdirectory |
| `--log-root` | Repository `logs/` | Parent output directory |
| `--smoke` | Off | Use 64 environments and 2 iterations |
| `--dry-run` | Off | Print the command and configuration |

GPU and environment counts determine the total PPO batch size. Choose a fresh
`--run-name` for each experiment; existing nonempty run directories are protected
against overwriting.

## Outputs

```text
logs/forge-20260801-s7/
├── launch.json             Configuration and source metadata
├── model_0.pt              First saved checkpoint
├── model_100.pt            Periodic checkpoint
├── ...
├── model_9999.pt           Final checkpoint after 10,000 iterations
└── ...                     TensorBoard events and training outputs
```

```bash
tensorboard --logdir logs
```

Checkpoints are saved every 100 iterations and at the end of training. The launcher
records resource settings and the source manifest in `launch.json`, resets
inherited experiment settings, and selects local TensorBoard logging.
