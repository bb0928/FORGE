# Playback

[Overview](../README.md) · [Training](training.md)

## Interactive viewer

Pass a checkpoint trained with this base-policy implementation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/play.py \
  --checkpoint logs/forge-20260801-s7/model_9999.pt
```

Open **http://localhost:8080** after initialization. The viewer simulates one
robot and restores the upper-body curriculum from the checkpoint, up to 0.6.

## Controls

The viewer starts in manipulation mode with a height target of 0.74 m.

| Control | Use |
|---|---|
| `force manip (squat enabled)` | Checked: height control with zero walking velocity. Unchecked: velocity control. |
| `base height (m)` | Adjust standing/squat height from 0.34–0.78 m in manipulation mode. |
| `lin_vel_x` | Forward/backward velocity in walking mode. |
| `lin_vel_y` | Lateral velocity in walking mode. |
| `ang_vel_z` | Turning velocity in walking mode. |
| `upper curriculum` | Upper-body target range, up to 0.6. |
| `show FK target vs actual` | Target and actual wrist/elbow positions and error segments. |

Start with small command changes. Interactive playback uses manual resets so the
scene remains available for inspection.

## Remote viewing

On your local computer, forward the viewer port from the GPU machine:

```bash
ssh -L 8080:localhost:8080 user@gpu-host
```

Then open http://localhost:8080 locally. To use another port, launch playback with
`--port 8081` and forward port 8081 instead.

## Headless replay

```bash
python scripts/play.py \
  --checkpoint logs/forge-20260801-s7/model_9999.pt \
  --headless --steps 300
```

This runs for a fixed number of steps with environment termination/reset rules
and finite-value checks. Success prints `HEADLESS_REPLAY_OK`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--checkpoint` | Required | Local checkpoint path |
| `--device` | `cuda:0` | GPU within the visible device list |
| `--port` | 8080 | Local viewer port |
| `--headless` | Off | Run without the viewer |
| `--steps` | 300 | Number of headless steps |
| `--randomization` | Off | Enable training randomization and observation noise |
| `--dry-run` | Off | Print the replay plan |

Checkpoints use PyTorch serialization; load files from trusted sources.
