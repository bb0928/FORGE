# Installation

[Overview](../README.md) · [Training](training.md)

## Environment

Use Linux, Python 3.12 and an NVIDIA GPU. The default training recipe uses four
GPUs; the smoke test and viewer use one.

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python scripts/check_runtime.py
```

The runtime check inspects package versions, visible CUDA devices, the configured
PPO implementation and robot assets. Success prints `RUNTIME_PREFLIGHT_OK`.
Training and playback run this check automatically.

The environment uses mjlab 1.4.0, RSL-RL 5.2.0 and MuJoCo 3.8.0. All package pins
are listed in [requirements.txt](../requirements.txt).

## GPU selection

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
```

PyTorch should report CUDA as available. `CUDA_VISIBLE_DEVICES` selects physical
GPUs; `--device` refers to their indices within that visible list. For example,
`CUDA_VISIBLE_DEVICES=2` exposes physical GPU 2 as `cuda:0`.

## Troubleshooting

| Symptom | Action |
|---|---|
| `CUDA is unavailable to PyTorch` | Activate the environment, install the CUDA PyTorch wheel above, and check the NVIDIA driver. |
| `Need 4 visible CUDA devices` | Expose four GPUs or select `--gpus 1`. |
| Version mismatch | Install `requirements.txt` in a fresh environment. |
| GPU memory exhausted | Start with `--gpus 1 --smoke`, then choose a suitable `--num-envs`. |
| Slow first initialization | Warp compiles simulation kernels on first use. Inspect the terminal for progress or errors. |
| Viewer inaccessible on a remote machine | Forward its port as described in [Playback](playback.md#remote-viewing). |

## Development tests

```bash
python -m unittest discover -s tests -v
```

For CPU tests in a separate environment, install `torch==2.7.1` and
`mujoco==3.8.0`. Neural and XML tests require these packages; otherwise they are
explicitly skipped. Training and playback launchers use Linux/CUDA.
