# Validation status — 2026-10-17

Status: **cleaned source release candidate; CPU integration passed; CUDA milestone
reproduction not yet performed.**

## Completed

| Check | Result |
|---|---|
| Unit/contract suite | 10 tests passed, zero skipped in the QA environment |
| Python syntax | All packaged Python modules parse/compile |
| Dependency consistency | `pip check`: no broken requirements |
| Asset closure | All 49 referenced meshes packaged; XML compiles in MuJoCo 3.8.0 |
| Robot structure | 29 articulated joints plus floating base; nq=36, nv=35; mjlab adds actuators |
| Model contract | Actor 852 → 29, critic 173 → 1; finite outputs and exact state-dict roundtrip |
| Symmetry | Applying the mirror twice restores inputs/actions; masks retained |
| Launcher behavior | Dry run, paths with spaces/metacharacters, stale environment cleanup, collision rejection, no writes after failed preflight |
| CPU training integration | Real mjlab environment, two environments, four rollout steps, one PPO update, checkpoint saved |
| CPU late-stage branch | Explicit iteration 1001 / curriculum 0.6; one PPO update with finite losses; StdL2 loss nonzero |
| Checkpoint replay | Saved CPU smoke checkpoint loaded strictly; 12 headless steps passed finite checks |
| Interactive viewer startup | Real viewer setup and FK overlay initialized; localhost HTTP 200, checkpoint curriculum 0.6 restored |
| Source preservation | 65/67 imported files byte-identical; only trainer and viewer changed |
| Training semantics | AST equality confirmed for environment wrapper, environment factory, reward setup, domain randomization and actuator-delay setup |
| Content scan | No matched private service addresses, original workstation/mount paths, embedded key/token patterns or private-key blocks in release content |

The late-stage CPU check **sets the iteration counter** to exercise the delayed
pose-guard/StdL2 path; it does not claim that 1,001 training iterations were run.
CPU checkpoints were temporary QA products and are not included as pretrained
milestone weights. Viewer startup was checked programmatically; browser image
quality was not visually inspected.

## QA runtime

Python 3.12.14, macOS ARM64 / CPU. Core packages:

- PyTorch 2.7.1, torchvision 0.22.1
- mjlab 1.4.0, RSL-RL 5.2.0
- MuJoCo 3.8.0, MuJoCo Warp 3.8.0.3, Warp 1.12.0
- NumPy 2.2.6, SciPy 1.15.3, TensorDict 0.14.3
- Viser 1.1.1, mjviser 0.0.14, TensorBoard 2.21.0

The CPU integration probe invoked the implementation directly with device `cpu`
and a shortened rollout. The released launchers intentionally require Linux/CUDA.
No claim is made that the CPU probe validates CUDA kernels, NCCL or DDP.

## Remaining evidence needed for a reproduced milestone release

1. Run the supplied single-GPU smoke and headless replay commands on Linux/CUDA.
2. Run four-GPU DDP with the documented full profile and evaluate the resulting
   checkpoint. Different batch topology or reconstructed simulator dependencies
   may change outcomes.
3. If distributing the historical result, supply the original August 1
   `model_9999.pt`, publish its SHA256, and verify it with this environment.

The original complete environment freeze was not available. Current pins were
reconstructed from the matching PPO source and G1 asset, then checked locally;
they are not represented as the original machine's complete dependency state.
