"""Fail early on incompatible packages, missing resources or unavailable CUDA."""
from __future__ import annotations

import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "rl")]


def main():
    expected = {"mjlab": "1.4.0", "rsl-rl-lib": "5.2.0", "mujoco": "3.8.0",
                "mujoco-warp": "3.8.0.3", "warp-lang": "1.12.0"}
    versions, errors = {}, []
    for name, version in expected.items():
        try:
            versions[name] = metadata.version(name)
            if versions[name] != version:
                errors.append(f"{name}: expected {version}, found {versions[name]}")
        except metadata.PackageNotFoundError:
            errors.append(f"Missing package: {name}")
    if platform.system() != "Linux":
        errors.append("Training/replay requires Linux with NVIDIA CUDA; CPU-only tests are separate.")
    if errors:
        raise SystemExit("Runtime preflight failed:\n- " + "\n- ".join(errors))
    import torch
    import mujoco
    import warp
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable to PyTorch.")
    required = int(os.environ.get("FORGE_REQUIRED_GPUS", "1"))
    if torch.cuda.device_count() < required:
        raise SystemExit(f"Need {required} visible CUDA devices; found {torch.cuda.device_count()}.")
    from train_forge_base import HAND_XML_PATH, make_env_cfg, make_train_cfg
    from rl.std_l2_ppo import PPO
    cfg = make_env_cfg()
    train_cfg = make_train_cfg()
    assert train_cfg["algorithm"]["class_name"] == "rl.std_l2_ppo.PPO"
    model = mujoco.MjModel.from_xml_path(str(HAND_XML_PATH))
    # Actuators are attached by mjlab's EntityCfg, not embedded in this XML.
    assert (model.nq, model.nv, model.njnt) == (36, 35, 30)
    for side in ("left", "right"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side + "_hand_payload") >= 0
    versions.update(torch=torch.__version__, cuda=torch.version.cuda,
                    python=platform.python_version(), visible_gpus=torch.cuda.device_count())
    print(json.dumps({"RUNTIME_PREFLIGHT_OK": versions, "algorithm": PPO.__module__,
                      "num_envs": cfg.scene.num_envs}, indent=2))


if __name__ == "__main__":
    main()
