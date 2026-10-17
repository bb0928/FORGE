"""Interactive or headless replay using the training observation/action contract.

Use scripts/play.py to select the milestone profile and a local checkpoint.
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rl'))

# Patch rsl-rl resolver BEFORE importing rsl-rl (same as training).
import rsl_rl.utils
_orig_resolve = rsl_rl.utils.resolve_callable

def _patched_resolve(name):
    if name == "ForgeRunner":
        from forge_runner import ForgeRunner
        return ForgeRunner
    if name == "ForgeActor":
        from forge_model import ForgeActor
        return ForgeActor
    if name == "ForgeCritic":
        from forge_model import ForgeCritic
        return ForgeCritic
    if name == "SplitStdGaussianDistribution":
        from forge_model import SplitStdGaussianDistribution
        return SplitStdGaussianDistribution
    return _orig_resolve(name)

rsl_rl.utils.resolve_callable = _patched_resolve

import torch  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.viewer.viser import ViserPlayViewer  # noqa: E402
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago  # noqa: E402

# Reuse the EXACT env wrapper + train cfg + env cfg used for base training.
from train_forge_base import ForgeBaseEnvWrapper, make_train_cfg, make_env_cfg  # noqa: E402
from forge_runner import ForgeRunner  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Play a FORGE-base checkpoint in viser")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model_*.pt. If unset, auto-find latest in --log-dir.")
    parser.add_argument("--log-dir", type=str, default=os.path.join(os.path.dirname(__file__), "logs", "forge-20260801-s7"),
                        help="Directory with model_*.pt checkpoints")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device (pick a free GPU)")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    if args.checkpoint is None:
        from pathlib import Path
        ckpts = sorted(Path(args.log_dir).glob("model_*.pt"),
                       key=lambda p: int(p.stem.split("_")[1]))
        if not ckpts:
            print(f"ERROR: no model_*.pt in {args.log_dir}"); sys.exit(1)
        args.checkpoint = str(ckpts[-1])
        print(f"[INFO] auto-selected latest: {args.checkpoint}")

    print(f"=== FORGE BASE play ===\n  ckpt={args.checkpoint}\n  device={args.device}\n  port={args.port}")

    # Env (play mode, single env). 100% manual reset: disable env auto-reset and
    # remove termination terms BEFORE ManagerBasedRlEnv builds its TerminationManager.
    cfg = make_env_cfg()
    cfg.scene.num_envs = args.num_envs
    try:
        cfg.auto_reset = not args.headless
        cfg.episode_length_s = cfg.episode_length_s if args.headless else 1e9
        cfg.terminations = cfg.terminations if args.headless else {}
    except Exception as _e:
        print("[play] pre-env disable-reset warn:", _e)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
        env.cfg.auto_reset = not args.headless
        env.cfg.episode_length_s = env.cfg.episode_length_s if args.headless else 1e9
        env.cfg.terminations = env.cfg.terminations if args.headless else {}
        if not args.headless and hasattr(env, "max_episode_length"):
            env.max_episode_length = int(1e9)
    except Exception as _e:
        print("[play] post-env disable-reset warn:", _e)

    # SAME wrapper as base training (base_mode: actor output IS the full action).
    wrapped_env = ForgeBaseEnvWrapper(env)

    # Runner just for checkpoint loading (base_mode=True via make_train_cfg).
    train_cfg = make_train_cfg()
    runner = ForgeRunner(wrapped_env, train_cfg, log_dir=None, device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)

    # Base inference policy: outputs the full 29-DOF action directly. No composition.
    raw_policy = runner.get_inference_policy(device=args.device)
    print(f"[INFO] base policy loaded; curriculum_ratio(from ckpt)={runner.curriculum_ratio:.2f}")

    if args.headless:
        from tensordict import TensorDict
        try:
            obs, _ = wrapped_env.reset()
            with torch.inference_mode():
                for _ in range(args.steps):
                    obs = TensorDict(obs, batch_size=env.num_envs)
                    action = raw_policy(obs)
                    if not torch.isfinite(action).all():
                        raise RuntimeError("Non-finite replay action")
                    obs, reward, done, extras = wrapped_env.step(action)
                    if not torch.isfinite(reward).all():
                        raise RuntimeError("Non-finite replay reward")
            print(f"HEADLESS_REPLAY_OK steps={args.steps}")
        finally:
            env.close()
        return

    # --- slider state + policy wrapper that injects commands every step ---
    class _Sliders:
        vx = 0.0; vy = 0.0; wz = 0.0
        height = 0.74      # nominal; lower -> squat
        force_manip = True  # force pos-ctrl (manip) mode so height/squat is active
        curriculum = min(runner.curriculum_ratio, runner.curriculum_max)
    sliders = _Sliders()
    _holder = {"policy": raw_policy}

    # ---- FK-vs-actual arm overlay (added: visualize reference pose) ----
    _viz = {"server": None, "on": True, "ids": None, "scene": None}
    _robot = env.scene.entities["robot"]
    class _PlayPolicy:
        def __call__(self, obs):
            cmd = env.command_manager.get_term("twist")
            s = sliders
            # LOCK OUT internal random command overrides/resampling for both walk and manip.
            # PlayBase must be 100% slider/manual controlled; no timed command reset.
            if hasattr(cmd, "is_standing_env"): cmd.is_standing_env[:] = False
            if hasattr(cmd, "is_heading_env"):  cmd.is_heading_env[:] = False
            if hasattr(cmd, "is_forward_env"):  cmd.is_forward_env[:] = False
            if hasattr(cmd, "is_world_env"):    cmd.is_world_env[:] = False
            try:
                cmd.cfg.resampling_time_range = (1e9, 1e9)
                if hasattr(cmd, "time_left"):
                    cmd.time_left[:] = 1e9
            except Exception:
                pass
            if s.force_manip:
                # put all envs into pos-ctrl (manip): vel forced 0 by _update_command,
                # height_command drives the squat the slider asks for.
                cmd.is_pos_ctrl_mode[:] = True
                cmd.is_force_ctrl_mode[:] = False
                cmd.is_walk_mode[:] = False
                cmd.vel_command_b[:, :] = 0.0
            else:
                cmd.is_walk_mode[:] = True
                cmd.is_pos_ctrl_mode[:] = False
                cmd.is_force_ctrl_mode[:] = False
                cmd.vel_command_b[:, 0] = s.vx
                cmd.vel_command_b[:, 1] = s.vy
                cmd.vel_command_b[:, 2] = s.wz
            cmd.height_command[:] = s.height
            # drive the upper-body random course amplitude (same as training curriculum)
            if hasattr(cmd, "set_curriculum_ratio"):
                cmd.set_curriculum_ratio(s.curriculum)
            _out = _holder["policy"](obs)
            return _out
        def reset(self):
            r = getattr(_holder["policy"], "reset", None)
            if r: r()
    policy = _PlayPolicy()

    _ov_ids = {}
    def _overlay_official(sc):
        if not _viz.get("on", True):
            return
        import numpy as _np
        d = _robot.data
        cmd = env.command_manager.get_term("twist")
        if not _ov_ids:
            def _b(n):
                r=_robot.find_bodies(n); return r[0][0] if hasattr(r[0],"__len__") else r[0]
            _ov_ids.update(lw=_b("left_wrist_yaw_link"), rw=_b("right_wrist_yaw_link"),
                           le=_b("left_elbow_link"), re=_b("right_elbow_link"))
        tw=cmd.wrist_pose_command[0].detach().cpu().numpy()
        te=cmd.elbow_pos_command[0].detach().cpu().numpy()
        def ap(b): return d.body_link_pos_w[0,b].detach().cpu().numpy()
        RED=(0.9,0.1,0.1,1.0); GRN=(0.1,0.85,0.2,1.0); YEL=(0.95,0.85,0.1,1.0)
        for tgt,act in [(tw[0:3],ap(_ov_ids["lw"])),(tw[3:6],ap(_ov_ids["rw"])),
                        (te[0:3],ap(_ov_ids["le"])),(te[3:6],ap(_ov_ids["re"]))]:
            sc.add_sphere(center=tgt, radius=0.04, color=RED)
            sc.add_sphere(center=act, radius=0.03, color=GRN)
            if _np.linalg.norm(tgt-act) > 1e-4:
                sc.add_arrow(start=act, end=tgt, color=YEL, width=0.008)
        if not hasattr(_overlay_official,"_n"): _overlay_official._n=0
        _overlay_official._n+=1
        if _overlay_official._n % 120 == 1:
            print(f"[overlay] frame {_overlay_official._n} queued spheres ok", flush=True)

    # Live checkpoint switching in the viser UI.
    ckpt_dir = os.path.dirname(args.checkpoint)

    def fetch_available_local():
        now = time.time(); entries = []
        for f in sorted([p for p in os.listdir(ckpt_dir) if p.startswith("model_") and p.endswith(".pt")],
                        key=lambda n: int(n.split("_")[1].split(".")[0])):
            ago = format_time_ago(int(now - os.path.getmtime(os.path.join(ckpt_dir, f))))
            entries.append((f, ago))
        return entries

    def reload_checkpoint(name):
        runner.load(os.path.join(ckpt_dir, name), load_cfg={"actor": True}, strict=True,
                    map_location=args.device)
        _holder["policy"] = runner.get_inference_policy(device=args.device)
        sliders.curriculum = min(runner.curriculum_ratio, runner.curriculum_max)
        g_c.value = sliders.curriculum
        print(f"[INFO] switched to {name}")
        return policy   # return the wrapper, not the raw policy

    ckpt_manager = CheckpointManager(
        current_name=os.path.basename(args.checkpoint),
        fetch_available=fetch_available_local,
        load_checkpoint=reload_checkpoint,
    )

    print(f"\n  Open http://localhost:{args.port}\n")
    # Build our own server on the requested port (ViserPlayViewer's internal default is 8081
    # and ignores any port arg, which collides when running base + residual side by side).
    import viser
    server = viser.ViserServer(label="forge-base", host="127.0.0.1", port=args.port)
    _viz["server"] = server
    with server.gui.add_folder("FK reference overlay"):
        g_viz = server.gui.add_checkbox("show FK target vs actual", initial_value=True)
    @g_viz.on_update
    def _(_):
        _viz["on"] = g_viz.value
        if not g_viz.value:
            # clear overlay nodes when turned off
            for nm in ["L_wrist","R_wrist","L_elbow","R_elbow"]:
                for suf in ["_tgt","_act","_err"]:
                    try: server.scene.add_icosphere(f"/fk/{nm}{suf}", radius=0.001, color=(0,0,0), position=(0,0,-100))
                    except Exception: pass

    # --- custom sliders ---
    with server.gui.add_folder("Mode"):
        g_manip = server.gui.add_checkbox("force manip (squat enabled)", initial_value=True)
    with server.gui.add_folder("Height (squat)"):
        g_h = server.gui.add_slider("base height (m)", min=0.34, max=0.78, step=0.01, initial_value=0.74)
    with server.gui.add_folder("Manipulation (upper-body course)"):
        g_c = server.gui.add_slider("upper curriculum", min=0.0, max=runner.curriculum_max, step=0.02, initial_value=sliders.curriculum)
    with server.gui.add_folder("Locomotion (when NOT manip)"):
        g_vx = server.gui.add_slider("lin_vel_x", min=-1.0, max=1.5, step=0.05, initial_value=0.0)
        g_vy = server.gui.add_slider("lin_vel_y", min=-0.6, max=0.6, step=0.05, initial_value=0.0)
        g_wz = server.gui.add_slider("ang_vel_z", min=-1.5, max=1.5, step=0.05, initial_value=0.0)

    @g_manip.on_update
    def _(_): sliders.force_manip = g_manip.value
    @g_h.on_update
    def _(_): sliders.height = g_h.value
    @g_c.on_update
    def _(_): sliders.curriculum = g_c.value
    @g_vx.on_update
    def _(_): sliders.vx = g_vx.value
    @g_vy.on_update
    def _(_): sliders.vy = g_vy.value
    @g_wz.on_update
    def _(_): sliders.wz = g_wz.value

    # Disable mjlab's auto command-term GUI (it writes vel_command_b, conflicting with our
    # locomotion sliders). Our _PlayPolicy is the single source of vel/height/curriculum.
    try:
        env.command_manager.create_gui = lambda *a, **k: None
    except Exception:
        pass
    viewer = ViserPlayViewer(env=wrapped_env, policy=policy, viser_server=server,
                             checkpoint_manager=ckpt_manager)
    # Install overlay hook AFTER scene is built. _scene is created in viewer.setup()
    # (inside run()), so wrap setup to attach our scene.update hook at the right time.
    _orig_setup = viewer.setup
    def _patched_setup(*a, **k):
        _orig_setup(*a, **k)
        try:
            _sc = viewer._scene
            _viz["scene"] = _sc
            _sc.debug_visualization_enabled = True
            _orig_u = _sc.update
            def _hooked(sim_data, *aa, **kk):
                try:
                    _sc.clear_debug_all()
                    _overlay_official(_sc)
                except Exception as _e:
                    print("[overlay] err:", _e)
                return _orig_u(sim_data, *aa, **kk)
            _sc.update = _hooked
            print("[overlay] hook installed after setup", flush=True)
        except Exception as _e:
            print("[overlay] hook fail:", _e)
    viewer.setup = _patched_setup
    viewer.run()
    env.close()


if __name__ == "__main__":
    main()
