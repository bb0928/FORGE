"""FORGE August 1 base-policy training implementation.

Use scripts/train.py to apply the recorded milestone profile. The actor outputs
all 29 joint actions directly. Training mixes walking and manipulation commands;
FK targets use a curriculum capped at 0.6 by the release launcher.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rl'))

# Patch rsl-rl resolver BEFORE importing rsl-rl
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
    if name == "forge_data_augmentation":
        from forge_symmetry_func import forge_data_augmentation
        return forge_data_augmentation
    return _orig_resolve(name)

rsl_rl.utils.resolve_callable = _patched_resolve

import torch
import math

from rsl_rl.runners import OnPolicyRunner  # noqa: F401  (let rsl-rl configure multi-gpu)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from commands import MultiModalCommandCfg
from obs_builder import BaseObsBuilder
from forge_model import ForgeActor, ForgeCritic, SplitStdGaussianDistribution  # noqa: F401
from forge_runner import ForgeRunner
from obs_spec import ACTOR_OBS_GROUPS, CRITIC_EXTRA_GROUPS, ACTOR_OBS_DIM, CRITIC_OBS_DIM


# ============================================================
# Config
# ============================================================
NUM_ENVS = int(os.environ.get("NUM_ENVS", "4096"))
MAX_ITER = int(os.environ.get("MAX_ITER", "10000"))
SEED = int(os.environ.get("SEED", "1"))
HAND_ASSET = os.environ.get("FORGE_HAND_ASSET", "dex31_payload").strip().lower()
HAND_XML_PATHS = {
    "dex31_payload": Path(__file__).resolve().parent / "assets" / "g1_xhand" / "xmls" / "g1_dex31_payload.xml",
}
if HAND_ASSET not in HAND_XML_PATHS:
    raise ValueError(f"unknown FORGE_HAND_ASSET={HAND_ASSET!r}; expected one of {tuple(HAND_XML_PATHS)}")
HAND_XML_PATH = HAND_XML_PATHS[HAND_ASSET]
HAND_PAYLOAD_MAX_KG = float(os.environ.get("FORGE_HAND_PAYLOAD_MAX_KG", "0.0"))

DEFAULT_RUN_NAME = "forge-20260801-s7"
RUN_NAME = os.environ.get("RUN_NAME", DEFAULT_RUN_NAME)
LOG_ROOT = Path(os.environ.get("FORGE_LOG_ROOT", str(Path(__file__).resolve().parent / "logs")))
LOG_DIR = str(LOG_ROOT / RUN_NAME)
ENABLE_DOMAIN_RAND = os.environ.get("ENABLE_DOMAIN_RAND", "0") == "1"

if "LOCAL_RANK" in os.environ:
    DEVICE = f"cuda:{int(os.environ['LOCAL_RANK'])}"
else:
    DEVICE = "cuda:0"

STEPS_PER_ENV = 24

# DOF split — MUST match train_forge_rsl.py so the loaded critic aligns.
LOWER = list(range(12)) + [13, 14]
UPPER = [12] + list(range(15, 29))

ACTOR_GROUP_DIMS = [d for _, d in ACTOR_OBS_GROUPS]
CRITIC_GROUP_DIMS = [d for _, d in ACTOR_OBS_GROUPS] + [d for _, d in CRITIC_EXTRA_GROUPS]


# ============================================================
# Environment Wrapper (base mode: frozen base policy returns ZERO)
# ============================================================
class ForgeBaseEnvWrapper(RslRlVecEnvWrapper):
    """Same obs/history pipeline as ForgeResidualEnvWrapper, but the 'base action' is
    always zero — the trained actor is the full-body policy. Keeps obs identical to the
    residual wrapper so the critic obs spec matches exactly."""

    def __init__(self, env, clip_actions=100.0):
        super().__init__(env, clip_actions=clip_actions)
        self.obs_builder = BaseObsBuilder(env, history_length=6, critic_history_length=1)
        self.obs_builder.reset_idx(torch.arange(env.num_envs, device=str(env.device)))
        print(f"ForgeBaseEnvWrapper: {self.num_actions} DOF, base_mode (no frozen base)")

    def get_observations(self):
        return {
            "actor": self.obs_builder.get_actor_obs(),
            "critic": self.obs_builder.get_critic_obs(),
        }

    def reset(self):
        obs_dict, extras = super().reset()
        self.obs_builder.reset_idx(torch.arange(self.num_envs, device=str(self.device)))
        return self.get_observations(), extras

    def step(self, actions):
        obs, rewards, dones, extras = super().step(actions)
        self.obs_builder.update()
        done_ids = torch.where(dones)[0]
        if len(done_ids) > 0:
            self.obs_builder.reset_idx(done_ids)
        return self.get_observations(), rewards, dones, extras

    def get_base_action(self, obs):
        # Base mode: no frozen base. Return zeros so runner's compose = actor output.
        # (Runner uses base_mode=True and won't call this, but keep it safe.)
        return torch.zeros(self.num_envs, self.num_actions, device=str(self.device))


# ============================================================
# RL Config
# ============================================================
def make_train_cfg(multi_gpu=None):
    train_cfg = {
        "runner": {"class_name": "ForgeRunner"},
        "base_mode": True,           # base = full-body policy (no frozen-base composition)
        # v21: curriculum_max capped at 0.5 (was 1.0). V18/V19/V20 all destabilized as
        # curriculum pushed past ~0.5 (upper-target amplitude got too extreme). Cap at 0.5 so
        # the upper-body random course never exceeds the level the policy can hold. Squat depth
        # is now decoupled (full range from the start, see commands.py), so 0.5 still covers the
        # full squat workspace; only the arm-target amplitude is bounded. Step 0.02.
        "curriculum_min": 0.0,
        "curriculum_max": float(os.environ.get("FORGE_CURRICULUM_MAX", "0.8")),  # env override for arm-target curriculum cap
        "curriculum_increment": 0.05,  # V32: HOMIE/FORGE-aligned
        # v14: curriculum advance threshold = 0.8 * curriculum_track_weight, compared against the
        # per-walk-step track_linear reward (= track weight * raw, max = track_weight). The runner
        # default is 4.0 (residual's track weight), but BASE uses track_linear weight 2.0 -> the
        # default made the threshold 0.8*4.0=3.2, which EXCEEDS base's physical max of 2.0, so the
        # condition was never satisfiable and curriculum was frozen at 0 forever. Set to 2.0 to
        # match base's actual track_linear weight -> threshold 0.8*2.0=1.6 (reachable once the gait
        # is accurate enough, which the lowered base_height now pushes for).
        "curriculum_track_weight": 2.0,
        "actor": {
            "class_name": "ForgeActor",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "SplitStdGaussianDistribution",
                "init_std": 1.0,
                "lower_dof_indices": LOWER,
                "upper_dof_indices": UPPER,
                "min_std": 1e-6,
                "max_std": 1e6,
            },
            "actor_group_dims": ACTOR_GROUP_DIMS,
            "critic_group_dims": CRITIC_GROUP_DIMS,
            "lower_dof_indices": LOWER,
            "upper_dof_indices": UPPER,
            "actor_history_length": 6,
            "critic_history_length": 1,
            "transformer_latent_dim": 128,
            "transformer_num_heads": 4,
            "transformer_ff_size": 256,
            "transformer_num_layers": 2,
            "use_transformer_critic": False,
            "use_adaptation": True,
            "adaptation_latent_dim": 64,
            "adaptation_encoder_dims": (512, 256, 128),
            "adaptation_decoder_dims": (128, 64),
            "adaptation_output_dim": 15,
            "init_noise_std": 0.1,
            "min_noise_std": 1e-3,
            "max_noise_std": 1.0,
            "base_policy_path": "",   # no frozen base in base training
            "base_obs_dim": 99,
            "action_scale": 1.0,
        },
        "critic": {
            "class_name": "ForgeCritic",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
            "critic_group_dims": CRITIC_GROUP_DIMS,
            "critic_history_length": 1,
            "transformer_latent_dim": 128,
            "transformer_num_heads": 4,
            "transformer_ff_size": 256,
            "transformer_num_layers": 2,
            "use_transformer_critic": False,
        },
        "algorithm": {
            "class_name": "rl.std_l2_ppo.PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,  # INITIAL only. ForgeRunner overrides this every iter,
                                    # annealing 0.01 -> 0.001 linearly (see entropy_start/end below).
                                    # std-escape is controlled by std_l2 reg, NOT by killing entropy.
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "schedule": "adaptive",
            "gamma": 0.998,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        # Entropy annealing (the REAL values the runner uses; algorithm.entropy_coef is just init).
        "entropy_start": 0.01,
        "entropy_end": 0.001,
        "entropy_decay_iters": 15000,
        # V33 upper-body std L2 regularization strength (curriculum-gated 0 -> std_l2_max).
        "std_l2_max": 0.02,
        # V35 two-stage std-reg gate (see forge_runner): latch on capability, base->max ramp.
        "std_l2_base": 0.02,   # V35.1: 1%->2%, strong enough to bite as soon as gate latches
        "std_l2_gate_iter": 500,
        "std_l2_gate_eplen": 980.0,
        "std_l2_gate_track": 1.0,
        "std_l2_hardstop": 3000,
        "num_steps_per_env": STEPS_PER_ENV,
        "save_interval": 100,
        "experiment_name": RUN_NAME,
        "logger": os.environ.get("FORGE_LOGGER", "tensorboard"),
        "wandb_project": os.environ.get("WANDB_PROJECT", "forge-mjlab"),
        "wandb_tags": ("forge-base", "v57", "dr", HAND_ASSET),
        "obs_groups": {"actor": ("actor",), "critic": ("critic",)},
        "multi_gpu": multi_gpu,
        "action_scale": 1.0,
        "adaptation_loss_coef": 1.0,
    }
    # V22: FORGE-aligned full-body symmetry mirror loss (symmetry_v3, base-yaw frame).
    # Gated by USE_SYMMETRY (default on). Matches FORGE EESymResidualPPO symmetry_scale=1e-3.
    if os.environ.get("USE_SYMMETRY", "1") == "1":
        train_cfg["algorithm"]["symmetry_cfg"] = {
            "data_augmentation_func": "forge_data_augmentation",
            "use_data_augmentation": True,   # V35: flip samples into batch (fix L/R asymmetry + torso lean)
            "use_mirror_loss": True,
            "mirror_loss_coeff": float(os.environ.get("SYMMETRY_COEF", "1e-3")),
        }
    return train_cfg




def _get_hand_spec():
    """Load the selected repo-local G1 hand asset without changing policy DOFs."""
    import mujoco
    if not HAND_XML_PATH.exists():
        raise FileNotFoundError(f"missing {HAND_ASSET} XML: {HAND_XML_PATH}")
    return mujoco.MjSpec.from_file(str(HAND_XML_PATH))


def _apply_hand_asset(cfg):
    """Use a repo-local asset variant instead of editing MJLab's installed G1 XML."""
    cfg.scene.entities["robot"].spec_fn = _get_hand_spec


def _disable_push_randomization(cfg):
    """First DR pass intentionally excludes external pushes."""
    cfg.events.pop("push_robot", None)


def _enable_actuator_delay(cfg, max_policy_steps: int = 1):
    """Enable 0..max_policy_steps command delay using MuJoCo physics-step lag."""
    max_lag = int(max_policy_steps * cfg.decimation)
    for actuator in cfg.scene.entities["robot"].articulation.actuators:
        actuator.delay_min_lag = 0
        actuator.delay_max_lag = max_lag
        actuator.delay_hold_prob = 0.0
        actuator.delay_update_period = 0
        actuator.delay_per_env_phase = True


def _enable_conservative_domain_randomization(cfg):
    """V57 first sim2real pass: conservative DR, no push."""
    from mjlab.envs import mdp as envs_mdp
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    _disable_push_randomization(cfg)
    cfg.events.pop("base_com", None)  # replaced by physically consistent pseudo_inertia below

    foot_geom_names = tuple(
        f"{side}_foot{i}_collision"
        for side in ("left", "right")
        for i in range(1, 8)
    )

    cfg.events["foot_friction"] = EventTermCfg(
        mode="startup",
        func=envs_mdp.dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
            "operation": "abs",
            "ranges": (0.7, 1.3),
            "shared_random": True,
        },
    )
    cfg.events["pd_gain_rand"] = EventTermCfg(
        mode="startup",
        func=envs_mdp.dr.pd_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
            "operation": "scale",
            "kp_range": (0.9, 1.1),
            "kd_range": (0.9, 1.1),
        },
    )
    cfg.events["link_pseudo_inertia"] = EventTermCfg(
        mode="startup",
        func=envs_mdp.dr.pseudo_inertia,
        params={
            # mass scale = exp(2*alpha), so this corresponds to roughly [0.9, 1.1].
            "alpha_range": (0.5 * math.log(0.9), 0.5 * math.log(1.1)),
            "t_range": (-0.02, 0.02),
            # Keep the selected hand inertias exact; randomize the robot links around them.
            "asset_cfg": SceneEntityCfg("robot", body_names=(r"^(?!.*(?:xhand_payload|hand_)).*$",)),
        },
    )
    if HAND_PAYLOAD_MAX_KG > 0.0:
        if HAND_ASSET != "dex31_payload":
            raise ValueError(
                "FORGE_HAND_PAYLOAD_MAX_KG requires FORGE_HAND_ASSET=dex31_payload "
                "so payload bodies exist in the MuJoCo asset."
            )
        cfg.events["hand_payload_mass"] = EventTermCfg(
            mode="startup",
            func=envs_mdp.dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("left_hand_payload", "right_hand_payload"),
                ),
                "operation": "abs",
                "ranges": (0.0, HAND_PAYLOAD_MAX_KG),
            },
        )
    _enable_actuator_delay(cfg, max_policy_steps=1)

def make_env_cfg():
    cfg = unitree_g1_flat_env_cfg(play=False)
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = SEED

    # Command: mixed modes (same proportions as residual) so the base policy covers the
    # manip state distribution (vel=0 + squat). zero_wrist_command keeps wrist targets at 0
    # -> base learns to stand/squat stably without learning manipulation. This fixes the
    # diagnosis that base (walk-only) fell whenever residual env entered manip mode.
    cfg.commands["twist"] = MultiModalCommandCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.1,
        rel_heading_envs=0.3,
        rel_forward_envs=0.2,
        walk_prob=0.8,
        pos_ctrl_prob=0.10,
        force_ctrl_prob=0.10,
        yaw_turn_walk_prob=0.20,
        yaw_turn_abs_min=0.20,
        yaw_turn_abs_max=0.50,
        force_start_iteration=0,
        zero_wrist_command=False,     # base: upper body driven by random course (FK targets on)
        nominal_height=0.74,
        ranges=MultiModalCommandCfg.Ranges(
            lin_vel_x=(-0.8, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.8, 0.8),
            heading=(-math.pi, math.pi),
            wrist_pos_radius=(0.1, 0.3),
            wrist_pos_theta=(0.2 * math.pi, 0.9 * math.pi),
            wrist_force_max=20.0,
            height=(-0.4, 0.0),
        ),
    )

    if "command_vel" in cfg.curriculum:
        cfg.curriculum["command_vel"].params["velocity_stages"] = [
            {"step": 0, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-0.5, 0.5)},
        ]

    _set_base_rewards(cfg)
    _apply_hand_asset(cfg)
    if ENABLE_DOMAIN_RAND:
        _enable_conservative_domain_randomization(cfg)
    else:
        _disable_push_randomization(cfg)
    return cfg


def _set_base_rewards(cfg):
    """KEEP mjlab's official G1 walk reward set (gait shaping fully aligned with mjlab),
    and only:
      1) swap the two velocity-tracking terms for walk-gated versions (必修2),
      2) add FORGE-specific stance terms (base_height tracking, stand_still).

    Rationale: gait/foot shaping (foot_clearance, foot_swing_height, foot_slip,
    soft_landing, variable_posture/pose, upright, self_collisions, body_ang_vel,
    angular_momentum, air_time, dof_pos_limits, action_rate_l2) MUST match mjlab — these
    are what make the mjlab base gait clean. cfg already carries the full mjlab reward dict
    (built by unitree_g1_flat_env_cfg with per-robot site/body/std overrides); we do NOT
    rebuild it from scratch (that previously dropped all foot shaping). We mutate in place.
    """
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    import rewards as fr

    R = cfg.rewards  # mjlab official G1 reward dict (foot shaping etc. already configured)

    # (1) Replace velocity tracking with walk-gated versions (no-op in base since walk-only,
    # but keeps the reward-function identity identical to residual -> V_walk homogeneous for
    # critic warm-start). Reuse mjlab's std values.
    R["track_linear_velocity"] = RewardTermCfg(
        func=fr.track_linear_velocity_walk,
        weight=2.0,
        params={"command_name": "twist", "std": math.sqrt(0.25)},
    )
    R["track_angular_velocity"] = RewardTermCfg(
        func=fr.track_angular_velocity_walk,
        weight=2.0,
        params={"command_name": "twist", "std": math.sqrt(0.5)},
    )

    # (2) FORGE-specific stance terms — DISABLED for V6 (user decision): both
    # tracking_base_height and stand_still commented out, so base = pure mjlab gait set
    # + walk-gated tracking only. Goal: see if std-fix alone lets tracking reach mjlab's
    # ~1.8-1.9 without these extra constraints competing for it.
    # base v8: ADD tracking_base_height (supervises the squat command in manip mode so the
    # base learns to stand at commanded heights / squat). Keep stand_still OFF for now
    # (control variables: only add base_height this round).
    # v14: base_height weight 2.0 -> 0.5. In v13 base_height (2.0) was a large reward the policy
    # could collect by just standing/holding height — it short-circuited the incentive to walk
    # ACCURATELY. Result: episode_length saturated to 1000 very early (iter~280) while velocity
    # tracking plateaued at raw~0.75 (per-walk-step ~1.5, never crossing the 1.6 curriculum
    # threshold) and std stalled at ~0.56. v7 (no base_height, pure walk) instead "earned" its
    # gait the hard way and reached track 1.69 / std 0.32. Lowering base_height restores walk
    # tracking (weight 2.0) as the dominant objective so the policy must walk precisely -> track
    # climbs past 1.6 -> curriculum advances -> std keeps annealing.
    R["tracking_base_height"] = RewardTermCfg(
        func=fr.tracking_base_height,
        weight=2.0,
        # v21: drop manip_scale -> flat weight 2.0 in BOTH walk and manip (no mode-dependent
        # amplification). Squat depth is now full-range from the start (commands.py), so a
        # constant strong base_height drive is appropriate; removing the manip ramp keeps the
        # reward scale stable (one less thing inflating return as the episode mix shifts).
        params={"command_name": "twist", "sensor_name": "feet_ground_contact",
                "ankle_sole_distance": 0.02, "manip_scale": 1.0},
    )

    # NEW design: base does the SAME upper-body task as residual (track FK wrist/elbow
    # target), just in the easy curriculum range (0->0.5). Same reward funcs/weights as
    # train_forge_rsl so base and residual share the objective (warm-start consistency).
    _WRIST_BODIES = SceneEntityCfg("robot", body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"))
    _UPPER_LINK_BODIES = SceneEntityCfg(
        "robot",
        body_names=(
            "left_shoulder_roll_link", "right_shoulder_roll_link",
            "left_elbow_link", "right_elbow_link",
            "left_wrist_yaw_link", "right_wrist_yaw_link",
        ),
    )
    # squatv3: restore wrist orientation tracking after pose guard stabilized late training.
    R["tracking_wrist_pos"] = RewardTermCfg(
        func=fr.tracking_wrist_pos,
        weight=1.0,
        params={"command_name": "twist", "sigma": 0.20, "asset_cfg": _WRIST_BODIES},
    )
    R["tracking_wrist_ori"] = RewardTermCfg(
        func=fr.tracking_wrist_ori,
        weight=1.0,
        params={"command_name": "twist", "sigma": 0.30, "asset_cfg": _WRIST_BODIES},
    )
    R["tracking_upper_link_pos"] = RewardTermCfg(
        func=fr.tracking_upper_link_pos,
        weight=1.0,
        params={
            "command_name": "twist",
            "std": 0.30,
            # order: L/R shoulder_roll_link, L/R elbow_link, L/R wrist_yaw_link
            "link_weights": (2.0, 2.0, 2.0, 2.0, 0.5, 0.5),
            "asset_cfg": _UPPER_LINK_BODIES,
        },
    )
    R["tracking_upper_link_ori"] = RewardTermCfg(
        func=fr.tracking_upper_link_ori,
        weight=1.0,
        params={
            "command_name": "twist",
            "std": 0.40,
            # order: L/R shoulder_roll_link, L/R elbow_link, L/R wrist_yaw_link
            "link_weights": (1.5, 1.5, 2.0, 2.0, 0.5, 0.5),
            "asset_cfg": _UPPER_LINK_BODIES,
        },
    )
    R["tracking_elbow_pos"] = RewardTermCfg(
        func=fr.tracking_elbow_pos,
        weight=2.0,
        params={"command_name": "twist", "sigma": 0.25,
                "asset_cfg": SceneEntityCfg("robot", body_names=("left_elbow_link", "right_elbow_link"))},
    )

    if os.environ.get("WRIST_TRACKING_TERMINATION", "0") == "1":
        from mjlab.managers.termination_manager import TerminationTermCfg
        cfg.terminations["wrist_tracking_bad"] = TerminationTermCfg(
            func=fr.WristTrackingBadTermination,
            params={
                "command_name": "twist",
                "pos_threshold_m": float(os.environ.get("WRIST_TERM_POS_CM", "20.0")) / 100.0,
                "ori_threshold_rad": math.radians(float(os.environ.get("WRIST_TERM_ORI_DEG", "90.0"))),
                "grace_steps": int(os.environ.get("WRIST_TERM_GRACE_STEPS", "10")),
                "warmup_steps": int(os.environ.get("WRIST_TERM_WARMUP_STEPS", "50")),
                "asset_cfg": _WRIST_BODIES,
            },
        )

    # v11: REMOVED the v10 stand_still / stand_still_ang_vel penalties.
    # v10 data (it=345) showed they were pure drag and fought the upper-body tracking task:
    # mean_reward went NEGATIVE (-8.4 vs v9 +11.4), std stuck at 0.997 (no annealing), and
    # the body "got distracted" — track_linear/base_height/upright/pose all regressed while
    # the policy thrashed (action_rate -2.79) trying to satisfy both "hold still" and "move
    # the arm to the target". The two goals are contradictory.
    # The precision pressure for manip is now supplied entirely by the wrist/elbow TRACKING
    # task itself (tracking_wrist_pos +0.77 was already learning) — that is the proper,
    # non-contradictory pressure. So manip = squat + track FK wrist/elbow target, no
    # stand-still crutch. Clean control variable vs v9: only the (fixed) tracking task is added.

    # v15: swap the pose reward (mjlab variable_posture) for a manip-squat-aware version.
    # In manip mode the vel command is ~0 -> variable_posture uses std_standing (legs locked
    # to UPRIGHT default), but manip also commands a squat (height < nominal) which REQUIRES
    # the sagittal leg chain to bend -> conflict -> extreme leg poses (the observed anomaly).
    # variable_posture_squat relaxes hip_pitch/knee/ankle_pitch std to 0.3 ONLY when manip+
    # squatting; roll/yaw + ankle_roll + waist stay tight (balance not loosened). Keeps the
    # official std_standing/walking/running configs (set on R["pose"] by unitree_g1_flat_env_cfg).
    if "pose" in R:
        _pose = R["pose"]
        _pose.func = fr.variable_posture_squat
        _pose.params["command_name"] = "twist"
        _pose.params["squat_std"] = 0.3
        pose_guard_weight = float(os.environ.get("FORGE_POSE_GUARD_WEIGHT", "0.0"))
        if pose_guard_weight != 0.0:
            pose_guard_params = dict(_pose.params)
            pose_guard_params.update({
                "warn_threshold": float(os.environ.get("FORGE_POSE_GUARD_WARN", "0.60")),
                "danger_threshold": float(os.environ.get("FORGE_POSE_GUARD_DANGER", "0.80")),
                "danger_scale": float(os.environ.get("FORGE_POSE_GUARD_DANGER_SCALE", "4.0")),
                "exclude_squat_chain": os.environ.get("FORGE_POSE_GUARD_EXCLUDE_SQUAT_CHAIN", "0") == "1",
                "start_iteration": int(os.environ.get("FORGE_POSE_GUARD_START_ITER", "0")),
                "upper_only": os.environ.get("FORGE_POSE_GUARD_UPPER_ONLY", "0") == "1",
                "disable_during_squat": os.environ.get("FORGE_POSE_GUARD_DISABLE_DURING_SQUAT", "0") == "1",
            })
            R["pose_guard"] = RewardTermCfg(
                func=fr.pose_deviation_guard,
                weight=pose_guard_weight,
                params=pose_guard_params,
            )

    # V55 hardened default: MJLab soft_landing with V54-rmk weight, no env tail.
    if "soft_landing" in R:
        R["soft_landing"].weight = -0.0003

    # V56: yaw-only stepping incentives. These are disabled for linear walk, manip, and walk still.
    R["yaw_turn_no_fly"] = RewardTermCfg(
        func=fr.yaw_turn_no_fly,
        weight=0.50,
        params={
            "command_name": "twist",
            "sensor_name": "feet_ground_contact",
            "yaw_threshold": 0.15,
            "lin_threshold": 0.10,
            "contact_threshold": 1.0,
        },
    )
    R["yaw_turn_feet_air_time"] = RewardTermCfg(
        func=fr.yaw_turn_feet_air_time,
        weight=0.03,
        params={
            "command_name": "twist",
            "sensor_name": "feet_ground_contact",
            "threshold_min": 0.05,
            "threshold_max": 0.18,
            "yaw_threshold": 0.15,
            "lin_threshold": 0.10,
        },
    )

    # V23: manip-squat anti-splay. Legs splay (knee lat 0.75m vs 0.21m standing) to balance
    # the upper-body manip motion during squat. Cap knee/feet lateral separation in base-yaw
    # frame, ONLY in manip mode (walking gait untouched). Negative weight = penalty when over.
    R["knee_distance_lateral_manip"] = RewardTermCfg(
        func=fr.knee_distance_lateral_manip,
        weight=-2.0,
        params={"command_name": "twist", "most": 0.30, "walk_still_most": 0.25, "walk_still_scale": 3.0},
    )
    R["feet_distance_lateral_manip"] = RewardTermCfg(
        func=fr.feet_distance_lateral_manip,
        weight=-1.0,
        params={"command_name": "twist", "most": 0.35, "walk_still_most": 0.30, "walk_still_scale": 3.0},
    )
    # V24-B: directly penalize hip roll/yaw splay in manip mode (the residual left-leg
    # outward-rotation that symmetry + lateral cap didn't fully kill).
    R["hip_roll_yaw_deviation_manip"] = RewardTermCfg(
        func=fr.hip_roll_yaw_deviation_manip,
        weight=-1.0,
        params={"command_name": "twist", "walk_still_scale": 3.0},
    )
    # V25: stand_still (FORGE-aligned) - penalize feet leaving ground in manip/standing.
    # Treats the trembling/foot-shuffle during stand-manip that fell_over/episode_len miss.
    R["stand_still"] = RewardTermCfg(
        func=fr.stand_still,
        weight=-1.0,
        params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
    )


    # V55 hardened default: FORGE-style IMU angular-velocity stand-still, manip-only in rewards.py.
    R["imu_stand_still"] = RewardTermCfg(
        func=fr.imu_stand_still,
        weight=-0.10,
        params={
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            "manip_scale": 1.0,
        },
    )

    # V55: no custom no_fly/feet_air_time injection. Walk/yaw stays MJLab-native.


    # v18: reverted the v17 leg symmetry/contact rewards (feet_parallel/ground_parallel/
    # distance_lateral/knee_distance_lateral/no_fly). They helped early but the run drifted
    # down after ~iter400 (std crept up, episode fell) vs V16 which stayed stable. So V18 =
    # V16 reward set + the slow full-range curriculum (increment 0.02, max 1.0). The leg
    # reward funcs remain in rewards.py for later, just not wired in here.

def main():
    multi_gpu = None
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size > 1:
            multi_gpu = "ddp"
            print(f"Multi-GPU detected: rank={rank}, world_size={world_size}")

    os.makedirs(LOG_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    print(f"=== FORGE BASE (mixed commands, standalone full-body policy) ===")
    print(
        f"Run: {RUN_NAME}, Seed: {SEED}, Envs: {NUM_ENVS}, Iters: {MAX_ITER}, "
        f"Multi-GPU: {multi_gpu}, DR: {ENABLE_DOMAIN_RAND}, "
        f"HandAsset: {HAND_ASSET}, HandXML: {HAND_XML_PATH}"
    )

    print("\n=== Creating environment ===")
    env_cfg = make_env_cfg()
    env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE)

    print("=== Wrapping environment (base mode) ===")
    wrapped_env = ForgeBaseEnvWrapper(env)

    train_cfg = make_train_cfg(multi_gpu=multi_gpu)

    print("=== Creating runner (base_mode=True) ===")
    runner = ForgeRunner(wrapped_env, train_cfg, log_dir=LOG_DIR, device=DEVICE)

    # v21: optional warm-start from a prior base checkpoint (continue training, not from
    # scratch). load_cfg loads actor+critic+optimizer + restores curriculum_ratio from infos.
    resume = os.environ.get("RESUME_CKPT", "")
    if resume:
        print(f"=== RESUME from {resume} ===")
        runner.load(resume, load_cfg={"actor": True, "critic": True, "optimizer": True, "iteration": True}, strict=False, map_location=DEVICE)
        # V55 hardened default: resume checkpoints are pinned to the code default curriculum cap.
        runner.curriculum_ratio = runner.curriculum_max
        print(f"    curriculum_ratio={runner.curriculum_ratio:.2f} (max={runner.curriculum_max})")

    print(f"\n=== Training {MAX_ITER} iterations ===")
    runner.learn(num_learning_iterations=MAX_ITER, init_at_random_ep_len=True)

    print("\n=== DONE ===")
    env.close()


if __name__ == "__main__":
    main()
