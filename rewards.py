"""Custom reward terms for FORGE residual environment.

Manipulation rewards: wrist tracking, elbow tracking, force tracking,
virtual target tracking, action vanish, and force smoothness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg


# ============================================================
# Walk-gated velocity tracking (必修2)
# ============================================================
# In manip mode the velocity command is zeroed (robot stands to manipulate), so the raw
# mjlab track_linear/angular_velocity reward saturates (~max) for free. That free reward
# (a) pollutes the critic's value estimate across modes and (b) inflates any walk metric.
# These wrappers gate the tracking reward by is_walk_mode so it ONLY scores during walking;
# in manip mode it returns 0 and the manip/stand rewards take over.

def track_linear_velocity_walk(env, std: float, command_name: str) -> torch.Tensor:
    from mjlab.tasks.velocity import mdp as _vmdp
    r = _vmdp.track_linear_velocity(env, std=std, command_name=command_name)
    cmd = env.command_manager.get_term(command_name)
    if hasattr(cmd, "is_walk_mode"):
        r = r * cmd.is_walk_mode.float()
    return r


def track_angular_velocity_walk(env, std: float, command_name: str) -> torch.Tensor:
    from mjlab.tasks.velocity import mdp as _vmdp
    r = _vmdp.track_angular_velocity(env, std=std, command_name=command_name)
    cmd = env.command_manager.get_term(command_name)
    if hasattr(cmd, "is_walk_mode"):
        r = r * cmd.is_walk_mode.float()
    return r


def _quat_geodesic_sq(q_actual, q_target):
    """Squared geodesic angle between two quats (wxyz). Both (B,4). Returns (B,).
    angle = 2*acos(|<q1,q2>|); we use 1-dot^2 as a smooth proxy (~ (angle/2)^2 near 0,
    monotonic, no acos NaN). Sign-invariant (q and -q same rotation)."""
    import torch as _t
    dot = _t.sum(q_actual * q_target, dim=-1)          # (B,)
    return 1.0 - dot * dot                              # in [0,1], 0 = aligned


def tracking_wrist_ori(
    env: "ManagerBasedRlEnv",
    command_name: str,
    sigma: float,
    asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """V36: track wrist WORLD ORIENTATION (additive, pairs with tracking_wrist_pos).

    reward = exp(-ori_err / sigma) * scale
    ori_err = geodesic(actual_wrist_quat, wrist_quat_command) for L+R, summed.

    Replaces the V35 tracking_wrist_joint (which tracked JOINT ANGLES -> IK-redundancy
    conflict with position -> policy dropped wrist & used extreme shoulder). Tracking the
    world orientation of wrist_yaw_link (the terminal EE body) is the natural 6D-pose
    complement to position. Target wrist_quat_command already exists in the command (used as
    obs) -> NO command change. Equal weight to wrist_pos, larger sigma (looser) so position
    stays priority. BeyondMimic-style additive pos+ori with comparable std. Active in
    pos/force ctrl (0.2x walk).
    """
    cmd = env.command_manager.get_term(command_name)
    if not hasattr(cmd, "wrist_quat_command") or not hasattr(cmd, "is_pos_ctrl_mode"):
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene.entities["robot"]; data = robot.data
    body_ids = asset_cfg.body_ids
    if not isinstance(body_ids, (list, tuple)) or len(body_ids) < 2:
        return torch.zeros(env.num_envs, device=env.device)

    lq = data.body_link_quat_w[:, body_ids[0]]            # (B,4) world wrist quat L
    rq = data.body_link_quat_w[:, body_ids[1]]            # (B,4) world wrist quat R
    qcmd = cmd.wrist_quat_command                         # (B,8) = L wxyz + R wxyz (world)
    ori_err = _quat_geodesic_sq(lq, qcmd[:, 0:4]) + _quat_geodesic_sq(rq, qcmd[:, 4:8])  # (B,)

    mode_mask = (cmd.is_pos_ctrl_mode | cmd.is_force_ctrl_mode).float()
    walk_mask = cmd.is_walk_mode.float()
    scale = mode_mask + 0.2 * walk_mask
    return torch.exp(-ori_err / sigma) * scale




class WristTrackingBadTermination:
    """Terminate manip/force episodes when wrist pose tracking is bad for N steps.

    Thresholds are deliberately tail-based and sustained, so this cuts obvious
    tracking failures without killing one-frame transients. Position threshold is
    meters; orientation threshold is radians geodesic angle.
    """

    def __init__(self, cfg, env):
        self._bad_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._bad_count[env_ids] = 0

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        command_name: str,
        pos_threshold_m: float,
        ori_threshold_rad: float,
        grace_steps: int,
        warmup_steps: int,
        asset_cfg: "SceneEntityCfg",
    ) -> torch.Tensor:
        cmd = env.command_manager.get_term(command_name)
        if (not hasattr(cmd, "wrist_pose_command") or not hasattr(cmd, "wrist_quat_command")
                or not hasattr(cmd, "is_pos_ctrl_mode")):
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

        body_ids = asset_cfg.body_ids
        if not isinstance(body_ids, (list, tuple)) or len(body_ids) < 2:
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

        data = env.scene.entities["robot"].data
        lpos = data.body_link_pos_w[:, body_ids[0]]
        rpos = data.body_link_pos_w[:, body_ids[1]]
        lq = data.body_link_quat_w[:, body_ids[0]]
        rq = data.body_link_quat_w[:, body_ids[1]]
        tpos = cmd.wrist_pose_command
        tq = cmd.wrist_quat_command

        pos_err = torch.maximum(
            torch.linalg.norm(lpos - tpos[:, 0:3], dim=-1),
            torch.linalg.norm(rpos - tpos[:, 3:6], dim=-1),
        )

        ldot = torch.sum(lq * tq[:, 0:4], dim=-1).abs().clamp(max=1.0)
        rdot = torch.sum(rq * tq[:, 4:8], dim=-1).abs().clamp(max=1.0)
        ori_err = torch.maximum(2.0 * torch.acos(ldot), 2.0 * torch.acos(rdot))

        mode_mask = cmd.is_pos_ctrl_mode | cmd.is_force_ctrl_mode
        warm_mask = env.episode_length_buf >= int(warmup_steps)
        bad = mode_mask & warm_mask & ((pos_err > pos_threshold_m) | (ori_err > ori_threshold_rad))
        self._bad_count = torch.where(bad, self._bad_count + 1, torch.zeros_like(self._bad_count))
        return self._bad_count >= int(grace_steps)

def tracking_wrist_pos(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Exponential reward for tracking wrist position targets.

    reward = exp(-||pos - target||^2 / sigma) * mode_mask
    Active in position/force control modes only.
    """
    cmd = env.command_manager.get_term(command_name)
    if not hasattr(cmd, "wrist_pose_command") or not hasattr(cmd, "is_pos_ctrl_mode"):
        return torch.zeros(env.num_envs, device=env.device)

    # target (wrist_pose_command) is in WORLD-ABSOLUTE coords (commands.py composes it via
    # live torso world xy + commanded height). FORGE compares actual vs target both in the
    # world frame (ee_residual_rewards.py:209-214). So actual wrist must ALSO be world-
    # absolute — do NOT subtract root_pos (that was a migration bug: actual=root-relative vs
    # target=world differ by the env-grid offset, up to ~7m -> error huge -> reward always 0).
    target = cmd.wrist_pose_command  # (B, 6) world-absolute
    robot = env.scene.entities["robot"]
    data = robot.data

    # Get wrist body positions (world-absolute, matching target frame)
    body_ids = asset_cfg.body_ids
    if not isinstance(body_ids, (list, tuple)) or len(body_ids) < 2:
        return torch.zeros(env.num_envs, device=env.device)

    left_pos = data.body_link_pos_w[:, body_ids[0]]
    right_pos = data.body_link_pos_w[:, body_ids[1]]
    wrist_pos = torch.cat([left_pos, right_pos], dim=-1)  # (B, 6)

    error = torch.sum((wrist_pos - target) ** 2, dim=-1)
    mode_mask = (cmd.is_pos_ctrl_mode | cmd.is_force_ctrl_mode).float()

    # Walk mode gets reduced weight (0.2x) matching FORGE walk_wrist_scale
    walk_mask = cmd.is_walk_mode.float()
    scale = mode_mask + 0.2 * walk_mask

    return torch.exp(-error / sigma) * scale


def tracking_elbow_pos(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Elbow position tracking reward for natural arm posture.

    Penalizes elbows being in unnatural positions during manipulation.
    """
    cmd = env.command_manager.get_term(command_name)
    if not hasattr(cmd, "is_pos_ctrl_mode") or not hasattr(cmd, "elbow_pos_command"):
        return torch.zeros(env.num_envs, device=env.device)

    mode_mask = (cmd.is_pos_ctrl_mode | cmd.is_force_ctrl_mode).float()
    if not mode_mask.any():
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene.entities["robot"]
    data = robot.data

    body_ids = asset_cfg.body_ids
    if len(body_ids) < 2:
        return torch.zeros(env.num_envs, device=env.device)

    # FORGE (ee_residual_rewards.py:247-258): track the FK-computed elbow target in the
    # WORLD frame. Previous migration bug pulled the elbow toward the pelvis origin
    # (||elbow - root||) and ignored elbow_pos_command entirely. Use actual elbow
    # (world-absolute) vs elbow_pos_command (world-absolute, set by FK), both same frame.
    target = cmd.elbow_pos_command  # (B, 6) world-absolute L+R elbow xyz
    left_elbow = data.body_link_pos_w[:, body_ids[0]]
    right_elbow = data.body_link_pos_w[:, body_ids[1]]
    elbow_pos = torch.cat([left_elbow, right_elbow], dim=-1)  # (B, 6)

    elbow_error = torch.sum((elbow_pos - target) ** 2, dim=-1)
    return torch.exp(-elbow_error / sigma) * mode_mask


# 29-dof wrist joint indices (L/R wrist roll/pitch/yaw) and their position within the
# 15-dim upper action vector (_UPPER_ACT = [12] + range(15,29)).
_WRIST_DOF_29 = [19, 20, 21, 26, 27, 28]
_WRIST_IN_UPPER = [5, 6, 7, 12, 13, 14]


# 15-dim upper command layout: waist_yaw + bilateral shoulder/elbow/wrist joints.
_UPPER_DOF_29 = [12] + list(range(15, 29))
_UPPER_IN_UPPER = list(range(15))




def tracking_upper_link_pos(
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    asset_cfg: "SceneEntityCfg",
    link_weights=None,
) -> torch.Tensor:
    """BeyondMimic-style upper-body link position tracking.

    Targets are generated by FK from ``current_upper_actions`` but compared in
    maximum-coordinate space: torso/elbow/wrist link positions. Error is averaged
    over links so the raw reward scale is stable if the link set changes.
    """
    cmd = env.command_manager.get_term(command_name)
    if not hasattr(cmd, "upper_link_pos_command") or not hasattr(cmd, "is_pos_ctrl_mode"):
        return torch.zeros(env.num_envs, device=env.device)
    body_ids = asset_cfg.body_ids
    if not isinstance(body_ids, (list, tuple)) or len(body_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene.entities["robot"]
    actual = robot.data.body_link_pos_w[:, body_ids]
    target = cmd.upper_link_pos_command[:, : len(body_ids)]
    per_link_error = torch.sum((actual - target) ** 2, dim=-1)
    if link_weights is not None:
        w = torch.as_tensor(link_weights, device=env.device, dtype=per_link_error.dtype)
        if w.numel() != per_link_error.shape[1]:
            raise ValueError(f"link_weights length {w.numel()} != upper link dim {per_link_error.shape[1]}")
        w = w.clamp_min(0.0)
        error = torch.sum(per_link_error * w.unsqueeze(0), dim=-1) / w.sum().clamp_min(1.0e-6)
    else:
        error = per_link_error.mean(dim=-1)
    mode_mask = (cmd.is_pos_ctrl_mode | cmd.is_force_ctrl_mode).float()
    return torch.exp(-error / (std * std)) * mode_mask


def tracking_upper_link_ori(
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    asset_cfg: "SceneEntityCfg",
    link_weights=None,
) -> torch.Tensor:
    """BeyondMimic-style upper-body link orientation tracking.

    Uses the same sign-invariant quaternion proxy as wrist orientation, averaged
    over upper-body links before applying a BeyondMimic-style std scale.
    """
    cmd = env.command_manager.get_term(command_name)
    if not hasattr(cmd, "upper_link_quat_command") or not hasattr(cmd, "is_pos_ctrl_mode"):
        return torch.zeros(env.num_envs, device=env.device)
    body_ids = asset_cfg.body_ids
    if not isinstance(body_ids, (list, tuple)) or len(body_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene.entities["robot"]
    actual = robot.data.body_link_quat_w[:, body_ids]
    target = cmd.upper_link_quat_command[:, : len(body_ids)]
    dot = torch.sum(actual * target, dim=-1).abs().clamp(max=1.0)
    per_link_error = 1.0 - dot * dot
    if link_weights is not None:
        w = torch.as_tensor(link_weights, device=env.device, dtype=per_link_error.dtype)
        if w.numel() != per_link_error.shape[1]:
            raise ValueError(f"link_weights length {w.numel()} != upper link dim {per_link_error.shape[1]}")
        w = w.clamp_min(0.0)
        error = torch.sum(per_link_error * w.unsqueeze(0), dim=-1) / w.sum().clamp_min(1.0e-6)
    else:
        error = per_link_error.mean(dim=-1)
    mode_mask = (cmd.is_pos_ctrl_mode | cmd.is_force_ctrl_mode).float()
    return torch.exp(-error / (std * std)) * mode_mask

def tracking_wrist_force(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float,
) -> torch.Tensor:
    """Reward for tracking wrist force commands via virtual target.

    virtual_target = FK_target + F_total / kp
    reward = exp(-||wrist_pos - virtual_target||^2 / sigma)
    Active in force control mode only.
    """
    cmd = env.command_manager.get_term(command_name)
    if not hasattr(cmd, "is_force_ctrl_mode"):
        return torch.zeros(env.num_envs, device=env.device)

    force_mask = cmd.is_force_ctrl_mode.float()
    if not force_mask.any():
        return torch.zeros(env.num_envs, device=env.device)

    # Virtual target computed from force command + position target
    if not hasattr(cmd, "virtual_wrist_pos_target"):
        return torch.zeros(env.num_envs, device=env.device)

    virtual_target = cmd.virtual_wrist_pos_target  # (B, 6)

    # Get current wrist positions
    robot = env.scene.entities["robot"]
    data = robot.data
    root_pos = data.root_link_pos_w[:, :3]

    wrist_pos = torch.zeros(env.num_envs, 6, device=env.device)
    try:
        left_id = robot.find_bodies("left_wrist_yaw_link")
        right_id = robot.find_bodies("right_wrist_yaw_link")
        if left_id is not None:
            wrist_pos[:, :3] = data.body_link_pos_w[:, left_id[0]] - root_pos
        if right_id is not None:
            wrist_pos[:, 3:] = data.body_link_pos_w[:, right_id[0]] - root_pos
    except Exception:
        pass

    error = torch.sum((wrist_pos - virtual_target) ** 2, dim=-1)
    return torch.exp(-error / sigma) * force_mask


def wrist_force_penalty(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Penalty for large virtual target tracking error during force control.

    Activates when error exceeds threshold, encouraging compliant behavior.
    """
    cmd = env.command_manager.get_term(command_name)
    if not hasattr(cmd, "is_force_ctrl_mode") or not hasattr(cmd, "virtual_wrist_pos_target"):
        return torch.zeros(env.num_envs, device=env.device)

    force_mask = cmd.is_force_ctrl_mode.float()
    virtual_target = cmd.virtual_wrist_pos_target

    robot = env.scene.entities["robot"]
    data = robot.data
    root_pos = data.root_link_pos_w[:, :3]

    wrist_pos = torch.zeros(env.num_envs, 6, device=env.device)
    try:
        left_id = robot.find_bodies("left_wrist_yaw_link")
        right_id = robot.find_bodies("right_wrist_yaw_link")
        if left_id is not None:
            wrist_pos[:, :3] = data.body_link_pos_w[:, left_id[0]] - root_pos
        if right_id is not None:
            wrist_pos[:, 3:] = data.body_link_pos_w[:, right_id[0]] - root_pos
    except Exception:
        pass

    error = torch.sum((wrist_pos - virtual_target) ** 2, dim=-1)
    return error * force_mask


def force_smoothness(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reward for smooth force error evolution.

    Penalizes large changes in force error between timesteps.
    """
    if not hasattr(env, "_prev_force_error"):
        env._prev_force_error = torch.zeros(env.num_envs, device=env.device)

    cmd = env.command_manager.get_term(command_name)
    if not hasattr(cmd, "wrist_force_command") or not hasattr(env, "observer"):
        return torch.zeros(env.num_envs, device=env.device)

    # Current force error
    force_cmd = cmd.wrist_force_command  # (B, 6)
    force_est = env.observer.wrist_forces_smoothed if env.observer is not None else torch.zeros_like(force_cmd)
    error = torch.sum((force_est - force_cmd) ** 2, dim=-1)

    # Rate of change penalty
    delta = (error - env._prev_force_error).abs()
    env._prev_force_error = error.detach()

    return delta


def action_vanish(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize residual actions during walking mode.

    Encourages residual policy to output near-zero corrections when
    the base walking policy should dominate.
    """
    cmd = env.command_manager.get_term("twist")
    if not hasattr(cmd, "is_walk_mode"):
        return torch.zeros(env.num_envs, device=env.device)

    walk_mask = cmd.is_walk_mode.float()
    action = env.action_manager.action
    if action is None:
        return torch.zeros(env.num_envs, device=env.device)

    return torch.sum(action ** 2, dim=-1) * walk_mask


def stand_still(env: ManagerBasedRlEnv, command_name: str, sensor_name: str) -> torch.Tensor:
    """Penalize feet leaving ground when stationary (FORGE stand_still).

    error = number of feet NOT in contact.
    Active when: (velocity command ≈ 0 AND not manip) OR manip mode.
    """
    cmd = env.command_manager.get_term(command_name)
    if not hasattr(cmd, "is_walk_mode"):
        return torch.zeros(env.num_envs, device=env.device)

    # Number of feet with no/low contact force
    sensor = env.scene.sensors[sensor_name]
    # contact force magnitude per foot; <threshold means airborne
    forces = sensor.data.force  # (B, num_feet, 3) or similar
    if forces.dim() == 3:
        contact_z = forces[..., 2].abs()  # (B, num_feet)
    else:
        contact_z = forces.abs()
    no_contact = (contact_z < 1.0).sum(dim=-1).float()  # count airborne feet

    # V55: this FORGE-specific stillness term is manip-only. Walk still/yaw stays MJLab-native.
    is_manip = cmd.is_pos_ctrl_mode | cmd.is_force_ctrl_mode
    return no_contact * is_manip.float()


def tracking_base_height(env: ManagerBasedRlEnv, command_name: str,
                         sensor_name: str, ankle_sole_distance: float = 0.02,
                         manip_scale: float = 1.0) -> torch.Tensor:
    """Track commanded base height (FORGE tracking_base_height).

    base_height = max(root_z - foot_l_z, root_z - foot_r_z)
    reward = exp(-|base_height - height_cmd + ankle_sole_distance| * 10)

    v16: manip_scale multiplies the reward ONLY in manip (pos/force) mode, so the effective
    base-height weight is (cfg weight) in walk and (cfg weight * manip_scale) in manip. Used to
    give a strong squat drive in manip (align with FORGE tracking_base_height=2.0) without
    re-introducing the walk-mode 'stand to farm reward' shortcut (base keeps weight 0.5 in walk).
    """
    cmd = env.command_manager.get_term(command_name)
    if not hasattr(cmd, "height_command"):
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene.entities["robot"]
    data = robot.data
    root_z = data.root_link_pos_w[:, 2]

    # Get foot heights
    foot_z = torch.zeros(env.num_envs, 2, device=env.device)
    try:
        left_id = robot.find_bodies("left_ankle_roll_link")
        right_id = robot.find_bodies("right_ankle_roll_link")
        if left_id is not None:
            foot_z[:, 0] = data.body_link_pos_w[:, left_id[0], 2]
        if right_id is not None:
            foot_z[:, 1] = data.body_link_pos_w[:, right_id[0], 2]
    except Exception:
        pass

    base_height = torch.max(root_z - foot_z[:, 0], root_z - foot_z[:, 1])
    height_error = torch.abs(base_height - cmd.height_command + ankle_sole_distance)
    rew = torch.exp(-height_error * 10.0)
    # v16: amplify in manip mode only (strong squat drive), keep walk as-is.
    if manip_scale != 1.0 and hasattr(cmd, "is_pos_ctrl_mode"):
        is_manip = (cmd.is_pos_ctrl_mode | cmd.is_force_ctrl_mode).float()
        rew = rew * (1.0 + (manip_scale - 1.0) * is_manip)
    return rew


# ============================================================
# Batch 1: locomotion regularizers (strictly aligned with FORGE)
# Suppress residual divergence. mjlab data interface.
# ============================================================

# G1 joint index groups (29-DOF order)
_HIP_JOINT_IDX = [0, 1, 2, 6, 7, 8]      # left/right hip yaw,roll,pitch
_KNEE_JOINT_IDX = [3, 9]
_ANKLE_JOINT_IDX = [4, 5, 10, 11]
_LOWER_IDX = list(range(15))
class variable_posture_squat:
    """mjlab variable_posture + manip-squat leg unlock.

    Problem: in manip mode the velocity command is ~0, so mjlab's variable_posture picks
    std_standing (全身 0.05) — it locks the LEGS tightly to the default UPRIGHT pose. But
    manip mode also issues a squat (height_command < nominal), which physically REQUIRES the
    sagittal leg chain (hip_pitch + knee + ankle_pitch) to bend. The two objectives fight ->
    the policy throws the legs into extreme poses (the observed manip leg anomaly).

    Fix: for envs that are in manip mode AND commanded to squat, relax the std of
    hip_pitch / knee / ankle_pitch to a walking-level value (default 0.3) so the legs can
    bend naturally. Everything else (hip_roll/yaw, ankle_roll, waist, arms) stays at the
    tight std_standing — balance and lateral symmetry are NOT loosened. Sagittal (pitch)
    chain unlocks; roll/yaw stays locked.
    """

    def __init__(self, cfg, env):
        from mjlab.tasks.velocity.mdp.rewards import variable_posture as _VP
        self._vp = _VP(cfg, env)
        self.default_joint_pos = self._vp.default_joint_pos
        self.std_standing = self._vp.std_standing
        self.std_walking = self._vp.std_walking
        self.std_running = self._vp.std_running
        # Per-joint squat relaxation std, graded to match FORGE's joint-deviation strengths
        # (v16). FORGE constrains knee strongest (-0.75), ankle mid (-0.5), hip weak (-0.2),
        # waist not at all -> so when squatting we relax: waist_pitch/hip_pitch strongly,
        # ankle_pitch moderately, knee only weakly. Larger std = looser. roll/yaw/arms untouched.
        asset = env.scene[cfg.params["asset_cfg"].name]
        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
        # default graded values; overridable via cfg.params["squat_std_map"]
        smap = cfg.params.get("squat_std_map", {
            "waist_pitch": 0.5, "hip_pitch": 0.5, "ankle_pitch": 0.3, "knee": 0.2,
        })
        self._squat_cols = []      # column indices
        self._squat_vals = []      # matching relaxed std
        for i, n in enumerate(joint_names):
            for key, val in smap.items():
                if key in n:
                    self._squat_cols.append(i)
                    self._squat_vals.append(float(val))
                    break

        # v19: upper-body pose ANNEAL with curriculum. The upper body has an active task
        # (wrist/elbow tracking + random course) that pulls the arms OUT toward FK targets,
        # while pose pulls them BACK to default. As curriculum rises the targets get farther
        # -> the pull-out grows but the fixed pose std didn't -> escalating tug-of-war (likely
        # the V18 curriculum~0.5 blow-up). Anneal upper-joint std from tight (curriculum 0) to
        # loose (curriculum 1) so the pose pull-back relaxes in step with the task pull-out.
        # tight = mjlab walking values (sane "don't flail" constraint); loose = 0.4 (still some
        # posture fallback, but lets the arm reach far). roll/yaw of waist stay locked (balance).
        upper_anneal = {  # joint-key -> (tight, loose)
            "shoulder_pitch": (0.15, 0.4), "shoulder_roll": (0.15, 0.4),
            "shoulder_yaw": (0.1, 0.4), "elbow": (0.15, 0.4),
            "wrist": (0.3, 0.5), "waist_yaw": (0.2, 0.4),
        }
        self._anneal_cols = []
        self._anneal_tight = []
        self._anneal_loose = []
        self._upper_guard_cols = []
        for i, n in enumerate(joint_names):
            for key, (t, lo) in upper_anneal.items():
                if key in n:
                    self._anneal_cols.append(i)
                    self._anneal_tight.append(float(t))
                    self._anneal_loose.append(float(lo))
                    break
            if any(key in n for key in ("shoulder", "elbow", "wrist")):
                self._upper_guard_cols.append(i)

    def _mean_normalized_pose_error(self, env, std_standing, std_walking, std_running, asset_cfg,
                                    command_name, walking_threshold: float = 0.5,
                                    running_threshold: float = 1.5, squat_std: float = 0.3,
                                    exclude_squat_chain: bool = False,
                                    upper_only: bool = False):
        asset = env.scene[asset_cfg.name]
        cmd = env.command_manager.get_command(command_name)
        term = env.command_manager.get_term(command_name)
        squatting = None

        linear_speed = torch.norm(cmd[:, :2], dim=1)
        angular_speed = torch.abs(cmd[:, 2])
        total_speed = linear_speed + angular_speed
        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = ((total_speed >= walking_threshold) & (total_speed < running_threshold)).float()
        running_mask = (total_speed >= running_threshold).float()

        # (B, J) per-env per-joint std
        std = (self.std_standing * standing_mask.unsqueeze(1)
               + self.std_walking * walking_mask.unsqueeze(1)
               + self.std_running * running_mask.unsqueeze(1))

        # V55 hardened default: no pose/curriculum anneal; use final loose upper-body std from start.
        if self._anneal_cols:
            cols = torch.tensor(self._anneal_cols, device=std.device, dtype=torch.long)
            loose = torch.tensor(self._anneal_loose, device=std.device, dtype=std.dtype)
            std[:, cols] = loose   # broadcast over batch

        # manip + squat -> relax the sagittal chain per-joint (graded, FORGE-style)
        if hasattr(term, "is_pos_ctrl_mode") and self._squat_cols:
            is_manip = (term.is_pos_ctrl_mode | term.is_force_ctrl_mode)
            nominal = float(getattr(term.cfg, "nominal_height", 0.74))
            squatting = is_manip & (term.height_command < nominal - 1e-4)  # (B,)
            if squatting.any():
                cols = torch.tensor(self._squat_cols, device=std.device, dtype=torch.long)
                vals = torch.tensor(self._squat_vals, device=std.device, dtype=std.dtype)
                relax = std.clone()
                relax[:, cols] = vals
                std = torch.where(squatting.unsqueeze(1), relax, std)

        cur = asset.data.joint_pos[:, asset_cfg.joint_ids]
        des = self.default_joint_pos[:, asset_cfg.joint_ids]
        err2 = torch.square(cur - des)
        normalized_error = err2 / (std ** 2)
        if upper_only:
            weights = torch.zeros_like(normalized_error)
            cols = torch.tensor(self._upper_guard_cols, device=weights.device, dtype=torch.long)
            weights[:, cols] = 1.0
            denom = weights.sum(dim=1).clamp_min(1.0)
            return (normalized_error * weights).sum(dim=1) / denom
        if exclude_squat_chain and squatting is not None and squatting.any():
            weights = torch.ones_like(normalized_error)
            cols = torch.tensor(self._squat_cols, device=weights.device, dtype=torch.long)
            squat_weights = weights[squatting]
            squat_weights[:, cols] = 0.0
            weights[squatting] = squat_weights
            denom = weights.sum(dim=1).clamp_min(1.0)
            return (normalized_error * weights).sum(dim=1) / denom
        return torch.mean(normalized_error, dim=1)

    def __call__(self, env, std_standing, std_walking, std_running, asset_cfg,
                 command_name, walking_threshold: float = 0.5, running_threshold: float = 1.5,
                 squat_std: float = 0.3):
        normalized_error = self._mean_normalized_pose_error(
            env, std_standing, std_walking, std_running, asset_cfg,
            command_name, walking_threshold, running_threshold, squat_std
        )
        return torch.exp(-normalized_error)


class pose_deviation_guard(variable_posture_squat):
    """Positive pose-deviation penalty used with a negative reward weight."""

    def __call__(self, env, std_standing, std_walking, std_running, asset_cfg,
                 command_name, walking_threshold: float = 0.5, running_threshold: float = 1.5,
                 squat_std: float = 0.3, warn_threshold: float = 0.60,
                 danger_threshold: float = 0.80, danger_scale: float = 4.0,
                 exclude_squat_chain: bool = True, start_iteration: int = 0,
                 disable_during_squat: bool = False, upper_only: bool = False):
        squat_guard_mask = None
        if disable_during_squat:
            try:
                term = env.command_manager.get_term(command_name)
                if hasattr(term, "is_pos_ctrl_mode"):
                    nominal = float(getattr(term.cfg, "nominal_height", 0.74))
                    squat_guard_mask = (term.is_pos_ctrl_mode | term.is_force_ctrl_mode) & (term.height_command < nominal - 1e-4)
            except Exception:
                squat_guard_mask = None
        if start_iteration > 0:
            try:
                term = env.command_manager.get_term(command_name)
                current_iteration = int(getattr(term, "_iteration", 0))
            except Exception:
                current_iteration = 0
            if current_iteration < int(start_iteration):
                return torch.zeros(env.num_envs, device=env.device)
        normalized_error = self._mean_normalized_pose_error(
            env, std_standing, std_walking, std_running, asset_cfg,
            command_name, walking_threshold, running_threshold, squat_std,
            exclude_squat_chain=exclude_squat_chain,
            upper_only=upper_only,
        )
        warn_penalty = torch.relu(normalized_error - warn_threshold)
        danger_penalty = torch.square(torch.relu(normalized_error - danger_threshold))
        penalty = warn_penalty + float(danger_scale) * danger_penalty
        if (not upper_only) and squat_guard_mask is not None and squat_guard_mask.any():
            penalty = torch.where(squat_guard_mask, torch.zeros_like(penalty), penalty)
        return penalty


# ============================================================
# Leg symmetry / contact rewards (v17, migrated from FORGE + HOMIE)
# Treat the body's lateral axis as world-y (root yaw stays ~0 when standing/squatting,
# which is exactly when these matter). Full-time (no mode gate), like FORGE/HOMIE.
# ============================================================
_FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
_KNEES = ("left_knee_link", "right_knee_link")
def _qri_y(base_quat, vec):
    """y-component of quat_rotate_inverse(base_quat[wxyz], vec). Lateral axis in base frame."""
    w = base_quat[:, 0]
    xyz = base_quat[:, 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=1)
    out = vec - w.unsqueeze(1) * t + torch.cross(xyz, t, dim=1)
    return out[:, 1]


def _lateral_dist_base(env, body_names):
    robot = env.scene.entities["robot"]
    d = robot.data
    lid = robot.find_bodies(body_names[0])[0][0]
    rid = robot.find_bodies(body_names[1])[0][0]
    root_pos = d.root_link_pos_w[:, :3]
    bq = d.root_link_quat_w
    ly = _qri_y(bq, d.body_link_pos_w[:, lid] - root_pos)
    ry = _qri_y(bq, d.body_link_pos_w[:, rid] - root_pos)
    return torch.abs(ly - ry)


def _manip_mask(env, command_name):
    term = env.command_manager.get_term(command_name)
    if hasattr(term, "is_pos_ctrl_mode"):
        return (term.is_pos_ctrl_mode | term.is_force_ctrl_mode).float()
    return torch.zeros(env.num_envs, device=env.device)


def _manip_walk_still_tall_masks(env, command_name, height_gate: float = 0.735,
                                  walk_threshold: float = 0.1):
    term = env.command_manager.get_term(command_name)
    zeros = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(term, "is_pos_ctrl_mode"):
        return torch.zeros_like(zeros), zeros
    is_manip = term.is_pos_ctrl_mode.bool() | term.is_force_ctrl_mode.bool()
    # V55: FORGE-specific anti-splay/still rewards must not enter walk mode.
    # MJLab handles walk still / yaw-only with its native velocity, pose, and foot-shaping terms.
    return is_manip, zeros

def _manip_or_walk_still_tall_mask(env, command_name, height_gate: float = 0.735,
                                   walk_threshold: float = 0.1) -> torch.Tensor:
    is_manip, walk_still = _manip_walk_still_tall_masks(env, command_name, height_gate, walk_threshold)
    return (is_manip | walk_still).float()


def knee_distance_lateral_manip(env, command_name: str, most: float = 0.40,
                                 walk_still_most: float = 0.25,
                                 walk_still_scale: float = 3.0) -> torch.Tensor:
    lat = _lateral_dist_base(env, _KNEES)
    is_manip, walk_still = _manip_walk_still_tall_masks(env, command_name)
    manip_over = torch.clamp(lat - most, min=0.0)
    still_over = torch.clamp(lat - walk_still_most, min=0.0)
    return manip_over * is_manip.float() + walk_still_scale * still_over * walk_still.float()


def feet_distance_lateral_manip(env, command_name: str, most: float = 0.45,
                                 walk_still_most: float = 0.30,
                                 walk_still_scale: float = 3.0) -> torch.Tensor:
    lat = _lateral_dist_base(env, _FEET)
    is_manip, walk_still = _manip_walk_still_tall_masks(env, command_name)
    manip_over = torch.clamp(lat - most, min=0.0)
    still_over = torch.clamp(lat - walk_still_most, min=0.0)
    return manip_over * is_manip.float() + walk_still_scale * still_over * walk_still.float()


# V24: manip-gated hip roll/yaw deviation penalty (kill residual left-leg outward splay).
# mjlab 29-dof leg order: 1=L_hip_roll 2=L_hip_yaw 7=R_hip_roll 8=R_hip_yaw.
_HIP_ROLLYAW_IDX = [1, 2, 7, 8]


def hip_roll_yaw_deviation_manip(env, command_name: str, walk_still_scale: float = 3.0) -> torch.Tensor:
    robot = env.scene.entities["robot"]
    d = robot.data
    dev = (d.joint_pos - d.default_joint_pos)[:, _HIP_ROLLYAW_IDX]
    pen = torch.sum(torch.square(dev), dim=-1)   # >=0
    is_manip, walk_still = _manip_walk_still_tall_masks(env, command_name)
    return pen * (is_manip.float() + walk_still_scale * walk_still.float())

def imu_stand_still(
    env: "ManagerBasedRlEnv",
    command_name: str,
    asset_cfg: "SceneEntityCfg",
    manip_scale: float = 1.0,
) -> torch.Tensor:
    """FORGE-style IMU stand-still penalty for stationary walk and manip.

    Penalizes angular velocity only. It intentionally does not use IMU linear velocity,
    so commanded squat height changes are not directly penalized.
    """
    robot = env.scene.entities[asset_cfg.name]
    base_ang_vel = robot.data.root_link_ang_vel_b[:, :3]
    penalty = torch.mean(torch.square(base_ang_vel), dim=-1)

    body_ids = asset_cfg.body_ids
    if body_ids is not None:
        link_ang_vel = robot.data.body_link_ang_vel_w[:, body_ids, :]
        penalty = penalty + torch.mean(torch.square(link_ang_vel), dim=(-1, -2))

    cmd = env.command_manager.get_term(command_name)
    if not hasattr(cmd, "is_pos_ctrl_mode"):
        return penalty

    is_manip = cmd.is_pos_ctrl_mode | cmd.is_force_ctrl_mode
    scale = torch.where(is_manip, torch.full_like(penalty, manip_scale), torch.zeros_like(penalty))
    return penalty * scale



# ============================================================
# Yaw-only stepping incentives: teach in-place turning without touching linear walk.
# Active only when walk mode has angular command and near-zero linear command.
# ============================================================
def _yaw_turn_walk_mask(env, command_name: str, yaw_threshold: float = 0.15,
                        lin_threshold: float = 0.10) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    if not hasattr(cmd, "vel_command_b") or not hasattr(cmd, "is_walk_mode"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    lin = torch.linalg.norm(cmd.vel_command_b[:, :2], dim=-1)
    yaw = torch.abs(cmd.vel_command_b[:, 2])
    return cmd.is_walk_mode.bool() & (yaw > yaw_threshold) & (lin < lin_threshold)


def yaw_turn_no_fly(env: ManagerBasedRlEnv, command_name: str, sensor_name: str,
                    yaw_threshold: float = 0.15, lin_threshold: float = 0.10,
                    contact_threshold: float = 1.0) -> torch.Tensor:
    """Reward single-foot support only for in-place walk turning."""
    mask = _yaw_turn_walk_mask(env, command_name, yaw_threshold, lin_threshold)
    if not torch.any(mask):
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene.sensors[sensor_name]
    forces = sensor.data.force
    contact_z = forces[..., 2].abs() if forces.dim() == 3 else forces.abs()
    num_contact = (contact_z > contact_threshold).sum(dim=-1)
    return (num_contact == 1).float() * mask.float()


def yaw_turn_feet_air_time(env: ManagerBasedRlEnv, command_name: str, sensor_name: str,
                           threshold_min: float = 0.05, threshold_max: float = 0.25,
                           yaw_threshold: float = 0.15, lin_threshold: float = 0.10) -> torch.Tensor:
    """Reward modest foot air time only for in-place walk turning."""
    mask = _yaw_turn_walk_mask(env, command_name, yaw_threshold, lin_threshold)
    if not torch.any(mask):
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene.sensors[sensor_name]
    current_air_time = getattr(sensor.data, "current_air_time", None)
    if current_air_time is None:
        return torch.zeros(env.num_envs, device=env.device)
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    return in_range.float().sum(dim=-1) * mask.float()

# ============================================================
# V37: lower-body foot/ankle stance stability (OpenHomie-inspired, NO reference traj).
# Root cause of "heel-up + backward drift" during stationary manip: legs fail to absorb
# upper-body disturbance -> policy cheats with ankle dorsiflexion + torso lean. These reward
# LOWER-body absorption directly (vs penalizing upper-body angular momentum).
# ============================================================
_ANKLE_DEV_IDX = [4, 5, 10, 11]   # L/R ankle pitch+roll in 29-dof


def deviation_ankle_joint(env, command_name, height_gate=0.735):
    """Penalize ankle deviation only in tall manip/force-manip mode."""
    cmd = env.command_manager.get_term(command_name)
    if not hasattr(cmd, "height_command") or not hasattr(cmd, "is_pos_ctrl_mode"):
        return torch.zeros(env.num_envs, device=env.device)
    d = env.scene.entities["robot"].data
    dev = (d.joint_pos - d.default_joint_pos)[:, _ANKLE_DEV_IDX]
    pen = torch.sum(torch.square(dev), dim=-1)
    is_manip = cmd.is_pos_ctrl_mode.bool() | cmd.is_force_ctrl_mode.bool()
    tall = cmd.height_command >= height_gate
    return pen * (is_manip & tall).float()

def feet_ground_parallel(env, command_name, sensor_name="foot_height_scan", height_gate=0.735):
    """Penalize foot sole tilt only in tall manip/force-manip mode."""
    cmd = env.command_manager.get_term(command_name)
    if not hasattr(cmd, "height_command") or not hasattr(cmd, "is_pos_ctrl_mode"):
        return torch.zeros(env.num_envs, device=env.device)
    try:
        fq = env.scene.sensors[sensor_name].data.frame_quat_w   # (B,2,4) wxyz
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)
    w = fq[..., 0]; x = fq[..., 1]; y = fq[..., 2]; z = fq[..., 3]
    gx = -(2.0 * (x * z - w * y))
    gy = -(2.0 * (y * z + w * x))
    tilt_sq = (gx * gx + gy * gy).sum(dim=-1)
    is_manip = cmd.is_pos_ctrl_mode.bool() | cmd.is_force_ctrl_mode.bool()
    tall = cmd.height_command >= height_gate
    return tilt_sq * (is_manip & tall).float()
