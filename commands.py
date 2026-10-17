"""Multi-modal command term for FORGE residual environment.

Three modes per env (resampled): walking / position-control / force-control.

FORGE upper-body mechanism (ported from ee_residual_wrist_control.py +
legged_robot_ee_residual.py step), the project's CORE design:
  - `current_upper_actions` (N, n_upper): the upper-body joint TARGET (in action units),
    which evolves by smoothly interpolating toward a `random_upper_actions` sample that is
    regenerated every `upper_interval` steps. The random amplitude is gated by
    `action_curriculum_ratio` through an exponential mapping (small ratio -> arms barely
    move; ratio=1 -> up to ~40% of joint range). Per-joint cap: wrist joints up to
    max_wrist_ratio, other upper joints up to max_upper_ratio.
  - Every `wrist_target_interval` steps, the wrist/elbow targets are computed by FORWARD
    KINEMATICS from `current_upper_actions` (warp-native FK, torso-relative -> world via
    upright torso yaw + commanded base height). Targets are therefore SELF-CONSISTENT and
    reachable (= where the arm would be if it moved to current_upper_actions).
  base mode: upper body is DRIVEN by current_upper_actions (runner does this).
  residual mode: upper body driven by the residual policy; current_upper_actions still
    evolves only to GENERATE the wrist target the policy must reach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# 29-DOF action layout split (matches train cfg LOWER/UPPER).
# UPPER = waist_yaw(12) + arms(15..28); the 15 upper action dims.
_UPPER_ACT = [12] + list(range(15, 29))
# Within UPPER, which are wrist joints (get the larger curriculum cap):
# action indices 19,20,21 (L wrist r/p/y), 26,27,28 (R wrist r/p/y).
_WRIST_ACT = {19, 20, 21, 26, 27, 28}


class MultiModalCommand(UniformVelocityCommand):
    cfg: "MultiModalCommandCfg"

    def __init__(self, cfg: "MultiModalCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        n = self.num_envs
        d = self.device
        self._env = env

        # Mode flags
        self.is_walk_mode = torch.ones(n, dtype=torch.bool, device=d)
        self.is_pos_ctrl_mode = torch.zeros(n, dtype=torch.bool, device=d)
        self.is_force_ctrl_mode = torch.zeros(n, dtype=torch.bool, device=d)

        # Wrist/elbow command targets (filled by FK each wrist_target_interval).
        self.wrist_pose_command = torch.zeros(n, 6, device=d)       # L+R wrist xyz (world)
        self.wrist_quat_command = torch.zeros(n, 8, device=d)       # L+R wrist quat
        self.wrist_quat_command[:, 0] = 1.0
        self.wrist_quat_command[:, 4] = 1.0
        self.wrist_force_command = torch.zeros(n, 6, device=d)
        self.elbow_pos_command = torch.zeros(n, 6, device=d)        # L+R elbow xyz (world)
        # BeyondMimic-style upper-body arm-chain link pose targets, world frame, ordered as:
        # left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist_yaw, right_wrist_yaw.
        self.upper_link_pos_command = torch.zeros(n, 6, 3, device=d)
        self.upper_link_quat_command = torch.zeros(n, 6, 4, device=d)
        self.upper_link_quat_command[:, :, 0] = 1.0
        self.virtual_wrist_pos_target = torch.zeros(n, 6, device=d)

        # Base height command.
        self.height_command = torch.full((n,), cfg.nominal_height, device=d)

        self._iteration = 0
        self._curriculum_ratio = 1.0   # pushed by runner; gates random upper amplitude + squat depth
        self._step_counter = 0

        # --- upper-body random course state ---
        self.n_upper = len(_UPPER_ACT)
        self.upper_act_idx = torch.tensor(_UPPER_ACT, dtype=torch.long, device=d)
        self.current_upper_actions = torch.zeros(n, self.n_upper, device=d)
        self.random_upper_actions = torch.zeros(n, self.n_upper, device=d)
        self.delta_upper_actions = torch.zeros(n, self.n_upper, device=d)
        # per-joint curriculum cap
        cap = torch.full((self.n_upper,), cfg.max_upper_ratio, device=d)
        for i, a in enumerate(_UPPER_ACT):
            if a in _WRIST_ACT:
                cap[i] = cfg.max_wrist_ratio
        self.upper_cap = cap
        # action_min/max in ACTION units for upper joints (range / action_scale).
        # We use a symmetric proxy: joint soft limits mapped to action units.
        robot = env.scene.entities["robot"]
        # default joint pos (29,)
        self._default_dof = robot.data.default_joint_pos[0].clone().to(d)
        # action_scale per-joint (G1_ACTION_SCALE)
        from mjlab.asset_zoo.robots import G1_ACTION_SCALE
        import re as _re
        joint_order = [
            "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint",
            "left_ankle_pitch_joint","left_ankle_roll_joint","right_hip_pitch_joint","right_hip_roll_joint",
            "right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
            "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
            "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint",
            "left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
            "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint",
            "right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
        ]
        scale_vec = []
        for nm in joint_order:
            for pat, v in G1_ACTION_SCALE.items():
                if _re.fullmatch(pat, nm) or _re.search(pat, nm):
                    scale_vec.append(v); break
        self._action_scale = torch.tensor(scale_vec, dtype=torch.float, device=d)  # (29,)
        # action_min/max for upper joints: |soft joint limit - default| / action_scale (FORGE-style).
        sjl = robot.data.soft_joint_pos_limits[0].to(d)  # (29,2)
        amin = (sjl[:, 0] - self._default_dof) / self._action_scale  # (29,)
        amax = (sjl[:, 1] - self._default_dof) / self._action_scale
        self._action_min_upper = amin[self.upper_act_idx]  # (n_upper,)
        self._action_max_upper = amax[self.upper_act_idx]

        # warp-native upper-body FK
        from wrist_fk import UpperBodyFK
        self._fk = UpperBodyFK(env, str(d))

    # ---------------- upper-body random course ----------------
    def _generate_random_upper_actions(self):
        """FORGE exponential curriculum -> random_upper_actions (in action units)."""
        n, d = self.num_envs, self.device
        capped = torch.clamp(
            torch.full((self.n_upper,), self._curriculum_ratio, device=d), max=self.upper_cap
        )  # (n_upper,)
        k = 20.0 * (1.0 - capped * 0.99)
        uu = torch.rand(n, self.n_upper, device=d)
        random_upper_ratio = -1.0 / k * torch.log(1 - uu + uu * torch.exp(-k))  # (n,n_upper)
        random_joint_ratio = random_upper_ratio * torch.rand(n, self.n_upper, device=d)
        rand_pos = torch.rand(n, self.n_upper, device=d) - 0.5
        amin = self._action_min_upper.unsqueeze(0)
        amax = self._action_max_upper.unsqueeze(0)
        self.random_upper_actions = (
            amin * (rand_pos >= 0).float() + amax * (rand_pos < 0).float()
        ) * random_joint_ratio

    def _step_upper_course(self):
        """Per-step: regenerate random target on interval, smooth-interpolate current."""
        if self._step_counter % self.cfg.upper_interval == 0:
            self._generate_random_upper_actions()
            self.delta_upper_actions = (
                self.random_upper_actions - self.current_upper_actions
            ) / float(self.cfg.upper_interval)
        self.current_upper_actions = self.current_upper_actions + self.delta_upper_actions

    @torch.no_grad()
    def _assign_wrist_targets_from_fk(self):
        """FK from current_upper_actions -> wrist/elbow world targets (FORGE)."""
        n = self.num_envs
        # full-29 action vector with only upper set to current_upper_actions, lower=0.
        act29 = torch.zeros(n, 29, device=self.device)
        act29[:, self.upper_act_idx] = self.current_upper_actions
        out = self._fk.compute_targets(act29, self._action_scale, self.height_command)
        # compose torso-relative -> world (upright torso yaw + commanded base height)
        self._compose_world_targets(out)

    def _compose_world_targets(self, fk):
        """fk has torso_pos/quat (canonical-FK frame) and wrist/elbow pos in that frame.
        FORGE: take wrist relative to torso, rotate by the LIVE torso's upright-yaw quat,
        translate to (live torso xy, commanded height)."""
        d = self.device
        robot = self._env.scene.entities["robot"]
        # V25: PELVIS-relative wrist/elbow (subtract canonical pelvis pos = FK origin).
        # Composing targets relative to pelvis (below the waist joints) removes the
        # waist-twist shortcut that caused systematic right-lean. FK chain still includes
        # waist (dofs=0 in V25); V26 unlocks waist sampling without touching this code.
        bp = fk["pelvis_pos"]                      # (N,3) canonical pelvis
        torso = fk["torso_pos"] - bp               # rel pelvis
        ls = fk["lshoulder_pos"] - bp              # rel pelvis
        rs = fk["rshoulder_pos"] - bp
        lw = fk["lwrist_pos"] - bp                 # rel pelvis
        rw = fk["rwrist_pos"] - bp
        le = fk["lelbow_pos"] - bp
        re_ = fk["relbow_pos"] - bp
        # live pelvis world pose
        pelvis_bid = self._fk.pelvis_bid
        import warp as wp
        live_xpos = wp.to_torch(self._env.sim._wp_data.xpos)   # (N,nbody,3) mj-id
        live_xquat = wp.to_torch(self._env.sim._wp_data.xquat)
        torso_pos_w = live_xpos[:self.num_envs, pelvis_bid]
        torso_quat_w = live_xquat[:self.num_envs, pelvis_bid]  # wxyz (pelvis)
        # upright yaw-only quat from PELVIS (not torso) -> waist twist no longer moves target
        yaw = _yaw_from_quat(torso_quat_w)
        up_quat = _quat_from_yaw(yaw)              # (N,4) wxyz
        # target base position: keep pelvis xy, use commanded height (+offset like FORGE 0.044)
        tgt_pos = torch.stack([torso_pos_w[:, 0], torso_pos_w[:, 1],
                               self.height_command + 0.044], dim=1)
        def to_world(rel):
            return _quat_rotate(up_quat, rel) + tgt_pos
        torso_w = to_world(torso)
        ls_w = to_world(ls)
        rs_w = to_world(rs)
        le_w = to_world(le)
        re_w = to_world(re_)
        lw_w = to_world(lw)
        rw_w = to_world(rw)
        self.wrist_pose_command = torch.cat([lw_w, rw_w], dim=1)
        self.virtual_wrist_pos_target = self.wrist_pose_command.clone()
        self.elbow_pos_command = torch.cat([le_w, re_w], dim=1)
        # link quat targets: rotate canonical FK quats by up_quat.
        torso_q = _quat_mul(up_quat, fk["torso_quat"])
        ls_q = _quat_mul(up_quat, fk["lshoulder_quat"])
        rs_q = _quat_mul(up_quat, fk["rshoulder_quat"])
        le_q = _quat_mul(up_quat, fk["lelbow_quat"])
        re_q = _quat_mul(up_quat, fk["relbow_quat"])
        lw_q = _quat_mul(up_quat, fk["lwrist_quat"])
        rw_q = _quat_mul(up_quat, fk["rwrist_quat"])
        self.wrist_quat_command = torch.cat([lw_q, rw_q], dim=1)
        self.upper_link_pos_command = torch.stack([ls_w, rs_w, le_w, re_w, lw_w, rw_w], dim=1)
        self.upper_link_quat_command = torch.stack([ls_q, rs_q, le_q, re_q, lw_q, rw_q], dim=1)

    # ---------------- mode / command resampling ----------------
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        n = len(env_ids)
        if n == 0:
            return
        d = self.device
        cfg = self.cfg
        probs = torch.tensor([cfg.walk_prob, cfg.pos_ctrl_prob, cfg.force_ctrl_prob], device=d)
        modes = torch.multinomial(probs.expand(n, -1), 1).squeeze(-1)
        walk = modes == 0; pos = modes == 1; force = modes == 2
        if self._iteration < cfg.force_start_iteration:
            pos = pos | force; force = torch.zeros_like(force)
        self.is_walk_mode[env_ids] = walk
        self.is_pos_ctrl_mode[env_ids] = pos
        self.is_force_ctrl_mode[env_ids] = force

        # Yaw-turn curriculum: among walk-mode commands, oversample pure in-place turns.
        # This creates stable exposure for stepping turns without changing linear-walk or manip commands.
        walk_ids = env_ids[walk]
        if len(walk_ids) > 0 and cfg.yaw_turn_walk_prob > 0.0:
            yaw_mask = torch.rand(len(walk_ids), device=d) < cfg.yaw_turn_walk_prob
            yaw_ids = walk_ids[yaw_mask]
            if len(yaw_ids) > 0:
                m = len(yaw_ids)
                signs = torch.ones(m, device=d)
                signs[torch.rand(m, device=d) < 0.5] = -1.0
                yaw_abs = torch.empty(m, device=d).uniform_(cfg.yaw_turn_abs_min, cfg.yaw_turn_abs_max)
                self.vel_command_b[yaw_ids, 0] = 0.0
                self.vel_command_b[yaw_ids, 1] = 0.0
                self.vel_command_b[yaw_ids, 2] = signs * yaw_abs

        # base height: walk=nominal; manip squat samples the FULL depth range from the start
        # (v21: decoupled from curriculum). Previously squat depth ramped with curriculum
        # (eff_lo = lo * ratio), so depth + wrist difficulty grew together and compounded into
        # the curriculum~0.5 blow-up. Now manip envs sample uniformly over the full [lo, hi]
        # (0.34 .. nominal) right away; only the UPPER-body target amplitude is curriculum-gated.
        lo, hi = cfg.ranges.height
        self.height_command[env_ids] = cfg.nominal_height
        manip = pos | force
        manip_ids = env_ids[manip]
        if len(manip_ids) > 0:
            self.height_command[manip_ids] = cfg.nominal_height + torch.empty(
                len(manip_ids), device=d).uniform_(lo, hi)

        # force command (force mode only, unless zero_wrist_command for base training)
        if not getattr(cfg, "zero_wrist_command", False):
            force_ids = env_ids[force]
            if len(force_ids) > 0:
                nf = len(force_ids)
                self.wrist_force_command[force_ids] = torch.empty(
                    nf, 6, device=d).uniform_(-cfg.ranges.wrist_force_max, cfg.ranges.wrist_force_max)
        # zero force for pos / walk
        self.wrist_force_command[env_ids[pos]] = 0.0
        self.wrist_force_command[env_ids[walk]] = 0.0

    def _update_command(self) -> None:
        super()._update_command()
        # manip envs stand (zero velocity command)
        manip_ids = (self.is_pos_ctrl_mode | self.is_force_ctrl_mode).nonzero(as_tuple=False).flatten()
        self.vel_command_b[manip_ids, :] = 0.0
        # advance upper-body random course + FK wrist targets every step
        self._step_upper_course()
        if self._step_counter % self.cfg.wrist_target_interval == 0:
            self._assign_wrist_targets_from_fk()
        self._step_counter += 1

    def set_iteration(self, iteration: int) -> None:
        self._iteration = iteration

    def set_curriculum_ratio(self, ratio: float) -> None:
        self._curriculum_ratio = float(max(0.0, min(1.0, ratio)))


# ---------------- quat helpers (wxyz) ----------------
def _yaw_from_quat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def _quat_from_yaw(yaw):
    h = yaw * 0.5
    w = torch.cos(h); z = torch.sin(h)
    zero = torch.zeros_like(w)
    return torch.stack([w, zero, zero, z], dim=1)  # wxyz

def _quat_rotate(q, v):
    w = q[:, 0]; xyz = q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=1)
    return v + w.unsqueeze(1) * t + torch.cross(xyz, t, dim=1)

def _quat_mul(a, b):
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=1)


@dataclass(kw_only=True)
class MultiModalCommandCfg(UniformVelocityCommandCfg):
    walk_prob: float = 0.5
    pos_ctrl_prob: float = 0.25
    force_ctrl_prob: float = 0.25
    yaw_turn_walk_prob: float = 0.20
    yaw_turn_abs_min: float = 0.20
    yaw_turn_abs_max: float = 0.50
    force_start_iteration: int = 3000
    nominal_height: float = 0.74
    zero_wrist_command: bool = False
    # FORGE upper-body course params
    upper_interval: int = 50            # V57DR reference: 1s upper random-target transition at policy dt=0.02
    wrist_target_interval: int = 5      # recompute FK wrist target every N steps
    max_upper_ratio: float = 0.8        # curriculum cap for non-wrist upper joints
    max_wrist_ratio: float = 1.0        # curriculum cap for wrist joints

    @dataclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        wrist_pos_radius: tuple[float, float] = (0.1, 0.3)   # legacy, unused now
        wrist_pos_theta: tuple[float, float] = (0.2 * math.pi, 0.9 * math.pi)
        wrist_force_max: float = 20.0
        height: tuple[float, float] = (-0.4, 0.0)

    ranges: Ranges

    def build(self, env: ManagerBasedRlEnv) -> MultiModalCommand:
        return MultiModalCommand(self, env)
