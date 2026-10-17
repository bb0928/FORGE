"""BaseObsBuilder: constructs FORGE-aligned 131-dim obs + mask with history.

Self-managed history ring buffer (FORGE-style, not mjlab's history mechanism).
Produces actor obs_batch = [history(T*131) | mask(T*11)] and the critic
counterpart = [history(T*146) | mask(T*14)].

obs_scales (from FORGE base config + residual overrides):
    ang_vel=0.5, dof_pos=1.0, dof_vel=0.05, wrist_force=0.01, dof_residual=0.1
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import os
import torch

from obs_spec import (
    ACTOR_OBS_DIM, CRITIC_OBS_DIM, ACTOR_NUM_GROUPS, CRITIC_NUM_GROUPS,
    ACTOR_MASK_RULES, CRITIC_MASK_RULES, ACTOR_GROUP_INDEX, CRITIC_GROUP_INDEX,
    build_group_mask,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# obs scales (FORGE)
SCALE_ANG_VEL = 0.5
SCALE_DOF_POS = 1.0
SCALE_DOF_VEL = 0.05
SCALE_WRIST_FORCE = 0.01
SCALE_DOF_RESIDUAL = 0.1
SCALE_LIN_VEL = 2.0

LEFT_WRIST_BODY = "left_wrist_yaw_link"
RIGHT_WRIST_BODY = "right_wrist_yaw_link"


class BaseObsBuilder:
    """Builds FORGE-aligned residual observations with self-managed history."""

    def __init__(self, env: ManagerBasedRlEnv, history_length: int = 6,
                 critic_history_length: int = 1):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.T = history_length
        self.Tc = critic_history_length

        self.robot = env.scene.entities["robot"]
        self._default_joint_pos = self.robot.data.default_joint_pos.clone()

        # History ring buffers (filled on reset)
        self._actor_hist = torch.zeros(self.num_envs, self.T, ACTOR_OBS_DIM, device=self.device)
        self._actor_mask_hist = torch.zeros(self.num_envs, self.T, ACTOR_NUM_GROUPS, device=self.device)
        self._critic_hist = torch.zeros(self.num_envs, self.Tc, CRITIC_OBS_DIM, device=self.device)
        self._critic_mask_hist = torch.zeros(self.num_envs, self.Tc, CRITIC_NUM_GROUPS, device=self.device)

        # cache wrist body ids
        self._left_wrist_id = self._find_body(LEFT_WRIST_BODY)
        self._right_wrist_id = self._find_body(RIGHT_WRIST_BODY)
        self._enable_obs_noise = os.environ.get("ENABLE_DOMAIN_RAND", "0") == "1"

    def _find_body(self, name):
        try:
            ids = self.robot.find_bodies(name)
            return ids[0] if ids is not None and len(ids) > 0 else None
        except Exception:
            return None

    # ---- single-frame builders ----


    # ---- FORGE Base-Yaw frame helpers (added 2026-06-14 coord-frame fix) ----
    @staticmethod
    def _yaw_quat(q):
        """quat (B,4 wxyz) -> yaw-only quat (roll/pitch zeroed). FORGE base_yaw_quat."""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        h = yaw * 0.5
        cw = torch.cos(h); sz = torch.sin(h); zero = torch.zeros_like(cw)
        return torch.stack([cw, zero, zero, sz], dim=1)

    @staticmethod
    def _quat_conj(q):
        return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=1)

    @staticmethod
    def _quat_rotate_inverse(q, v):
        """Rotate vector v (B,3) by inverse of quat q (B,4 wxyz). Matches isaacgym quat_rotate_inverse."""
        w = q[:, 0]; xyz = q[:, 1:]
        t = 2.0 * torch.cross(xyz, v, dim=1)
        return v - w.unsqueeze(1) * t + torch.cross(xyz, t, dim=1)

    @staticmethod
    def _quat_mul(a, b):
        aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return torch.stack([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ], dim=1)

    def _wrist_targets_base_yaw(self, cmd, data):
        """Transform world-frame wrist pos/quat targets into FORGE Base-Yaw Frame.

        Pos:  quat_rotate_inverse(base_yaw_quat, wrist_world - root_pos)   (L,R) -> (B,6)
        Quat: error quat q_curr^-1 * q_target (w>=0)                       (L,R) -> (B,8)
        """
        root_pos = data.root_link_pos_w[:, :3]
        root_quat = data.root_link_quat_w                      # (B,4) wxyz
        base_yaw = self._yaw_quat(root_quat)

        wp_world = cmd.wrist_pose_command                      # (B,6) L_xyz,R_xyz world-absolute
        lw = self._quat_rotate_inverse(base_yaw, wp_world[:, 0:3] - root_pos)
        rw = self._quat_rotate_inverse(base_yaw, wp_world[:, 3:6] - root_pos)
        wrist_pose_by = torch.cat([lw, rw], dim=-1)            # (B,6) base-yaw

        # error quat: q_curr^-1 * q_target, per wrist, force w>=0
        if self._left_wrist_id is not None and self._right_wrist_id is not None:
            lcur = data.body_link_quat_w[:, self._left_wrist_id].reshape(self.num_envs, 4)
            rcur = data.body_link_quat_w[:, self._right_wrist_id].reshape(self.num_envs, 4)
            wq = cmd.wrist_quat_command                        # (B,8) L_wxyz,R_wxyz world target
            le = self._quat_mul(self._quat_conj(lcur), wq[:, 0:4])
            re_ = self._quat_mul(self._quat_conj(rcur), wq[:, 4:8])
            le = torch.where(le[:, 0:1] < 0, -le, le)          # w>=0 continuity (wxyz: w at idx0)
            re_ = torch.where(re_[:, 0:1] < 0, -re_, re_)
            wrist_quat_by = torch.cat([le, re_], dim=-1)       # (B,8)
        else:
            wrist_quat_by = cmd.wrist_quat_command
        return wrist_pose_by, wrist_quat_by

    def _add_uniform_noise(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        if not self._enable_obs_noise or magnitude <= 0.0:
            return x
        return x + (torch.rand_like(x) * 2.0 - 1.0) * magnitude

    def _build_single_actor(self, add_noise: bool = True):
        """Build one frame of 131-dim actor obs (FORGE order)."""
        cmd = self.env.command_manager.get_term("twist")
        data = self.robot.data

        lin_command = cmd.vel_command_b[:, :2]
        ang_command = cmd.vel_command_b[:, 2:3]
        height_command = cmd.height_command.unsqueeze(-1)
        # FORGE Base-Yaw Frame (coord-frame fix 2026-06-14): world target -> base-yaw.
        wrist_pose, wrist_quat = self._wrist_targets_base_yaw(cmd, data)
        wrist_force = cmd.wrist_force_command * SCALE_WRIST_FORCE

        imu_ang_vel = data.root_link_ang_vel_b[:, :3] * SCALE_ANG_VEL
        imu_grav = data.projected_gravity_b[:, :3]
        imu_info = torch.cat([imu_ang_vel, imu_grav], dim=-1)

        dof_pos = (data.joint_pos - self._default_joint_pos) * SCALE_DOF_POS
        dof_vel = data.joint_vel * SCALE_DOF_VEL

        if add_noise and self._enable_obs_noise:
            # Conservative proprioceptive observation noise in already-scaled obs units.
            imu_ang_vel = self._add_uniform_noise(imu_ang_vel, 0.10)
            imu_grav = self._add_uniform_noise(imu_grav, 0.03)
            dof_pos = self._add_uniform_noise(dof_pos, 0.01)
            dof_vel = self._add_uniform_noise(dof_vel, 0.05)

        action = self.env.action_manager.action
        if action is None:
            action = torch.zeros(self.num_envs, 29, device=self.device)

        if hasattr(self.env, "observer") and self.env.observer is not None:
            dof_residual = self.env.observer.dof_residual * SCALE_DOF_RESIDUAL
        else:
            dof_residual = torch.zeros(self.num_envs, 14, device=self.device)

        return torch.cat([
            lin_command, ang_command, height_command,
            wrist_pose, wrist_quat, wrist_force,
            imu_info, dof_pos, dof_vel, action, dof_residual,
        ], dim=-1)

    def _build_critic_extra(self):
        """Privileged critic-only extras:
        wrist_forces(6) + virtual_target(6) + base_lin_vel(3)
        + foot_contact_forces(6) + foot_air_time(2) + foot_contact(2)  [mjlab god-view].
        """
        cmd = self.env.command_manager.get_term("twist")
        data = self.robot.data

        if hasattr(self.env, "observer") and self.env.observer is not None:
            wrist_forces = self.env.observer.wrist_forces_smoothed * SCALE_WRIST_FORCE
        else:
            wrist_forces = torch.zeros(self.num_envs, 6, device=self.device)

        if hasattr(cmd, "virtual_wrist_pos_target"):
            virtual_target = cmd.virtual_wrist_pos_target
        else:
            virtual_target = torch.zeros(self.num_envs, 6, device=self.device)

        base_lin_vel = data.root_link_lin_vel_b[:, :3] * SCALE_LIN_VEL

        # mjlab privileged foot-contact signals (same as official velocity critic).
        foot_forces, foot_air, foot_contact = self._foot_privileged()

        return torch.cat([
            wrist_forces, virtual_target, base_lin_vel,
            foot_forces, foot_air, foot_contact,
        ], dim=-1)

    def _foot_privileged(self):
        """Read feet_ground_contact sensor -> (forces[B,6], air_time[B,2], contact[B,2]).

        Robust to envs where the sensor is absent or has a different foot count: pads/zeros
        to the fixed (6,2,2) layout so the critic obs dim stays constant.
        """
        B = self.num_envs
        zeros6 = torch.zeros(B, 6, device=self.device)
        zeros2 = torch.zeros(B, 2, device=self.device)
        try:
            sensor = self.env.scene.sensors["feet_ground_contact"]
            sd = sensor.data
        except Exception:
            return zeros6, zeros2, zeros2

        # forces: (B, N, 3) -> log1p-compressed flat (B, 6); take first 2 feet.
        forces = zeros6
        try:
            f = sd.force  # (B, N, 3) or (B, 3)
            if f.dim() == 2:
                f = f.unsqueeze(1)
            f2 = f[:, :2].reshape(B, -1)             # up to 6
            f2 = torch.sign(f2) * torch.log1p(f2.abs())
            forces = torch.zeros(B, 6, device=self.device)
            forces[:, : f2.shape[1]] = f2[:, :6]
        except Exception:
            pass

        # air_time: (B, N) -> (B, 2)
        air = zeros2
        try:
            a = sd.current_air_time
            if a is not None:
                a = a.reshape(B, -1)
                air = torch.zeros(B, 2, device=self.device)
                air[:, : min(2, a.shape[1])] = a[:, :2]
        except Exception:
            pass

        # contact flag: (B, N) -> (B, 2)
        contact = zeros2
        try:
            found = sd.found
            if found is not None:
                c = (found > 0).float().reshape(B, -1)
                contact = torch.zeros(B, 2, device=self.device)
                contact[:, : min(2, c.shape[1])] = c[:, :2]
        except Exception:
            pass

        return forces, air, contact

    def _mode_flags(self):
        cmd = self.env.command_manager.get_term("twist")
        is_walk = cmd.is_walk_mode
        is_force = cmd.is_force_ctrl_mode
        is_op = cmd.is_pos_ctrl_mode | cmd.is_force_ctrl_mode
        return is_walk, is_op, is_force

    # ---- public API ----

    def reset_idx(self, env_ids: torch.Tensor):
        """Fill history buffers with current frame for reset envs."""
        if len(env_ids) == 0:
            return
        actor_frame = self._build_single_actor(add_noise=True)
        critic_actor_frame = self._build_single_actor(add_noise=False)
        critic_extra = self._build_critic_extra()
        critic_frame = torch.cat([critic_actor_frame, critic_extra], dim=-1)

        is_walk, is_op, is_force = self._mode_flags()
        amask = build_group_mask(ACTOR_MASK_RULES, ACTOR_GROUP_INDEX, is_walk, is_op, is_force).float()
        cmask = build_group_mask(CRITIC_MASK_RULES, CRITIC_GROUP_INDEX, is_walk, is_op, is_force).float()

        # broadcast current frame across all history slots
        self._actor_hist[env_ids] = actor_frame[env_ids].unsqueeze(1).expand(-1, self.T, -1)
        self._actor_mask_hist[env_ids] = amask[env_ids].unsqueeze(1).expand(-1, self.T, -1)
        self._critic_hist[env_ids] = critic_frame[env_ids].unsqueeze(1).expand(-1, self.Tc, -1)
        self._critic_mask_hist[env_ids] = cmask[env_ids].unsqueeze(1).expand(-1, self.Tc, -1)

    def update(self):
        """Push current frame, drop oldest (ring buffer)."""
        actor_frame = self._build_single_actor(add_noise=True)
        critic_actor_frame = self._build_single_actor(add_noise=False)
        critic_extra = self._build_critic_extra()
        critic_frame = torch.cat([critic_actor_frame, critic_extra], dim=-1)

        is_walk, is_op, is_force = self._mode_flags()
        amask = build_group_mask(ACTOR_MASK_RULES, ACTOR_GROUP_INDEX, is_walk, is_op, is_force).float()
        cmask = build_group_mask(CRITIC_MASK_RULES, CRITIC_GROUP_INDEX, is_walk, is_op, is_force).float()

        # roll: drop oldest (index 0), append newest at end
        self._actor_hist = torch.cat([self._actor_hist[:, 1:], actor_frame.unsqueeze(1)], dim=1)
        self._actor_mask_hist = torch.cat([self._actor_mask_hist[:, 1:], amask.unsqueeze(1)], dim=1)
        self._critic_hist = torch.cat([self._critic_hist[:, 1:], critic_frame.unsqueeze(1)], dim=1)
        self._critic_mask_hist = torch.cat([self._critic_mask_hist[:, 1:], cmask.unsqueeze(1)], dim=1)

    def get_actor_obs(self):
        """Return [history(T*131) | mask(T*11)] flattened. Shape (B, T*131 + T*11)."""
        hist = self._actor_hist.reshape(self.num_envs, -1)
        mask = self._actor_mask_hist.reshape(self.num_envs, -1)
        return torch.cat([hist, mask], dim=-1)

    def get_critic_obs(self):
        """Return [history(Tc*146) | mask(Tc*14)] flattened."""
        hist = self._critic_hist.reshape(self.num_envs, -1)
        mask = self._critic_mask_hist.reshape(self.num_envs, -1)
        return torch.cat([hist, mask], dim=-1)
