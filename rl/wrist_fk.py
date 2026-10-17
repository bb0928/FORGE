"""Warp-native forward kinematics for the upper body (FORGE wrist/elbow target generation).

Replaces FORGE's pytorch-kinematics. Given a TARGET upper-body joint configuration
(default + upper_actions * action_scale), compute where the wrists / elbows WOULD be,
using mujoco_warp.kinematics on a SEPARATE Data copy (so the live sim is never disturbed).

FORGE semantics (legged_robot_ee_residual.py + ee_residual_wrist_control.py):
  target_upper_dof = default_dof[upper] + upper_actions * action_scale
  (shoulder_pitch clamped to <= 0 to avoid arms swinging backward into the torso)
  FK -> wrist/elbow pose RELATIVE TO TORSO, then composed into world with an upright
  torso yaw quaternion + commanded base height. We expose the torso-relative pose; the
  caller does the world composition (it already has torso yaw + height command).

mjlab facts (verified 2026-06-12, G1 flat, nq=36):
  qpos addr: waist_yaw=19 roll=20 pitch=21; left arm 22..28; right arm 29..35
  body id (robot/ prefix): torso_link=17, left_shoulder_roll, left_elbow=21, left_wrist_yaw=24,
                           right_shoulder_roll, right_elbow=28, right_wrist_yaw=31
  mujoco_warp.kinematics(m, d): writes d.qpos -> updates d.xpos / d.xquat.
"""

from __future__ import annotations

import torch
import mujoco
import mujoco_warp as mjw
import warp as wp


# qpos addresses of the 15 upper-body joints, in the order matching UPPER dof indices
# (waist_yaw is treated as "upper" per the LOWER/UPPER split, plus the 14 arm joints).
# NOTE: order here is the JOINT order in qpos, mapped from action/UPPER layout by the caller.
_UPPER_QPOS_ADDR = {
    "waist_yaw": 19,
    "left_shoulder_pitch": 22, "left_shoulder_roll": 23, "left_shoulder_yaw": 24,
    "left_elbow": 25, "left_wrist_roll": 26, "left_wrist_pitch": 27, "left_wrist_yaw": 28,
    "right_shoulder_pitch": 29, "right_shoulder_roll": 30, "right_shoulder_yaw": 31,
    "right_elbow": 32, "right_wrist_roll": 33, "right_wrist_pitch": 34, "right_wrist_yaw": 35,
}


class UpperBodyFK:
    """Batched warp FK for wrist/elbow targets, on an isolated Data copy."""

    def __init__(self, env, device: str):
        self.env = env
        self.device = device
        sim = env.sim
        self._m = sim._wp_model          # mujoco_warp Model (shared, read-only here)
        self._mj = sim._mj_model
        self.num_envs = env.num_envs

        # Resolve body ids (with robot/ prefix).
        def bid(name):
            i = mujoco.mj_name2id(self._mj, mujoco.mjtObj.mjOBJ_BODY, name)
            assert i >= 0, f"body {name} not found"
            return i
        self.torso_bid = bid("robot/torso_link")
        self.pelvis_bid = bid("robot/pelvis")
        self.lshoulder_bid = bid("robot/left_shoulder_roll_link")
        self.rshoulder_bid = bid("robot/right_shoulder_roll_link")
        self.lwrist_bid = bid("robot/left_wrist_yaw_link")
        self.rwrist_bid = bid("robot/right_wrist_yaw_link")
        self.lelbow_bid = bid("robot/left_elbow_link")
        self.relbow_bid = bid("robot/right_elbow_link")

        # qpos addresses of all 29 actuated joints in ACTION order (0..28), built from mj model.
        # action index i -> qpos addr. Joint i+1 (skip floating base joint 0).
        self.act_qadr = torch.tensor(
            [int(self._mj.jnt_qposadr[j + 1]) for j in range(29)],
            dtype=torch.long, device=device,
        )
        # shoulder_pitch action indices (for clamp <= 0): left=15, right=22 in 29-DOF order.
        self.lshoulder_pitch_act = 15
        self.rshoulder_pitch_act = 22

        # Default joint pos (29,) from the entity.
        self.default_dof = env.scene.entities["robot"].data.default_joint_pos[0].clone().to(device)

        # An isolated Data copy for FK (so the live sim qpos is never touched).
        # IMPORTANT (multi-GPU): make_data has no device arg and otherwise lands on warp's
        # global default device (cuda:0). Under DDP, rank N runs on cuda:N, so the FK data
        # MUST be created on this env's device or qpos (cuda:0) vs target_dof (cuda:N) device
        # mismatch crashes. Scope all warp allocs/ops here and in compute_targets to self.device.
        # Mirror mjlab sim.py: resolve a warp device handle from the torch device string, then
        # scope all warp allocs to it. sim._wp_model is already on this device (sim.py creates
        # it under ScopedDevice(wp.get_device(self.device))), so kinematics(self._m, self._d)
        # will have both operands on the same device once self._d is allocated here.
        self._wp_device = wp.get_device(self.device)
        with wp.ScopedDevice(self._wp_device):
            self._d = mjw.make_data(self._mj, nworld=self.num_envs)   # MjModel + nworld
        self._nq = self._mj.nq

    @torch.no_grad()
    def compute_targets(self, upper_actions_29: torch.Tensor, action_scale, base_height_cmd: torch.Tensor):
        """Compute wrist/elbow targets in the WORLD frame (upright torso yaw + commanded height).

        Args:
            upper_actions_29: (N,29) action vector; only upper indices are used as targets.
                              (lower indices ignored — FK only needs upper joints set to target,
                               rest stay at default for a clean torso-relative arm pose.)
            action_scale: scalar or (29,) per-joint scale (mjlab G1_ACTION_SCALE).
            base_height_cmd: (N,) commanded base height (= commands height).

        Returns:
            dict with wrist_pos(N,6), wrist_quat(N,8), elbow_pos(N,6) in world frame.
        """
        N = upper_actions_29.shape[0]
        dev = self.device

        # 1. target dof pos = default + action*scale, with shoulder_pitch clamp<=0.
        target_dof = self.default_dof.unsqueeze(0) + upper_actions_29 * (
            action_scale if not torch.is_tensor(action_scale) else action_scale.unsqueeze(0)
        )
        target_dof = target_dof.clone()
        # clamp shoulder pitch <= 0 (default + action*scale), FORGE behavior
        target_dof[:, self.lshoulder_pitch_act] = torch.clamp(target_dof[:, self.lshoulder_pitch_act], max=0.0)
        target_dof[:, self.rshoulder_pitch_act] = torch.clamp(target_dof[:, self.rshoulder_pitch_act], max=0.0)

        # 2-3. build qpos, write to isolated data, run kinematics, read back — ALL warp ops
        # scoped to self.device so under DDP (rank N on cuda:N) nothing falls back to cuda:0.
        # self._d.qpos is on self.device (created under ScopedDevice in __init__); to_torch
        # therefore yields a self.device tensor, matching target_dof / act_qadr.
        with wp.ScopedDevice(self._wp_device):
            qpos = wp.to_torch(self._d.qpos).clone()        # (nworld, nq)
            if qpos.shape[0] != N:
                qpos = qpos[:1].expand(N, -1).clone()
            qpos[:, 0:3] = 0.0
            qpos[:, 3] = 1.0; qpos[:, 4:7] = 0.0            # identity base quat (w,x,y,z)
            qpos[:, self.act_qadr] = target_dof             # set all 29 joints to target

            self._d.qpos.assign(wp.from_torch(qpos.contiguous()))
            mjw.kinematics(self._m, self._d)

            xpos = wp.to_torch(self._d.xpos)                # (nworld, nbody, 3)
            xquat = wp.to_torch(self._d.xquat)              # (nworld, nbody, 4) wxyz

        # 4. read wrist/elbow/torso world pose in this canonical FK config (mj global body id)
        torso_pos = xpos[:N, self.torso_bid]
        torso_quat = xquat[:N, self.torso_bid]
        pelvis_pos = xpos[:N, self.pelvis_bid]
        pelvis_quat = xquat[:N, self.pelvis_bid]
        return {
            "torso_pos": torso_pos, "torso_quat": torso_quat,
            "pelvis_pos": pelvis_pos, "pelvis_quat": pelvis_quat,
            "lshoulder_pos": xpos[:N, self.lshoulder_bid], "lshoulder_quat": xquat[:N, self.lshoulder_bid],
            "rshoulder_pos": xpos[:N, self.rshoulder_bid], "rshoulder_quat": xquat[:N, self.rshoulder_bid],
            "lwrist_pos": xpos[:N, self.lwrist_bid], "lwrist_quat": xquat[:N, self.lwrist_bid],
            "rwrist_pos": xpos[:N, self.rwrist_bid], "rwrist_quat": xquat[:N, self.rwrist_bid],
            "lelbow_pos": xpos[:N, self.lelbow_bid], "lelbow_quat": xquat[:N, self.lelbow_bid],
            "relbow_pos": xpos[:N, self.relbow_bid], "relbow_quat": xquat[:N, self.relbow_bid],
        }
