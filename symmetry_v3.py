"""G1 full-body symmetry flip — V3, STRICTLY aligned to FORGE original.

Reference (authoritative): FORGE rsl_rl_repo/rsl_rl/algorithms/ee_sym_residual_ppo.py
  - flip_vector_waist()  (per-dim obs flip, 138-dim FORGE layout)
  - flip_g1_actions()    (29-dim action flip)

This module maps FORGE's flip RULES onto the migration's 131-dim actor frame
(obs_spec ACTOR_OBS_GROUPS — feet_pose & head_pose REMOVED vs FORGE's 138).

Migration 131 layout:
  lin_command(2) ang_command(1) height_command(1) wrist_pose(6) wrist_quat(8)
  wrist_force(6) imu_info(6) dof_pos(29) dof_vel(29) action(29) dof_residual(14)

CORRECTIONS vs the old symmetry_v2:
  * wrist_quat mirror is FORGE's [w+, x+, y-, z-] + L/R swap  (v2 wrongly used [w,-x,y,-z]).
  * everything cross-checked dim-by-dim against FORGE in smoke_sym_v3.

Joint flip rule (FORGE flip_g1_actions, mjlab order 0=hip_pitch..):
  leg(L<->R): [pitch+, roll-, yaw-, knee+, ank_pitch+, ank_roll-]
  waist(in place): [roll-, pitch+, yaw-]   (order: waist_yaw(12), waist_roll(13), waist_pitch(14))
  arm(L<->R): [sh_pitch+, sh_roll-, sh_yaw-, elbow+, wr_roll-, wr_pitch+, wr_yaw-]
"""
import torch

# mjlab 29-dof order: 0-5 Lleg,6-11 Rleg,12 waist_yaw,13 waist_roll,14 waist_pitch,
#                     15-21 Larm,22-28 Rarm
JOINT_SWAP_INDICES = [
    6, 7, 8, 9, 10, 11,  0, 1, 2, 3, 4, 5,   # legs L<->R
    12, 13, 14,                               # waist in place
    22, 23, 24, 25, 26, 27, 28,  15, 16, 17, 18, 19, 20, 21,  # arms L<->R
]
# signs: leg[pitch+,roll-,yaw-,knee+,ankpitch+,ankroll-]; waist[yaw-,roll-,pitch+];
#        arm[shpitch+,shroll-,shyaw-,elbow+,wrroll-,wrpitch+,wryaw-]
JOINT_NEGATE_MASK = torch.tensor([
    1, -1, -1, 1, 1, -1,  1, -1, -1, 1, 1, -1,
    -1, -1, 1,
    1, -1, -1, 1, -1, 1, -1,   1, -1, -1, 1, -1, 1, -1,
], dtype=torch.float32)

# dof_residual: FORGE 14 = Larm7 + Rarm7 (NOT legs!). FORGE lines 124-137:
# [sh_pitch+, sh_roll-, sh_yaw-, elbow+, wr_roll-, wr_pitch+, wr_yaw-] per arm, L<->R swap.
RESIDUAL_SWAP_INDICES = [7, 8, 9, 10, 11, 12, 13,  0, 1, 2, 3, 4, 5, 6]
RESIDUAL_NEGATE_MASK = torch.tensor(
    [1, -1, -1, 1, -1, 1, -1,  1, -1, -1, 1, -1, 1, -1], dtype=torch.float32
)


def flip_joint_vector(v):
    return v[:, JOINT_SWAP_INDICES] * JOINT_NEGATE_MASK.to(v.device)


def flip_residual_vector(v):
    return v[:, RESIDUAL_SWAP_INDICES] * RESIDUAL_NEGATE_MASK.to(v.device)


def _flip_wrist_pos(p6):
    """(B,6)=Lxyz,Rxyz: swap L/R, negate y (FORGE wrist_pose 8-13)."""
    out = torch.empty_like(p6)
    out[:, 0] = p6[:, 3]; out[:, 1] = -p6[:, 4]; out[:, 2] = p6[:, 5]
    out[:, 3] = p6[:, 0]; out[:, 4] = -p6[:, 1]; out[:, 5] = p6[:, 2]
    return out


def _flip_wrist_force(f6):
    """(B,6): swap L/R, negate y (FORGE wrist_force 22-27). Same pattern as pos."""
    return _flip_wrist_pos(f6)


def _flip_wrist_quat(q8):
    """(B,8)=Lwxyz,Rwxyz: swap L/R, mirror quat [w+,x+,y-,z-] (FORGE 14-21)."""
    out = torch.empty_like(q8)
    # left <- right with y,z negated
    out[:, 0] = q8[:, 4]; out[:, 1] = q8[:, 5]; out[:, 2] = -q8[:, 6]; out[:, 3] = -q8[:, 7]
    # right <- left with y,z negated
    out[:, 4] = q8[:, 0]; out[:, 5] = q8[:, 1]; out[:, 6] = -q8[:, 2]; out[:, 7] = -q8[:, 3]
    return out


# 131-frame offsets
OFF = {
    "lin_command": (0, 2), "ang_command": (2, 3), "height_command": (3, 4),
    "wrist_pose": (4, 10), "wrist_quat": (10, 18), "wrist_force": (18, 24),
    "imu_info": (24, 30), "dof_pos": (30, 59), "dof_vel": (59, 88),
    "action": (88, 117), "dof_residual": (117, 131),
}
ACTOR_FRAME_DIM = 131


def flip_actor_frame(x):
    assert x.shape[-1] == ACTOR_FRAME_DIM, f"expected 131 got {x.shape[-1]}"
    o = x.clone()
    # lin_command (vx,vy): x+, y-
    o[:, 1] = -x[:, 1]
    # ang_command (yaw): -
    o[:, 2] = -x[:, 2]
    # height_command: unchanged (o already copy)
    s, e = OFF["wrist_pose"];  o[:, s:e] = _flip_wrist_pos(x[:, s:e])
    s, e = OFF["wrist_quat"];  o[:, s:e] = _flip_wrist_quat(x[:, s:e])
    s, e = OFF["wrist_force"]; o[:, s:e] = _flip_wrist_force(x[:, s:e])
    # imu_info: ang_vel(roll-,pitch+,yaw-) + grav(x+,y-,z+)
    s, e = OFF["imu_info"]
    o[:, s+0] = -x[:, s+0]; o[:, s+1] = x[:, s+1]; o[:, s+2] = -x[:, s+2]
    o[:, s+3] = x[:, s+3];  o[:, s+4] = -x[:, s+4]; o[:, s+5] = x[:, s+5]
    for g in ("dof_pos", "dof_vel", "action"):
        s, e = OFF[g]; o[:, s:e] = flip_joint_vector(x[:, s:e])
    s, e = OFF["dof_residual"]; o[:, s:e] = flip_residual_vector(x[:, s:e])
    return o


def verify_involution(batch=16, atol=1e-6):
    x = torch.randn(batch, ACTOR_FRAME_DIM)
    return torch.allclose(x, flip_actor_frame(flip_actor_frame(x)), atol=atol)
