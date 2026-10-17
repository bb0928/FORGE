"""FORGE observation spec: group layout + per-mode mask truth table.

Single source of truth for the residual policy's observation structure.
Strictly aligned with FORGE ee_residual_observations.py MODE_OBS_MASK
(feet_pose and head_pose groups removed per project decision).

Actor obs layout (131 dims), in order:
    lin_command(2) ang_command(1) height_command(1)
    wrist_pose(6) wrist_quat(8) wrist_force(6)
    imu_info(6) dof_pos(29) dof_vel(29) action(29) dof_residual(14)

Critic adds privileged groups: wrist_forces(6) wrist_virtual_target(6) base_lin_vel(3)

Mask is per-GROUP (not per-dim): one bool per group per env, indicating
whether that group is valid in the current command mode. Invalid groups
are replaced by a learnable impute value in the obs encoder.
"""

from __future__ import annotations

# ---- Actor obs groups: (name, dim) in concatenation order ----
ACTOR_OBS_GROUPS = [
    ("lin_command", 2),
    ("ang_command", 1),
    ("height_command", 1),
    ("wrist_pose", 6),
    ("wrist_quat", 8),
    ("wrist_force", 6),
    ("imu_info", 6),
    ("dof_pos", 29),
    ("dof_vel", 29),
    ("action", 29),
    ("dof_residual", 14),
]

# ---- Critic = actor groups + privileged groups ----
# Privileged groups the CRITIC sees (actor never does). Beyond FORGE's manip privileged
# info we ALSO add mjlab's foot-contact privileged signals (foot_contact_forces / air_time
# / contact), matching what the official mjlab velocity critic uses — these are "god-view"
# signals that make value estimation much sharper. Base and residual share this exact spec
# (via the shared obs_builder), so the critic still loads cleanly for warm-start.
CRITIC_EXTRA_GROUPS = [
    ("wrist_forces", 6),
    ("wrist_virtual_target", 6),
    ("base_lin_vel", 3),
    ("foot_contact_forces", 6),   # mjlab privileged: 2 feet x 3 (log1p-compressed)
    ("foot_air_time", 2),         # mjlab privileged: per-foot current air time
    ("foot_contact", 2),          # mjlab privileged: per-foot contact flag
]

ACTOR_OBS_DIM = sum(d for _, d in ACTOR_OBS_GROUPS)            # 131
CRITIC_OBS_DIM = ACTOR_OBS_DIM + sum(d for _, d in CRITIC_EXTRA_GROUPS)  # 146

ACTOR_NUM_GROUPS = len(ACTOR_OBS_GROUPS)                       # 11
CRITIC_NUM_GROUPS = ACTOR_NUM_GROUPS + len(CRITIC_EXTRA_GROUPS)  # 14

# ---- Group name -> (start, end) slice into the flat obs vector ----
def _build_offsets(groups):
    offsets = {}
    cursor = 0
    for name, dim in groups:
        offsets[name] = (cursor, cursor + dim)
        cursor += dim
    return offsets

ACTOR_OBS_OFFSETS = _build_offsets(ACTOR_OBS_GROUPS)
CRITIC_OBS_OFFSETS = _build_offsets(ACTOR_OBS_GROUPS + CRITIC_EXTRA_GROUPS)

# ---- Group index (position in the per-group mask vector) ----
ACTOR_GROUP_INDEX = {name: i for i, (name, _) in enumerate(ACTOR_OBS_GROUPS)}
CRITIC_GROUP_INDEX = {
    name: i for i, (name, _) in enumerate(ACTOR_OBS_GROUPS + CRITIC_EXTRA_GROUPS)
}

# ---- Mode-based mask rule per group ----
# Values are one of: "all_true", "is_walking", "is_operation", "is_force_ctrl"
# (matches FORGE MODE_OBS_MASK exactly for the kept groups)
ACTOR_MASK_RULES = {
    "lin_command": "is_walking",
    "ang_command": "is_walking",
    "height_command": "is_operation",
    "wrist_pose": "all_true",
    "wrist_quat": "all_true",
    "wrist_force": "is_force_ctrl",
    "imu_info": "all_true",
    "dof_pos": "all_true",
    "dof_vel": "all_true",
    "action": "all_true",
    "dof_residual": "is_force_ctrl",
}

CRITIC_MASK_RULES = {
    **ACTOR_MASK_RULES,
    "wrist_forces": "is_force_ctrl",
    "wrist_virtual_target": "is_force_ctrl",
    "base_lin_vel": "all_true",
    "foot_contact_forces": "all_true",
    "foot_air_time": "all_true",
    "foot_contact": "all_true",
}


def build_group_mask(rules, group_index, is_walking, is_operation, is_force_ctrl):
    """Build a (num_envs, num_groups) bool mask from mode flags.

    Args:
        rules: dict group_name -> rule string
        group_index: dict group_name -> column index
        is_walking, is_operation, is_force_ctrl: (num_envs,) bool tensors

    Returns:
        (num_envs, num_groups) bool tensor
    """
    import torch
    n = is_walking.shape[0]
    device = is_walking.device
    num_groups = len(group_index)
    mask = torch.ones(n, num_groups, dtype=torch.bool, device=device)
    rule_map = {
        "all_true": torch.ones(n, dtype=torch.bool, device=device),
        "is_walking": is_walking,
        "is_operation": is_operation,
        "is_force_ctrl": is_force_ctrl,
    }
    for name, rule in rules.items():
        if name in group_index:
            mask[:, group_index[name]] = rule_map[rule]
    return mask


if __name__ == "__main__":
    print(f"ACTOR_OBS_DIM = {ACTOR_OBS_DIM} (expect 131)")
    print(f"CRITIC_OBS_DIM = {CRITIC_OBS_DIM} (expect 146)")
    print(f"ACTOR_NUM_GROUPS = {ACTOR_NUM_GROUPS} (expect 11)")
    print(f"CRITIC_NUM_GROUPS = {CRITIC_NUM_GROUPS} (expect 14)")
    print("\nActor offsets:")
    for name, (s, e) in ACTOR_OBS_OFFSETS.items():
        print(f"  {name}: [{s}:{e}] dim={e-s} mask={ACTOR_MASK_RULES[name]}")
