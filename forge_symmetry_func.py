"""rsl-rl symmetry data_augmentation_func for the V21 FORGE base policy.

Contract (rsl_rl/extensions/symmetry.py):
    func(env, obs, actions) -> (obs_out, actions_out)
  * Always returns [original; mirrored] concatenated along batch dim (num_aug=2).
  * obs is a TensorDict with keys "actor" (B, 6*131+66) and "critic" (B, 156+17);
    when called for the mirror loss, actions may be None (obs-only) or obs None (actions-only).
  * We mirror via the verified full-body flip in symmetry_v2.

Layout (verified on env 2026-06-14):
  actor  = 6 frames * 131 (flattened, [:786]) + per-group mask 6*11=66 ([786:852])
  critic = 1 * 156 ([:156]) + mask 17 ([156:173])
  actions = (B, 29) full-body joint action.

Mirror policy:
  * each 131 actor frame -> symmetry_v2.flip_actor_frame
  * mask block left unchanged (group validity is mirror-invariant)
  * critic frame: mirror the 131 actor-subset groups (same offsets) + the privileged
    extras; foot/wrist privileged groups are swapped L/R, base_lin_vel y-negated.
  * actions -> symmetry_v2.flip_joint_vector
"""
import torch
import symmetry_v3 as S2

ACTOR_FRAME = 131
ACTOR_NFRAMES = 6
ACTOR_HIST = ACTOR_FRAME * ACTOR_NFRAMES   # 786
CRITIC_FRAME = 156

# critic extra groups after the 131 actor-shared block:
# wrist_forces(6) wrist_virtual_target(6) base_lin_vel(3)
# foot_contact_forces(6) foot_air_time(2) foot_contact(2)
C = {
    "wrist_forces": (131, 137), "wrist_virtual_target": (137, 143),
    "base_lin_vel": (143, 146), "foot_contact_forces": (146, 152),
    "foot_air_time": (152, 154), "foot_contact": (154, 156),
}


def _flip_actor_block(actor: torch.Tensor) -> torch.Tensor:
    """actor: (B, 852). Flip each of the 6 frames; keep the 66-mask tail unchanged."""
    B = actor.shape[0]
    out = actor.clone()
    hist = actor[:, :ACTOR_HIST].reshape(B, ACTOR_NFRAMES, ACTOR_FRAME)
    flipped = S2.flip_actor_frame(hist.reshape(B * ACTOR_NFRAMES, ACTOR_FRAME))
    out[:, :ACTOR_HIST] = flipped.reshape(B, ACTOR_NFRAMES, ACTOR_FRAME).reshape(B, ACTOR_HIST)
    # mask tail [786:852] unchanged
    return out


def _flip_lr6(v):
    """(B,6) L_xyz,R_xyz -> swap + y-negate (same as S2._flip_lr_xyz)."""
    return S2._flip_wrist_pos(v)


def _flip_critic_block(critic: torch.Tensor) -> torch.Tensor:
    """critic: (B, 173) = 156 frame + 17 mask. Flip the frame, keep mask."""
    out = critic.clone()
    frame = critic[:, :CRITIC_FRAME]
    # the first 131 dims share the actor frame layout
    f131 = S2.flip_actor_frame(frame[:, :ACTOR_FRAME])
    out[:, :ACTOR_FRAME] = f131
    # privileged extras
    s, e = C["wrist_forces"]; out[:, s:e] = _flip_lr6(frame[:, s:e])
    s, e = C["wrist_virtual_target"]; out[:, s:e] = _flip_lr6(frame[:, s:e])
    s, e = C["base_lin_vel"]; b = frame[:, s:e].clone(); b[:, 1] *= -1; out[:, s:e] = b
    # foot_contact_forces (6) = L_xyz,R_xyz -> swap + y-negate
    s, e = C["foot_contact_forces"]; out[:, s:e] = _flip_lr6(frame[:, s:e])
    # foot_air_time (2) = [L,R] -> swap
    s, e = C["foot_air_time"]; out[:, s:e] = frame[:, s:e].flip(-1)
    # foot_contact (2) = [L,R] -> swap
    s, e = C["foot_contact"]; out[:, s:e] = frame[:, s:e].flip(-1)
    # mask tail [156:173] unchanged
    return out


def forge_data_augmentation(env=None, obs=None, actions=None):
    """rsl-rl symmetry hook. Returns ([orig; mirror]) for whichever of obs/actions given."""
    obs_out = None
    if obs is not None:
        mirrored = obs.clone()
        if "actor" in obs.keys():
            mirrored["actor"] = _flip_actor_block(obs["actor"])
        if "critic" in obs.keys():
            mirrored["critic"] = _flip_critic_block(obs["critic"])
        obs_out = torch.cat([obs, mirrored], dim=0)

    act_out = None
    if actions is not None:
        act_out = torch.cat([actions, S2.flip_joint_vector(actions)], dim=0)

    return obs_out, act_out
