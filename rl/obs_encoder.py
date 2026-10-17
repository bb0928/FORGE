"""ObsEncoderSepMLP, strictly ported from FORGE masked_actor_critic_v2_residual.py.

Per-frame independent MLP encoders + softmax attention pooling over time.
Invalid obs groups (mask=0) are replaced by a learnable per-frame impute value.

Input:
    obs: (B, T, D)         D = sum of one_step_obs_dims (131 for actor)
    obs_mask: (B, T, G)    G = num groups (11 for actor); per-group bool/float
Output:
    (B, 1, latent_dim)     time-pooled latent
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name):
    return {"elu": nn.ELU(), "relu": nn.ReLU(), "tanh": nn.Tanh(), "selu": nn.SELU()}[name]


class ObsEncoderSepMLP(nn.Module):
    def __init__(
        self,
        latent_dim,
        group_dims,          # list of per-group dims, e.g. [2,1,1,6,8,6,6,29,29,29,14]
        hidden_dims=(256, 128, 64),
        activation="elu",
        history_length=6,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.history_length = history_length
        total_dim = sum(group_dims)

        # per-group dim, used to expand group-mask -> per-dim mask
        self.register_buffer(
            "group_dims_tensor",
            torch.tensor(list(group_dims), dtype=torch.long),
        )

        # learnable impute, one row per history frame (FORGE: (T, D))
        self.impute = nn.Parameter(torch.zeros((history_length, total_dim), dtype=torch.float32))

        act = get_activation(activation)
        hidden_dims = list(hidden_dims)

        mlp_modules = nn.ModuleDict()
        for h in range(history_length):
            layers = [nn.Linear(total_dim, hidden_dims[0]), act]
            for l in range(len(hidden_dims)):
                if l == len(hidden_dims) - 1:
                    layers.append(nn.Linear(hidden_dims[l], latent_dim + 1))  # +1 for score
                else:
                    layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                    layers.append(act)
            mlp_modules[f"mlp_{h}"] = nn.Sequential(*layers)
        self.mlp_modules = mlp_modules

    def forward(self, obs, obs_mask):
        # obs: (B, T, D), obs_mask: (B, T, G)
        B = obs.shape[0]
        T = self.history_length

        obs_flat = obs.reshape(B * T, -1)               # (B*T, D)
        mask_flat = obs_mask.reshape(B * T, -1)          # (B*T, G)

        # expand group mask to per-dim mask
        expanded_mask = mask_flat.repeat_interleave(self.group_dims_tensor, dim=1)  # (B*T, D)

        impute_expanded = self.impute.unsqueeze(0).expand(B, -1, -1).reshape(B * T, -1)  # (B*T, D)
        obs_flat = obs_flat * expanded_mask + (1 - expanded_mask) * impute_expanded

        obs_seq = obs_flat.reshape(B, T, -1)
        encode = []
        for h in range(T):
            encode.append(self.mlp_modules[f"mlp_{h}"](obs_seq[:, h, :]))
        encode = torch.stack(encode, dim=1)              # (B, T, latent+1)

        score = encode[:, :, -1]                          # (B, T)
        score = F.softmax(score, dim=-1).unsqueeze(-1)    # (B, T, 1)
        encode = encode[:, :, :-1]                        # (B, T, latent)
        encode = (encode * score).sum(dim=1)              # (B, latent)
        return encode.unsqueeze(1)                         # (B, 1, latent)
