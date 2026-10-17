"""BaseActorCritic: full FORGE-aligned actor-critic.

Actor path:  obs_batch -> split(history,mask) -> ObsEncoderSepMLP -> Transformer
             -> [+adaptation_latent] -> actor MLP -> mean; clamped split std
Critic path: critic_batch -> split -> privileged encoder -> [transformer] -> critic MLP
Adaptation:  last-N masked frames -> enc -> dec -> predict (wrist_forces, wrist_pos, base_vel)

Action composition (residual): final[lower] += base_action  (done by caller/runner,
here we expose residual mean+std; base composition happens in the env wrapper).

This module is NOT yet wired to rsl-rl's MLPModel interface; it's a standalone
nn.Module with FORGE's act/evaluate/update_distribution API, to be adapted next.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rl.transformer_v2 import TransformerEncoder
from rl.obs_encoder import ObsEncoderSepMLP
from rl.normalizer import EmpiricalNormalization


def _mlp(in_dim, hidden_dims, out_dim, activation="elu"):
    act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    layers = [nn.Linear(in_dim, hidden_dims[0]), act()]
    for l in range(len(hidden_dims)):
        if l == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dims[l], out_dim))
        else:
            layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
            layers.append(act())
    return nn.Sequential(*layers)


class BaseActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        actor_group_dims,        # list of actor group dims (11 groups)
        critic_group_dims,       # list of critic group dims (14 groups)
        num_actions,
        lower_dof_indices,
        upper_dof_indices,
        actor_history_length=6,
        critic_history_length=1,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        # transformer
        transformer_latent_dim=128,
        transformer_num_heads=4,
        transformer_ff_size=256,
        transformer_num_layers=2,
        use_transformer_critic=False,
        # adaptation
        use_adaptation=True,
        adaptation_latent_dim=64,
        adaptation_encoder_dims=(512, 256, 128),
        adaptation_decoder_dims=(128, 64),
        adaptation_history_length=1,
        adaptation_output_dim=15,   # wrist_forces(6)+wrist_pos(6)+base_vel(3)
        # noise
        init_noise_std=0.1,
        min_noise_std=1e-3,
        max_noise_std=1.0,
        obs_normalization=True,
    ):
        super().__init__()
        Normal.set_default_validate_args(False)
        self.obs_normalization = obs_normalization

        self.num_actions = num_actions
        self.register_buffer("lower_dof_indices", torch.tensor(lower_dof_indices, dtype=torch.long))
        self.register_buffer("upper_dof_indices", torch.tensor(upper_dof_indices, dtype=torch.long))
        self.actor_history_length = actor_history_length
        self.critic_history_length = critic_history_length
        self.actor_group_count = len(actor_group_dims)
        self.critic_group_count = len(critic_group_dims)
        self.num_one_step_obs = sum(actor_group_dims)
        self.num_one_step_critic_obs = sum(critic_group_dims)
        self.use_adaptation = use_adaptation
        self.use_transformer_critic = use_transformer_critic
        self.adaptation_history_length = adaptation_history_length
        self.adaptation_output_dim = adaptation_output_dim
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        self.transformer_latent_dim = transformer_latent_dim

        # --- Actor: obs encoder + transformer ---
        self.obs_encoder = ObsEncoderSepMLP(
            latent_dim=transformer_latent_dim, group_dims=actor_group_dims,
            activation=activation, history_length=actor_history_length,
        )
        self.actor_transformer = TransformerEncoder(
            embed_dim=transformer_latent_dim, num_heads=transformer_num_heads,
            ff_dim=transformer_ff_size, num_layers=transformer_num_layers,
        )

        # --- Critic: privileged encoder + optional transformer ---
        self.privileged_obs_encoder = ObsEncoderSepMLP(
            latent_dim=transformer_latent_dim, group_dims=critic_group_dims,
            activation=activation, history_length=critic_history_length,
        )
        if use_transformer_critic:
            self.critic_transformer = TransformerEncoder(
                embed_dim=transformer_latent_dim, num_heads=transformer_num_heads,
                ff_dim=transformer_ff_size, num_layers=transformer_num_layers,
            )
        else:
            self.critic_transformer = None

        # --- Adaptation module ---
        if use_adaptation:
            adapt_in = self.num_one_step_obs * adaptation_history_length
            self.adaptation_encoder_module = _mlp(adapt_in, list(adaptation_encoder_dims), adaptation_latent_dim, activation)
            self.adaptation_decoder_module = _mlp(adaptation_latent_dim, list(adaptation_decoder_dims), adaptation_output_dim, activation)
            actor_in = transformer_latent_dim + adaptation_latent_dim
        else:
            self.adaptation_encoder_module = None
            self.adaptation_decoder_module = None
            actor_in = transformer_latent_dim

        # --- Actor & Critic heads ---
        self.actor = _mlp(actor_in, list(actor_hidden_dims), num_actions, activation)
        self.critic = _mlp(transformer_latent_dim, list(critic_hidden_dims), 1, activation)

        # init actor final layer near zero (FORGE stability)
        last = self.actor[-1]
        nn.init.normal_(last.weight, mean=0.0, std=0.01)
        nn.init.zeros_(last.bias)

        # --- Split std (lower/upper) ---
        self.std_lower = nn.Parameter(init_noise_std * 0.1 * torch.ones(len(lower_dof_indices)))
        self.std_upper = nn.Parameter(init_noise_std * torch.ones(len(upper_dof_indices)))

        # Obs normalizers (per-frame, applied before mask/encoder)
        if obs_normalization:
            self.actor_obs_norm = EmpiricalNormalization(self.num_one_step_obs)
            self.critic_obs_norm = EmpiricalNormalization(self.num_one_step_critic_obs)
        else:
            self.actor_obs_norm = nn.Identity()
            self.critic_obs_norm = nn.Identity()

        self.distribution = None
        self.last_adaptation_predictions = None

    # ---- std ----
    @property
    def clamped_std(self):
        cl = torch.clamp(self.std_lower, self.min_noise_std, self.max_noise_std * 0.2)
        cu = torch.clamp(self.std_upper, self.min_noise_std, self.max_noise_std)
        full = torch.zeros(self.num_actions, device=cl.device)
        full[self.lower_dof_indices] = cl
        full[self.upper_dof_indices] = cu
        return full

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    def set_norm_training(self, mode: bool):
        """Enable/disable normalizer stat updates (rollout=True, ppo update=False)."""
        if isinstance(self.actor_obs_norm, EmpiricalNormalization):
            self.actor_obs_norm.train(mode)
            self.critic_obs_norm.train(mode)

    # ---- obs splitting ----
    def _split_actor(self, obs_batch):
        mask_len = self.actor_history_length * self.actor_group_count
        mask = obs_batch[:, -mask_len:]
        hist = obs_batch[:, :-mask_len]
        B = obs_batch.shape[0]
        hist = hist.reshape(B, self.actor_history_length, self.num_one_step_obs)
        mask = mask.reshape(B, self.actor_history_length, self.actor_group_count)
        # normalize each frame (before mask/impute), matches rsl-rl whitening
        hist = self.actor_obs_norm(hist)
        return hist, mask

    def _split_critic(self, critic_batch):
        mask_len = self.critic_history_length * self.critic_group_count
        mask = critic_batch[:, -mask_len:]
        hist = critic_batch[:, :-mask_len]
        B = critic_batch.shape[0]
        hist = hist.reshape(B, self.critic_history_length, self.num_one_step_critic_obs)
        mask = mask.reshape(B, self.critic_history_length, self.critic_group_count)
        hist = self.critic_obs_norm(hist)
        return hist, mask

    def _apply_mask(self, frame_obs, frame_mask, group_dims_tensor):
        masked = frame_obs.clone()
        start = 0
        for idx in range(group_dims_tensor.shape[0]):
            dim = int(group_dims_tensor[idx])
            mv = frame_mask[:, idx].unsqueeze(-1)
            masked[:, start:start + dim] = masked[:, start:start + dim] * mv
            start += dim
        return masked

    def _run_adaptation(self, hist, mask):
        if not self.use_adaptation:
            empty = torch.zeros(hist.shape[0], 0, device=hist.device)
            return empty, empty
        B = hist.shape[0]
        n = min(self.adaptation_history_length, self.actor_history_length)
        obs_n = hist[:, -n:, :]      # (B, n, D)
        mask_n = mask[:, -n:, :]     # (B, n, G)
        gdt = self.obs_encoder.group_dims_tensor
        masked_list = []
        for t in range(n):
            masked_list.append(self._apply_mask(obs_n[:, t, :], mask_n[:, t, :], gdt))
        masked = torch.cat(masked_list, dim=1)   # (B, n*D)
        latent = self.adaptation_encoder_module(masked)
        pred = self.adaptation_decoder_module(latent)
        return latent, pred

    # ---- actor ----
    def update_distribution(self, hist, mask):
        emb = self.obs_encoder(hist, mask)               # (B,1,latent)
        actor_token = self.actor_transformer(emb)        # (B,latent)
        if self.use_adaptation:
            adapt_latent, adapt_pred = self._run_adaptation(hist, mask)
            self.last_adaptation_predictions = adapt_pred
            actor_input = torch.cat([actor_token, adapt_latent], dim=-1)
        else:
            self.last_adaptation_predictions = None
            actor_input = actor_token
        mean = self.actor(actor_input)
        std = self.clamped_std
        self.distribution = Normal(mean, mean * 0.0 + std)

    def act(self, obs_batch):
        """Sample residual action. Returns (residual_action, adaptation_pred)."""
        hist, mask = self._split_actor(obs_batch)
        self.update_distribution(hist, mask)
        action = self.distribution.sample()
        return action, self.last_adaptation_predictions

    def act_inference(self, obs_batch):
        hist, mask = self._split_actor(obs_batch)
        emb = self.obs_encoder(hist, mask)
        actor_token = self.actor_transformer(emb)
        if self.use_adaptation:
            adapt_latent, adapt_pred = self._run_adaptation(hist, mask)
            actor_input = torch.cat([actor_token, adapt_latent], dim=-1)
        else:
            adapt_pred = None
            actor_input = actor_token
        return self.actor(actor_input), adapt_pred

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # ---- critic ----
    def evaluate(self, critic_batch):
        hist, mask = self._split_critic(critic_batch)
        B = hist.shape[0]
        hist_flat = hist.reshape(B * self.critic_history_length, -1)
        mask_flat = mask.reshape(B * self.critic_history_length, -1)
        # privileged encoder expects (B, T, ...) shape; encoder internally reshapes
        emb = self.privileged_obs_encoder(hist, mask)  # (B,1,latent)
        if self.critic_transformer is not None:
            token = self.critic_transformer(emb)        # (B,latent)
        else:
            token = emb.squeeze(1)                       # (B,latent)
        return self.critic(token)
