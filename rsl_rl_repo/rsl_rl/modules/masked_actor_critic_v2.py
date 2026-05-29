# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn import functional as F

# from rsl_rl.modules.masked_him_estimator_v2 import MaskedHIMEstimatorV2
from .transformer_v2 import TransformerEncoder
from abc import ABC, abstractmethod


class ObsEncoder(nn.Module, ABC):
    def __init__(self, latent_dim, one_step_obs_dims):
        super(ObsEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.one_step_obs_dims = one_step_obs_dims

    @abstractmethod
    def forward(self, obs, obs_mask):
        # only encode the one_step_obs part
        pass


class ObsEncoderMLP(ObsEncoder):
    def __init__(
        self,
        latent_dim,
        one_step_obs_dims,
        hidden_dims=[256, 128, 64],
        activation="elu",
        history_length=4,
    ):
        super(ObsEncoderMLP, self).__init__(latent_dim, one_step_obs_dims)
        self.register_buffer(
            "one_step_obs_dims_indexes",
            torch.tensor([d for d in one_step_obs_dims.values()], dtype=torch.long),
        )
        self.impute = nn.Parameter(torch.zeros(sum(one_step_obs_dims.values()), dtype=torch.float32))
        activation = get_activation(activation)
        self.history_length = history_length

        mlp_layers = []
        mlp_layers.append(nn.Linear(sum(one_step_obs_dims.values()), hidden_dims[0]))
        mlp_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                mlp_layers.append(nn.Linear(hidden_dims[l], latent_dim + 1))
            else:
                mlp_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                mlp_layers.append(activation)
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, obs, obs_mask):
        # only encode the one_step_obs part
        # obs: [B, T, D]
        # obs_mask: [B, T, D]

        B_size = obs.shape[0]
        obs = obs.reshape(B_size * self.history_length, -1)
        obs_mask = obs_mask.reshape(B_size * self.history_length, -1)

        expanded_obs_mask = obs_mask.repeat_interleave(self.one_step_obs_dims_indexes, dim=1)
        obs = obs * expanded_obs_mask + (1 - expanded_obs_mask) * self.impute.unsqueeze(0).expand(obs.shape[0], -1)
        obs = self.mlp(obs)  # [B * T, latent_dim + 1]
        obs = obs.reshape(B_size, self.history_length, -1)

        score = obs[:, :, -1]  # [B, T]
        score = F.softmax(score, dim=-1).unsqueeze(-1)  # [B, T, 1]
        obs = obs[:, :, :-1]  # [B, T, latent_dim]
        obs = (obs * score).sum(dim=1)  # [B, latent_dim]
        obs = obs.unsqueeze(1)  # [B, 1, latent_dim]

        return obs


class ObsEncoderSepMLP(ObsEncoder):
    def __init__(
        self,
        latent_dim,
        one_step_obs_dims,
        hidden_dims=[256, 128, 64],
        activation="elu",
        history_length=4,
    ):
        super(ObsEncoderSepMLP, self).__init__(latent_dim, one_step_obs_dims)
        self.register_buffer(
            "one_step_obs_dims_indexes",
            torch.tensor([d for d in one_step_obs_dims.values()], dtype=torch.long),
        )
        self.impute = nn.Parameter(torch.zeros((history_length, sum(one_step_obs_dims.values())), dtype=torch.float32))
        activation = get_activation(activation)
        self.history_length = history_length

        mlp_modules = nn.ModuleDict()
        for h in range(self.history_length):
            mlp_layers = []
            mlp_layers.append(nn.Linear(sum(one_step_obs_dims.values()), hidden_dims[0]))
            mlp_layers.append(activation)
            for l in range(len(hidden_dims)):
                if l == len(hidden_dims) - 1:
                    mlp_layers.append(nn.Linear(hidden_dims[l], latent_dim + 1))
                else:
                    mlp_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                    mlp_layers.append(activation)
            mlp_modules[f"mlp_{h}"] = nn.Sequential(*mlp_layers)
        self.mlp_modules = mlp_modules

    def forward(self, obs, obs_mask):
        # only encode the one_step_obs part
        # obs: [B, T, D]
        # obs_mask: [B, T, D]

        B_size = obs.shape[0]
        obs = obs.reshape(B_size * self.history_length, -1)
        obs_mask = obs_mask.reshape(B_size * self.history_length, -1)

        expanded_obs_mask = obs_mask.repeat_interleave(self.one_step_obs_dims_indexes, dim=1)
        obs = obs * expanded_obs_mask + (1 - expanded_obs_mask) * self.impute.unsqueeze(0).expand(
            B_size, -1, -1
        ).reshape(B_size * self.history_length, -1)

        obs = obs.reshape(B_size, self.history_length, -1)
        encode = []
        for h in range(self.history_length):
            encode.append(self.mlp_modules[f"mlp_{h}"](obs[:, h, :]))
        encode = torch.stack(encode, dim=1)  # [B, T, latent_dim + 1]

        score = encode[:, :, -1]  # [B, T]
        score = F.softmax(score, dim=-1).unsqueeze(-1)  # [B, T, 1]
        encode = encode[:, :, :-1]  # [B, T, latent_dim]
        encode = (encode * score).sum(dim=1)  # [B, latent_dim]
        encode = encode.unsqueeze(1)  # [B, 1, latent_dim]

        return encode


class ObsEncoderAllMLP(ObsEncoder):
    def __init__(
        self,
        latent_dim,
        one_step_obs_dims,
        hidden_dims=[256, 128, 64],
        activation="elu",
        history_length=4,
    ):
        super(ObsEncoderAllMLP, self).__init__(latent_dim, one_step_obs_dims)
        self.register_buffer(
            "one_step_obs_dims_indexes",
            torch.tensor([d for d in one_step_obs_dims.values()], dtype=torch.long),
        )
        self.impute = nn.Parameter(torch.zeros(sum(one_step_obs_dims.values()), dtype=torch.float32))
        activation = get_activation(activation)
        self.history_length = history_length

        mlp_layers = []
        mlp_layers.append(nn.Linear(sum(one_step_obs_dims.values()) * history_length, hidden_dims[0]))
        mlp_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                mlp_layers.append(nn.Linear(hidden_dims[l], latent_dim))
            else:
                mlp_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                mlp_layers.append(activation)
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, obs, obs_mask):
        # only encode the one_step_obs part
        # obs: [B, T, D]
        # obs_mask: [B, T, D]

        B_size = obs.shape[0]
        obs = obs.reshape(B_size * self.history_length, -1)
        obs_mask = obs_mask.reshape(B_size * self.history_length, -1)

        expanded_obs_mask = obs_mask.repeat_interleave(self.one_step_obs_dims_indexes, dim=1)
        obs = obs * expanded_obs_mask + (1 - expanded_obs_mask) * self.impute.unsqueeze(0).expand(obs.shape[0], -1)
        obs = obs.reshape(B_size, -1)  # [B, T * D]
        obs = self.mlp(obs)  # [B, latent_dim]
        obs = obs.unsqueeze(1)  # [B, 1, latent_dim]

        return obs


class MaskedActorCriticV2(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_one_step_obs,
        num_one_step_critic_obs,
        actor_history_length,
        critic_history_length,
        num_actions=19,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        fix_waist=True,
        freq_control=False,
        one_step_obs_dims=None,
        one_step_privileged_obs_dims=None,
        **kwargs,
    ):
        # if kwargs:
        #     print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(MaskedActorCriticV2, self).__init__()
        # import pdb; pdb.set_trace()
        activation = get_activation(activation)
        self.transformer_latent_dim = kwargs["transformer_latent_dim"]

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_one_step_obs = num_one_step_obs
        self.num_one_step_critic_obs = num_one_step_critic_obs
        self.actor_history_length = actor_history_length
        self.critic_history_length = critic_history_length
        self.actor_proprioceptive_obs_length = self.actor_history_length * self.num_one_step_obs
        self.critic_proprioceptive_obs_length = self.critic_history_length * self.num_one_step_critic_obs
        self.num_height_points = self.num_actor_obs - self.actor_proprioceptive_obs_length
        self.num_critic_height_points = self.num_critic_obs - self.critic_proprioceptive_obs_length
        self.actor_use_height = True if self.num_height_points > 0 else False
        self.num_actions = num_actions
        self.one_step_obs_dims = one_step_obs_dims
        self.one_step_privileged_obs_dims = one_step_privileged_obs_dims

        self.fix_waist = fix_waist
        self.freq_control = freq_control

        self.one_step_obs_dims_indexes = {}  # one_step_privileged_obs_dims
        index_sum = 0
        for k, v in self.one_step_obs_dims.items():
            indexes = np.arange(index_sum, index_sum + v)
            index_sum += v
            self.one_step_obs_dims_indexes[k] = [
                (t * self.num_one_step_obs + indexes).tolist() for t in range(self.actor_history_length)
            ]
            self.one_step_obs_dims_indexes[k] = [
                item for sublist in self.one_step_obs_dims_indexes[k] for item in sublist
            ]

        self.one_step_privileged_obs_dims_indexes = {}  # one_step_privileged_obs_dims
        index_sum = 0
        for k, v in self.one_step_privileged_obs_dims.items():
            indexes = np.arange(index_sum, index_sum + v)
            index_sum += v
            self.one_step_privileged_obs_dims_indexes[k] = [
                (t * self.num_one_step_critic_obs + indexes).tolist() for t in range(self.critic_history_length)
            ]
            self.one_step_privileged_obs_dims_indexes[k] = [
                item for sublist in self.one_step_privileged_obs_dims_indexes[k] for item in sublist
            ]

        self.dynamic_latent_dim = kwargs["transformer_latent_dim"]
        self.terrain_latent_dim = kwargs["transformer_latent_dim"]
        if self.actor_use_height:
            pass  # TODO: caculate terrain token
        else:
            mlp_input_dim_a = self.transformer_latent_dim
        mlp_input_dim_c = self.transformer_latent_dim

        if kwargs["obs_encoder"] == "one_mlp":
            self.obs_encoder = ObsEncoderMLP(
                latent_dim=self.transformer_latent_dim,
                one_step_obs_dims=self.one_step_obs_dims,
                history_length=self.actor_history_length,
            )
            self.privileged_obs_encoder = ObsEncoderMLP(
                latent_dim=self.transformer_latent_dim,
                one_step_obs_dims=self.one_step_privileged_obs_dims,
                history_length=self.critic_history_length,
            )
        elif kwargs["obs_encoder"] == "sep_mlp":
            self.obs_encoder = ObsEncoderSepMLP(
                latent_dim=self.transformer_latent_dim,
                one_step_obs_dims=self.one_step_obs_dims,
                history_length=self.actor_history_length,
            )
            self.privileged_obs_encoder = ObsEncoderSepMLP(
                latent_dim=self.transformer_latent_dim,
                one_step_obs_dims=self.one_step_privileged_obs_dims,
                history_length=self.critic_history_length,
            )
        elif kwargs["obs_encoder"] == "all_mlp":
            self.obs_encoder = ObsEncoderAllMLP(
                latent_dim=self.transformer_latent_dim,
                one_step_obs_dims=self.one_step_obs_dims,
                history_length=self.actor_history_length,
            )
            self.privileged_obs_encoder = ObsEncoderAllMLP(
                latent_dim=self.transformer_latent_dim,
                one_step_obs_dims=self.one_step_privileged_obs_dims,
                history_length=self.critic_history_length,
            )
        else:
            raise NotImplementedError("ObsEncoder with attention is not implemented yet")

        self.actor_transformer = TransformerEncoder(
            embed_dim=kwargs["transformer_latent_dim"],
            num_heads=kwargs["transformer_num_heads"],
            ff_dim=kwargs["transformer_ff_size"],
            num_layers=kwargs["transformer_num_layers"],
        )

        if kwargs["use_transformer_critic"]:
            self.critic_transformer = TransformerEncoder(
                embed_dim=self.transformer_latent_dim,
                num_heads=kwargs["transformer_num_heads"],
                ff_dim=kwargs["transformer_ff_size"],
                num_layers=kwargs["transformer_num_layers"],
            )
        else:
            self.critic_transformer = None

        # self.estimator = MaskedHIMEstimatorV2(
        #     num_one_step_obs=sum([d for k, d in one_step_privileged_obs_dims.items() if "command" not in k]),
        #     latent_dim=self.dynamic_latent_dim,
        #     **kwargs,
        # )
        # self.update_encoder_when_estimating = kwargs["update_encoder_when_estimating"]
        #import pdb; pdb.set_trace()
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        # import pdb; pdb.set_trace()
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        # print(f"Estimator: {self.estimator.encoder}")
        if self.actor_use_height:
            print(f"Terrain Encoder: {self.terrain_encoder}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs_history, obs_mask):

        # batch_size = obs_history.shape[0]
        # obs_history = obs_history.reshape(batch_size * self.actor_history_length, -1)
        # obs_mask = obs_mask.reshape(batch_size * self.actor_history_length, -1)

        obs_embedding = self.obs_encoder(obs_history, obs_mask)
        # obs_embedding = obs_embedding.reshape(batch_size, self.actor_history_length, -1)  # [B, T, latent_dim]

        # with torch.no_grad():
        #     vel, dynamic_latent = self.estimator(obs_embedding)
        # import pdb; pdb.set_trace()
        if self.actor_use_height:
            raise NotImplementedError("Terrain encoder is not implemented yet")
            terrain_latent = self.terrain_encoder(obs_history[:, -(self.num_height_points + self.num_one_step_obs) :])
            actor_input = torch.cat(
                (
                    obs_history[:, -(self.num_height_points + self.num_one_step_obs) : -self.num_height_points],
                    vel,
                    dynamic_latent,
                    terrain_latent,
                ),
                dim=-1,
            )
        else:
            actor_input = self.actor_transformer(obs_embedding)
            # actor_input = torch.cat((actor_input, vel), dim=-1)
        action_mean = self.actor(actor_input)
        self.distribution = Normal(action_mean, action_mean * 0.0 + self.std)

    def act(self, obs_history, **kwargs):
        # import pdb; pdb.set_trace()
        obs_mask = obs_history[:, -1 * (self.actor_history_length * len(self.one_step_obs_dims)) :]
        obs_history = obs_history[:, : -1 * (self.actor_history_length * len(self.one_step_obs_dims))]
        self.update_distribution(obs_history, obs_mask)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history, observations=None):
        obs_mask = obs_history[:, -1 * (self.actor_history_length * len(self.one_step_obs_dims)) :]
        obs_history = obs_history[:, : -1 * (self.actor_history_length * len(self.one_step_obs_dims))]

        # batch_size = obs_history.shape[0]
        # obs_history = obs_history.reshape(batch_size * self.actor_history_length, -1)
        # obs_mask = obs_mask.reshape(batch_size * self.actor_history_length, -1)

        obs_embedding = self.obs_encoder(obs_history, obs_mask)
        # obs_embedding = obs_embedding.reshape(batch_size, self.actor_history_length, -1)  # [B, T, latent_dim]

        # with torch.no_grad():
        #     vel, dynamic_latent = self.estimator(obs_embedding)
        if self.actor_use_height:
            raise NotImplementedError("Terrain encoder is not implemented yet")
            terrain_latent = self.terrain_encoder(obs_history[:, -(self.num_height_points + self.num_one_step_obs) :])
            actor_input = torch.cat(
                (
                    obs_history[:, -(self.num_height_points + self.num_one_step_obs) : -self.num_height_points],
                    vel,
                    dynamic_latent,
                    terrain_latent,
                ),
                dim=-1,
            )
        else:
            actor_input = self.actor_transformer(obs_embedding)
            # actor_input = torch.cat((actor_input, vel), dim=-1)
        action_mean = self.actor(actor_input)
        return action_mean

    def evaluate(self, critic_observations, **kwargs):
        critic_observations_mask = critic_observations[
            :, -1 * (self.critic_history_length * len(self.one_step_privileged_obs_dims)) :
        ]
        critic_observations = critic_observations[
            :, : -1 * (self.critic_history_length * len(self.one_step_privileged_obs_dims))
        ]

        batch_size = critic_observations.shape[0]
        critic_observations = critic_observations.reshape(batch_size * self.critic_history_length, -1)
        critic_observations_mask = critic_observations_mask.reshape(batch_size * self.critic_history_length, -1)

        obs_embedding = self.privileged_obs_encoder(critic_observations, critic_observations_mask)
        obs_embedding = obs_embedding.reshape(batch_size, self.critic_history_length, -1)  # [B, T, latent_dim]
        if self.critic_transformer is not None:
            value = self.critic_transformer(obs_embedding)
            value = self.critic(value)
        else:
            value = self.critic(obs_embedding.squeeze())
        return value

    # def update_estimator(self, obs_history, next_critic_obs, lr=None):
    #     obs_mask = obs_history[:, -1 * (self.actor_history_length * len(self.one_step_obs_dims)) :]
    #     obs_history = obs_history[:, : -1 * (self.actor_history_length * len(self.one_step_obs_dims))]
    #     next_critic_obs = next_critic_obs[
    #         :, : -1 * (self.critic_history_length * len(self.one_step_privileged_obs_dims))
    #     ]

    #     with torch.set_grad_enabled(self.update_encoder_when_estimating):
    #         batch_size = obs_history.shape[0]
    #         obs_history = obs_history.reshape(batch_size * self.actor_history_length, -1)
    #         obs_mask = obs_mask.reshape(batch_size * self.actor_history_length, -1)

    #         obs_embedding = self.obs_encoder(obs_history, obs_mask)
    #         obs_embedding = obs_embedding.reshape(batch_size, self.actor_history_length, -1)  # [B, T, latent_dim]

    #     return self.estimator.update(
    #         obs_embedding, next_critic_obs[:, -3:], next_critic_obs[:, -self.estimator.tar_input_dim :], lr
    #     )


def get_activation(act_name, return_type="nn"):
    if act_name == "elu":
        return nn.ELU() if return_type == "nn" else F.elu
    elif act_name == "selu":
        return nn.SELU() if return_type == "nn" else F.selu
    elif act_name == "relu":
        return nn.ReLU() if return_type == "nn" else F.relu
    elif act_name == "crelu":
        return nn.ReLU() if return_type == "nn" else F.relu
    elif act_name == "lrelu":
        return nn.LeakyReLU() if return_type == "nn" else F.leaky_relu
    elif act_name == "tanh":
        return nn.Tanh() if return_type == "nn" else F.tanh
    elif act_name == "sigmoid":
        return nn.Sigmoid() if return_type == "nn" else F.sigmoid
    else:
        print("invalid activation function!")
        return None
