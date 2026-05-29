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
from .masked_actor_critic_v2 import MaskedActorCriticV2

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


class MaskedActorCriticV2Residual(nn.Module):
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
        min_noise_std=1e-3,
        max_noise_std=1.0,
        fix_waist=True,
        freq_control=False,
        one_step_obs_dims=None,
        one_step_privileged_obs_dims=None,
        base_model_params=None,
        base_model_pretrained_path="",
        lower_dof_indices=None,
        upper_dof_indices=None,
        init_critic_from_base_model=False,
        **kwargs,
    ):
        # if kwargs:
        #     print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(MaskedActorCriticV2Residual, self).__init__()

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
        
        self.base_model = self.make_base_model(base_model_params, base_model_pretrained_path, **kwargs)
        
        self.lower_dof_indices = lower_dof_indices
        self.upper_dof_indices = upper_dof_indices

        # 快速查找每个观测在残差环境中确切位置-xlj1124--------
        self.residual_obs_offsets = {}
        offset = 0
        for k, v in self.one_step_obs_dims.items():
            self.residual_obs_offsets[k] = (offset, offset + v)
            offset += v
        #--------------------------------
        
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

        self.privileged_obs_offsets = {}
        offset = 0
        for k, v in self.one_step_privileged_obs_dims.items():
            self.privileged_obs_offsets[k] = (offset, offset + v)
            offset += v
        self.actor_group_count = len(self.one_step_obs_dims)
        self.privileged_group_count = len(self.one_step_privileged_obs_dims)
        self.privileged_group_indices = {name: idx for idx, name in enumerate(self.one_step_privileged_obs_dims.keys())}

        default_adaptation_groups = [
            "wrist_forces",
            "wrist_pos",  # 原始wrist位置，而不是virtual target（与B2Z1对齐）
            "base_lin_vel",
        ]
        self.adaptation_groups = kwargs.get("adaptation_groups", default_adaptation_groups)
        self.adaptation_encoder_dims = kwargs.get("adaptation_encoder_dims", [256, 128])
        self.adaptation_decoder_dims = kwargs.get("adaptation_decoder_dims", [128, 64])
        self.adaptation_latent_dim = kwargs.get("adaptation_latent_dim", 0)
        self.adaptation_loss_scales = kwargs.get("adaptation_loss_scales", {})
        self.adaptation_history_length = kwargs.get("adaptation_history_length", 1)  # 默认1帧保持兼容性
        self.wrist_force_loss_active_eps = kwargs.get("wrist_force_loss_active_eps", 1.0)
        self.wrist_force_inactive_reg_coef = kwargs.get("wrist_force_inactive_reg_coef", 0.01)

        self._build_adaptation_modules(activation)

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
        base_actor_input_dim = self.transformer_latent_dim + (self.adaptation_latent_dim if self.use_adaptation else 0)
        if self.actor_use_height:
            pass  # TODO: caculate terrain token
        else:
            mlp_input_dim_a = base_actor_input_dim
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
        # self.initialize_weights_zero(self.actor)
        # Initialize the final layer of the actor to output zeros.
        # This ensures the initial residual actions are zero, providing a stable start for learning.
        last_actor_layer = self.actor[-1]
        if isinstance(last_actor_layer, nn.Linear):
            #torch.nn.init.zeros_(last_actor_layer.weight)
            torch.nn.init.normal_(last_actor_layer.weight, mean=0.0, std=0.01)
            torch.nn.init.zeros_(last_actor_layer.bias)
        # import pdb; pdb.set_trace()
        if init_critic_from_base_model:
            self.init_critic_from_base_model(self.base_model)
        # import pdb; pdb.set_trace()
        
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        if self.use_adaptation:
            print(f"Adaptation Encoder: {self.adaptation_encoder_module}")
            print(f"Adaptation Decoder: {self.adaptation_decoder_module}")
            print(f"Adaptation History Length: {self.adaptation_history_length} frames")
        # print(f"Estimator: {self.estimator.encoder}")
        if self.actor_use_height:
            print(f"Terrain Encoder: {self.terrain_encoder}")

        # Action noise
        # print(init_noise_std)
        
        self.std_lower = nn.Parameter(init_noise_std * 0.1 * torch.ones(len(self.lower_dof_indices))) # 下半身给一个更小的初始std
        self.std_upper = nn.Parameter(init_noise_std * torch.ones(len(self.upper_dof_indices))) # 上半身保持正常初始std

        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        self.init_noise_std = init_noise_std
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
    
    def load_state_dict(self, state_dict, strict=True, init_std_upper=False):
        # 自定义 load_state_dict 方法，以处理 std 到 std_lower 和 std_upper 的转换
        std_loaded = state_dict.pop("std", None)
        
        # 处理adaptation模块和privileged_obs_encoder的维度不匹配
        # 检测checkpoint中的维度与当前模型是否匹配
        keys_to_remove = []
        for key in list(state_dict.keys()):
            if any(prefix in key for prefix in ["adaptation_encoder_module", "adaptation_decoder_module", "privileged_obs_encoder"]):
                try:
                    # 获取当前模型中对应参数的形状
                    current_param = None
                    for name, param in self.named_parameters():
                        if name == key:
                            current_param = param
                            break
                    for name, buffer in self.named_buffers():
                        if name == key:
                            current_param = buffer
                            break
                    
                    if current_param is not None:
                        checkpoint_param = state_dict[key]
                        if current_param.shape != checkpoint_param.shape:
                            print(f"Warning: Skipping '{key}' due to shape mismatch: "
                                  f"checkpoint {tuple(checkpoint_param.shape)} vs current {tuple(current_param.shape)}")
                            keys_to_remove.append(key)
                    else:
                        print(f"Warning: Skipping unexpected key '{key}' from checkpoint")
                        keys_to_remove.append(key)
                except Exception as e:
                    print(f"Warning: Error processing '{key}': {e}")
                    keys_to_remove.append(key)
        
        # 移除不匹配的参数
        for key in keys_to_remove:
            state_dict.pop(key, None)
        
        if std_loaded is not None and 'std_lower' not in state_dict:
            # 从旧的只包含 std 的 state_dict 加载
            super().load_state_dict(state_dict, strict=False)
            with torch.no_grad():
                self.std_lower.copy_(std_loaded[self.lower_dof_indices])
                self.std_upper.copy_(std_loaded[self.upper_dof_indices])
        else:
            # 使用strict=False允许部分加载（adaptation模块可能不匹配）
            print(f"Loading state dict with strict=False to handle dimension mismatches...")
            super().load_state_dict(state_dict, strict=False)
        
        if init_std_upper:
            with torch.no_grad():
                self.std_upper.fill_(self.init_noise_std)
        
        # 报告哪些adaptation模块使用了随机初始化
        if keys_to_remove:
            print(f"\nNote: {len(keys_to_remove)} adaptation module parameters were re-initialized due to dimension changes.")
            print("These modules may need fine-tuning for optimal performance.")
    def make_base_model(self, base_model_params_dict, base_model_pretrained_path="", **kwargs):
        num_actor_obs = base_model_params_dict.num_observations # 
        num_privileged_obs = base_model_params_dict.num_privileged_obs
        num_obs = base_model_params_dict.num_observations
        
        num_one_step_obs = base_model_params_dict.num_one_step_observations # 
        num_one_step_privileged_obs = base_model_params_dict.num_one_step_privileged_obs
        
        if num_privileged_obs is not None:
            num_critic_obs = num_privileged_obs #
            num_one_step_critic_obs = num_one_step_privileged_obs #
        else:
            num_critic_obs = num_obs
            num_one_step_critic_obs = num_one_step_obs
        
        actor_history_length = base_model_params_dict.num_actor_history #
        critic_history_length = base_model_params_dict.num_critic_history #
        num_actions = base_model_params_dict.num_actions #
        fix_waist = base_model_params_dict.fix_waist #
        freq_control = base_model_params_dict.freq_control #
        one_step_obs_dims = base_model_params_dict.one_step_obs_dims #
        one_step_privileged_obs_dims = base_model_params_dict.one_step_privileged_obs_dims #
        # import pdb; pdb.set_trace()
        others = kwargs["base_model_policy"]
        
        base_model = MaskedActorCriticV2(
            num_actor_obs,
            num_critic_obs,
            num_one_step_obs,
            num_one_step_critic_obs,
            actor_history_length,
            critic_history_length,
            num_actions,
            fix_waist=fix_waist,
            freq_control=freq_control,
            one_step_obs_dims=one_step_obs_dims,
            one_step_privileged_obs_dims=one_step_privileged_obs_dims,
            **others,
        )
        if base_model_pretrained_path != "":
            base_model.load_state_dict(torch.load(base_model_pretrained_path)["model_state_dict"])
        self.frozen_parameters(base_model)
        return base_model
    
    def frozen_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
    def init_critic_from_base_model(self, base_model):
        self.privileged_obs_encoder.load_state_dict(base_model.privileged_obs_encoder.state_dict())
        if self.critic_transformer is not None and base_model.critic_transformer is not None:
            self.critic_transformer.load_state_dict(base_model.critic_transformer.state_dict())
        self.critic.load_state_dict(base_model.critic.state_dict())
        
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def initialize_weights_zero(self, model: nn.Module):
        """
        Initialize all weights and biases of the model to zero, including layers in Transformer.
        
        Args:
            model (nn.Module): The neural network model to initialize.
        """
        for name, module in model.named_children():
            # 如果是卷积层或者线性层
            if isinstance(module, (nn.Linear)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.zeros_(module.weight)  # 将权重初始化为0
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)    # 将偏置初始化为0
            # 对子模块进行递归初始化
            if len(list(module.children())) > 0:
                self.initialize_weights_zero(module)
            
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    @property
    def std(self):
        # 按正确的顺序拼接std
        full_std = torch.zeros(self.num_actions, device=self.std_lower.device)
        full_std[self.lower_dof_indices] = self.std_lower
        full_std[self.upper_dof_indices] = self.std_upper
        return full_std
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @property
    def clamped_std(self):
        # 分别裁剪后再拼接
        clamped_lower = torch.clamp(self.std_lower, min=self.min_noise_std, max=self.max_noise_std * 0.2) # 给下半身更严格的上限
        clamped_upper = torch.clamp(self.std_upper, min=self.min_noise_std, max=self.max_noise_std)

        full_clamped_std = torch.zeros(self.num_actions, device=clamped_lower.device)
        full_clamped_std[self.lower_dof_indices] = clamped_lower
        full_clamped_std[self.upper_dof_indices] = clamped_upper
        return full_clamped_std

    def update_distribution(self, obs_history, obs_mask):

        obs_embedding = self.obs_encoder(obs_history, obs_mask)

        if self.actor_use_height:
            raise NotImplementedError("Terrain encoder is not implemented yet")
        else:
            actor_token = self.actor_transformer(obs_embedding).squeeze(1)
            if self.use_adaptation:
                adaptation_latent, adaptation_predictions = self._run_adaptation_modules(obs_history, obs_mask)
                # Store adaptation predictions for use in act() method
                self.last_adaptation_predictions = adaptation_predictions
                actor_input = torch.cat((actor_token, adaptation_latent), dim=-1)
            else:
                self.last_adaptation_predictions = None
                actor_input = actor_token
        action_mean = self.actor(actor_input)
        current_std = self.clamped_std
        self.distribution = Normal(action_mean, action_mean * 0.0 + current_std)

    

    
    def get_base_model_obs_history(self, obs_dims, obs_history, obs_mask):
        # import pdb; pdb.set_trace()
        
        batch_size = obs_history.shape[0]
        obs_history = obs_history.view(batch_size, self.actor_history_length, -1)
        base_model_obs_spec = self.base_model.one_step_obs_dims
        tmp_obs_list = []
        total_extracted_dims = 0
        for obs_name in base_model_obs_spec.keys():
            start_dim, end_dim = self.residual_obs_offsets[obs_name]
            source_block = obs_history[:,:, start_dim:end_dim]
            if obs_name == "action_actual":
                source_block = source_block[..., self.lower_dof_indices]
                extracted_dims = len(self.lower_dof_indices)
                tmp_obs_list.append(source_block)
                #print(f"DEBUG: action_actual extracted {extracted_dims} dims (expected {base_model_obs_spec[obs_name]})")
            else:
                extracted_dims = source_block.shape[-1]
                tmp_obs_list.append(source_block)
                #print(f"DEBUG: {obs_name} extracted {extracted_dims} dims (expected {base_model_obs_spec[obs_name]})")
            total_extracted_dims += extracted_dims
        
        base_model_history = torch.cat(tmp_obs_list, dim=-1).view(batch_size, -1)
        #print(f"DEBUG: Total extracted dims: {total_extracted_dims}, base_model_history shape: {base_model_history.shape}")
        #print(f"DEBUG: Expected total dims: {sum(base_model_obs_spec.values())}")
        
        # 为base model创建正确的mask：只包含base model需要的观测类别
        base_model_obs_spec_keys = list(base_model_obs_spec.keys())
        residual_obs_spec_keys = list(obs_dims.keys())
        
        # 构建base model的mask，只选择base model需要的维度
        base_mask_indices = []
        for base_key in base_model_obs_spec_keys:
            if base_key in residual_obs_spec_keys:
                residual_idx = residual_obs_spec_keys.index(base_key)
                base_mask_indices.append(residual_idx)
        
        # obs_mask shape: [batch, history_len, num_obs_categories]
        obs_mask_reshaped = obs_mask.view(batch_size, self.actor_history_length, -1)
        base_model_obs_mask = obs_mask_reshaped[:, :, base_mask_indices].contiguous().view(batch_size, -1)
        
        # print(f"DEBUG: base_model_obs_mask shape: {base_model_obs_mask.shape}")
        # print(f"DEBUG: base_mask_indices: {base_mask_indices}")
        # print(f"DEBUG: obs_mask original shape: {obs_mask.shape}")
        # Follow env output layout: [history, mask]
        final_base_obs = torch.cat([base_model_history, base_model_obs_mask], dim=-1)
        # print(f"DEBUG: final_base_obs shape: {final_base_obs.shape}")
        return final_base_obs

    
    def act(self, obs_batch, **kwargs):

        batch_size = obs_batch.shape[0]
        obs_history, obs_mask = self._split_actor_obs(obs_batch)

        base_model_obs_history = self.get_base_model_obs_history(self.one_step_obs_dims, obs_history, obs_mask)
        base_action = self.base_model.act_inference(base_model_obs_history)

        self.update_distribution(obs_history, obs_mask)
        action = self.distribution.sample()
        
        # Extract wrist_forces prediction from adaptation predictions (if available)
        if self.use_adaptation and self.last_adaptation_predictions is not None:
            # Find wrist_forces in adaptation_target_layout
            wrist_forces_layout = None
            for layout in self.adaptation_target_layout:
                if layout["name"] == "wrist_forces":
                    wrist_forces_layout = layout
                    break
            
            if wrist_forces_layout is not None:
                start_pred, end_pred = wrist_forces_layout["pred_slice"]
                force_residual_pred = self.last_adaptation_predictions[:, start_pred:end_pred]  # [B, 6]
            else:
                # If wrist_forces not in adaptation groups, return zero tensor
                force_residual_pred = torch.zeros(
                    action.shape[0], 6,
                    device=action.device,
                    dtype=action.dtype
                )
        else:
            # If adaptation is disabled, return zero tensor
            force_residual_pred = torch.zeros(
                action.shape[0], 6,
                device=action.device,
                dtype=action.dtype
            )
        
        # import pdb; pdb.set_trace()
        # action = torch.cat([base_action+action[..., :len(self.lower_dof_indices)], action[..., len(self.lower_dof_indices):]], dim=-1)
        return action, base_action, force_residual_pred
        
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_batch, observations=None):
        obs_history, obs_mask = self._split_actor_obs(obs_batch)

        base_model_obs_history = self.get_base_model_obs_history(self.one_step_obs_dims, obs_history, obs_mask)
        base_action = self.base_model.act_inference(base_model_obs_history)
        # # --------------xlj调试修改1123,下半身base action上半身全部为0-----------------
        # residual_output_shape = list(base_action.shape)
        # residual_output_shape[-1] = self.num_actions
        # full_action = torch.zeros(residual_output_shape).to(base_action.device)
        # full_action[..., self.lower_dof_indices] = base_action
        # import pdb; pdb.set_trace()
        # return full_action
        # # --------------------------------------------------------------
        #import pdb; pdb.set_trace()
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
            actor_token = self.actor_transformer(obs_embedding).squeeze(1)
            if self.use_adaptation:
                adaptation_latent, adaptation_predictions = self._run_adaptation_modules(obs_history, obs_mask)
                actor_input = torch.cat((actor_token, adaptation_latent), dim=-1)
            else:
                actor_input = actor_token
                adaptation_predictions = None
        action_mean = self.actor(actor_input)
        
        # 将base_action与残差网络的输出相加，得到最终动作
        # 恢复下半身的残差叠加
        final_action_mean = action_mean.clone()
        final_action_mean[..., self.lower_dof_indices] += base_action
        
        # Extract wrist_forces prediction from adaptation predictions (if available)
        # Adaptation predictions are in the same order as adaptation_target_layout
        if self.use_adaptation and adaptation_predictions is not None:
            # Find wrist_forces in adaptation_target_layout
            wrist_forces_layout = None
            for layout in self.adaptation_target_layout:
                if layout["name"] == "wrist_forces":
                    wrist_forces_layout = layout
                    break
            
            if wrist_forces_layout is not None:
                start_pred, end_pred = wrist_forces_layout["pred_slice"]
                adaptation_pred_local = adaptation_predictions[:, start_pred:end_pred]  # [B, 6] (left+right wrist forces)
            else:
                # If wrist_forces not in adaptation groups, return zero tensor
                adaptation_pred_local = torch.zeros(
                    adaptation_predictions.shape[0], 6, 
                    device=adaptation_predictions.device, 
                    dtype=adaptation_predictions.dtype
                )
        else:
            # If adaptation is disabled, return zero tensor
            adaptation_pred_local = torch.zeros(
                final_action_mean.shape[0], 6,
                device=final_action_mean.device,
                dtype=final_action_mean.dtype
            )
        
        return final_action_mean, adaptation_pred_local

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

    def _split_actor_obs(self, obs_batch):
        mask_len = self.actor_history_length * len(self.one_step_obs_dims)
        if mask_len == 0:
            return obs_batch, None
        obs_mask = obs_batch[:, -mask_len:]
        obs_history = obs_batch[:, :-mask_len]
        return obs_history, obs_mask

    def _split_critic_obs(self, critic_batch):
        mask_len = self.critic_history_length * len(self.one_step_privileged_obs_dims)
        if mask_len == 0:
            return critic_batch, None
        obs_mask = critic_batch[:, -mask_len:]
        obs_history = critic_batch[:, :-mask_len]
        return obs_history, obs_mask

    def _apply_actor_mask(self, obs, group_mask):
        if group_mask is None:
            return obs
        masked = obs.clone()
        start = 0
        for idx, dim in enumerate(self.one_step_obs_dims.values()):
            mask_val = group_mask[:, idx].unsqueeze(-1)
            masked[:, start : start + dim] = masked[:, start : start + dim] * mask_val
            start += dim
        return masked

    def _run_adaptation_modules(self, obs_history, obs_mask):
        if not self.use_adaptation:
            empty = torch.zeros(obs_history.shape[0], 0, device=obs_history.device, dtype=obs_history.dtype)
            return empty, empty

        if obs_mask is None:
            obs_mask = torch.ones(
                obs_history.shape[0],
                self.actor_history_length * self.actor_group_count,
                device=obs_history.device,
                dtype=obs_history.dtype,
            )
        batch_size = obs_history.shape[0]
        obs_seq = obs_history.view(batch_size, self.actor_history_length, self.num_one_step_obs)
        mask_seq = obs_mask.view(batch_size, self.actor_history_length, self.actor_group_count)
        
        # 使用历史帧而不是只取最后一帧（对齐B2Z1）
        # 如果adaptation_history_length <= actor_history_length，取最后N帧
        num_frames_to_use = min(self.adaptation_history_length, self.actor_history_length)
        adaptation_obs = obs_seq[:, -num_frames_to_use:, :].reshape(batch_size, -1)  # [B, num_frames * num_one_step_obs]
        adaptation_mask_seq = mask_seq[:, -num_frames_to_use:, :]  # [B, num_frames, num_groups]
        
        # 对多帧观测应用mask：对每一帧分别应用mask
        # 将obs reshape成[batch, num_frames, num_one_step_obs]，逐帧处理
        adaptation_obs_reshaped = adaptation_obs.view(batch_size, num_frames_to_use, self.num_one_step_obs)
        masked_obs_list = []
        for t in range(num_frames_to_use):
            frame_obs = adaptation_obs_reshaped[:, t, :]  # [B, num_one_step_obs]
            frame_mask = adaptation_mask_seq[:, t, :]  # [B, num_groups]
            masked_frame = self._apply_actor_mask(frame_obs, frame_mask)
            masked_obs_list.append(masked_frame)
        masked_obs = torch.cat(masked_obs_list, dim=1)  # [B, num_frames * num_one_step_obs]
        
        latent = self.adaptation_encoder_module(masked_obs)
        predictions = self.adaptation_decoder_module(latent)
        return latent, predictions

    def _extract_privileged_targets(self, critic_history, critic_mask):
        if len(self.adaptation_target_layout) == 0:
            return None, None
        batch_size = critic_history.shape[0]
        priv_seq = critic_history.view(batch_size, self.critic_history_length, self.num_one_step_critic_obs)
        current_priv = priv_seq[:, -1, :]
        teacher_chunks = []
        for layout in self.adaptation_target_layout:
            start, end = layout["teacher_slice"]
            teacher_chunks.append(current_priv[:, start:end])
        teacher = torch.cat(teacher_chunks, dim=-1)
        group_mask = None
        if critic_mask is not None:
            mask_seq = critic_mask.view(batch_size, self.critic_history_length, self.privileged_group_count)
            group_mask = mask_seq[:, -1, :]
        return teacher, group_mask

    def compute_adaptation_loss(self, obs_batch, critic_obs_batch, observer_r_wrist_batch=None):
        if not self.use_adaptation or len(self.adaptation_target_layout) == 0:
            zero = obs_batch.new_tensor(0.0)
            return zero, {}

        obs_history, obs_mask = self._split_actor_obs(obs_batch)
        critic_history, critic_mask = self._split_critic_obs(critic_obs_batch)
        if obs_mask is None or critic_history is None:
            zero = obs_batch.new_tensor(0.0)
            return zero, {}

        latent, predictions = self._run_adaptation_modules(obs_history, obs_mask)
        teacher, teacher_group_mask = self._extract_privileged_targets(critic_history, critic_mask)
        if teacher is None:
            zero = obs_batch.new_tensor(0.0)
            return zero, {}

        batch_size = obs_history.shape[0]

        if observer_r_wrist_batch is not None and observer_r_wrist_batch.shape[-1] >= 7:
            active_mask = observer_r_wrist_batch[:, 6:7].to(teacher.dtype)
        else:
            active_mask = torch.ones(batch_size, 1, device=obs_batch.device, dtype=obs_batch.dtype)

        def masked_mean(values, mask):
            mask = mask.to(values.dtype)
            return (values * mask).sum() / (mask.sum().clamp_min(1.0) * values.shape[-1])

        loss = obs_batch.new_tensor(0.0)
        metrics = {}
        for layout in self.adaptation_target_layout:
            start_pred, end_pred = layout["pred_slice"]
            pred_slice = predictions[:, start_pred:end_pred]
            target_slice = teacher[:, start_pred:end_pred]
            if teacher_group_mask is not None and layout["mask_index"] is not None:
                mask_vals = teacher_group_mask[:, layout["mask_index"]].unsqueeze(-1).to(pred_slice.dtype)
            else:
                mask_vals = torch.ones_like(pred_slice[:, :1])

            w_gt = layout["weight"]
            if layout["name"] == "wrist_forces":
                main_mask = mask_vals * active_mask
                inactive_mask = mask_vals * (1.0 - active_mask)
                main_loss = masked_mean((pred_slice - target_slice) ** 2, main_mask)
                zero_reg = masked_mean(pred_slice ** 2, inactive_mask)
                group_loss = w_gt * main_loss + self.wrist_force_inactive_reg_coef * zero_reg
                metrics[f"adapt_{layout['name']}"] = main_loss.detach()
                metrics["adapt_wrist_forces_active_ratio"] = active_mask.mean().detach()
                metrics["adapt_wrist_forces_main"] = main_loss.detach()
                metrics["adapt_wrist_forces_zero_reg"] = zero_reg.detach()
            else:
                group_loss_gt = masked_mean((pred_slice - target_slice) ** 2, mask_vals)
                group_loss = w_gt * group_loss_gt
                metrics[f"adapt_{layout['name']}"] = group_loss_gt.detach()

            loss = loss + group_loss

        return loss, metrics

    def _build_adaptation_modules(self, activation):
        self.adaptation_target_layout = []
        pred_cursor = 0
        valid_groups = []
        for group in self.adaptation_groups:
            if group not in self.one_step_privileged_obs_dims:
                continue
            start, end = self.privileged_obs_offsets[group]
            dim = end - start
            if dim <= 0:
                continue
            layout = {
                "name": group,
                "teacher_slice": (start, end),
                "pred_slice": (pred_cursor, pred_cursor + dim),
                "weight": self.adaptation_loss_scales.get(group, 1.0),
                "mask_index": self.privileged_group_indices.get(group, None),
            }
            pred_cursor += dim
            self.adaptation_target_layout.append(layout)
            valid_groups.append(group)

        self.adaptation_groups = valid_groups
        self.adaptation_num_outputs = pred_cursor
        self.use_adaptation = self.adaptation_latent_dim > 0 and self.adaptation_num_outputs > 0

        if not self.use_adaptation:
            self.adaptation_encoder_module = None
            self.adaptation_decoder_module = None
            return

        # 使用多帧历史观测作为adaptation encoder输入（对齐B2Z1）
        adaptation_input_dim = self.num_one_step_obs * self.adaptation_history_length
        self.adaptation_encoder_module = self._build_mlp(
            input_dim=adaptation_input_dim,
            hidden_dims=self.adaptation_encoder_dims,
            output_dim=self.adaptation_latent_dim,
            activation=activation,
        )
        self.adaptation_decoder_module = self._build_mlp(
            input_dim=self.adaptation_latent_dim,
            hidden_dims=self.adaptation_decoder_dims,
            output_dim=self.adaptation_num_outputs,
            activation=activation,
        )

    def _build_mlp(self, input_dim, hidden_dims, output_dim, activation):
        layers = []
        if not hidden_dims:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(activation)
            for l in range(len(hidden_dims)):
                if l == len(hidden_dims) - 1:
                    layers.append(nn.Linear(hidden_dims[l], output_dim))
                else:
                    layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                    layers.append(activation)
        return nn.Sequential(*layers)

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

