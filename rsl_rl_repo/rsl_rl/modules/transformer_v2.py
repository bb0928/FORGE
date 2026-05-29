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
from torch.nn.modules import rnn
import math


class RoPEPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=1000, base=10000):
        super(RoPEPositionalEncoding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Create frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x shape: (batch_size, seq_len, dim)
        batch_size, seq_len, _ = x.shape

        # Create position indices
        position = torch.arange(seq_len, device=x.device, dtype=torch.float32)

        # Calculate frequencies
        freqs = torch.outer(position, self.inv_freq)  # (seq_len, dim//2)

        # Create rotation matrices
        cos_freqs = torch.cos(freqs)  # (seq_len, dim//2)
        sin_freqs = torch.sin(freqs)  # (seq_len, dim//2)

        # Apply RoPE rotation
        x_rotated = self.apply_rope(x, cos_freqs, sin_freqs)

        return x_rotated

    def apply_rope(self, x, cos_freqs, sin_freqs):
        # x shape: (batch_size, seq_len, dim)
        # cos_freqs, sin_freqs shape: (seq_len, dim//2)

        batch_size, seq_len, dim = x.shape

        # Reshape x to separate even and odd dimensions
        x_even = x[:, :, 0::2]  # (batch_size, seq_len, dim//2)
        x_odd = x[:, :, 1::2]  # (batch_size, seq_len, dim//2)

        # Expand cos_freqs and sin_freqs to match batch dimension
        cos_freqs = cos_freqs.unsqueeze(0)  # (1, seq_len, dim//2)
        sin_freqs = sin_freqs.unsqueeze(0)  # (1, seq_len, dim//2)

        # Apply rotation
        x_even_rotated = x_even * cos_freqs - x_odd * sin_freqs
        x_odd_rotated = x_even * sin_freqs + x_odd * cos_freqs

        # Interleave the rotated dimensions
        x_rotated = torch.zeros_like(x)
        x_rotated[:, :, 0::2] = x_even_rotated
        x_rotated[:, :, 1::2] = x_odd_rotated

        return x_rotated


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x shape: (..., dim)
        # TorchScript-friendly RMSNorm implementation
        # rms = sqrt(mean(x^2)), so x / rms is x * rsqrt(mean(x^2))
        rms = torch.mean(x * x, dim=-1, keepdim=True)
        rms = torch.rsqrt(rms + self.eps)
        return self.scale * x * rms


class SwiGLU(nn.Module):
    """SwiGLU activation function: SwiGLU(x) = SiLU(xW) ⊗ (xV)
    where SiLU(x) = x * sigmoid(x) (also known as Swish)
    """

    def __init__(self, input_dim, hidden_dim):
        super(SwiGLU, self).__init__()
        self.w = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, input_dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        # x shape: (..., input_dim)
        # SiLU activation: x * sigmoid(x)
        swish_gate = self.silu(self.w(x))
        # Element-wise multiplication with the value branch
        gated = swish_gate * self.v(x)
        # Project back to input dimension
        return self.output(gated)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Replace standard FFN with SwiGLU
        self.feed_forward = SwiGLU(embed_dim, ff_dim)
        self.rmsnorm1 = RMSNorm(embed_dim)
        self.rmsnorm2 = RMSNorm(embed_dim)

        # Add RoPE for position encoding
        self.rope = RoPEPositionalEncoding(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)

        # Pre-norm: RMS norm before self-attention
        x_norm = self.rmsnorm1(x)

        # Apply RoPE before self-attention
        x_rope = self.rope(x_norm)

        # Self-attention with RoPE encoded positions
        attn_output, _ = self.attention(x_rope, x_rope, x_norm)  # Use normalized x for values
        x = x + attn_output

        # Pre-norm: RMS norm before feed forward
        x_norm2 = self.rmsnorm2(x)
        ff_output = self.feed_forward(x_norm2)
        x = x + ff_output

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim

        # Empty embedding for action prediction (similar to [CLS] token)
        self.empty_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )

        # Final RMS normalization layer
        self.final_norm = RMSNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, T, embed_dim)
        batch_size = x.shape[0]

        # Add empty embedding for action prediction
        empty_emb = self.empty_embedding.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)

        # Concatenate: [empty_embedding, prop_obs, task_obs]
        x = torch.cat((empty_emb, x), dim=1)  # (batch_size, 1 + T, embed_dim)

        # Apply transformer blocks (RoPE is applied inside each block)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Apply final normalization
        x = self.final_norm(x)

        # Use the first token (empty embedding) for prediction
        action_token = x[:, 0, :]  # (batch_size, embed_dim)

        return action_token


class ActorCriticTransformer(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_actor_prop_obs,
        num_actor_task_obs,
        num_critic_prop_obs,
        num_critic_task_obs,
        use_transformer_critic=True,
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
        init_noise_std=1.0,
        fixed_std=False,
        embed_dim=256,
        num_heads=8,
        ff_dim=512,
        num_layers=3,
        **kwargs
    ):
        if kwargs:
            print(
                "ActorCriticTransformer.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(ActorCriticTransformer, self).__init__()

        activation = get_activation(activation)

        # Policy - Replace MLP with Transformer
        self.actor_transformer = TransformerEncoder(
            input_dim=num_actor_obs,
            prop_obs_dim=num_actor_prop_obs,
            task_obs_dim=num_actor_task_obs,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
        )
        # Output layer for actions
        self.actor_output = nn.Linear(embed_dim, num_actions)

        # Value function - Replace MLP with Transformer
        self.use_transformer_critic = use_transformer_critic
        if self.use_transformer_critic:
            self.critic_transformer = TransformerEncoder(
                input_dim=num_critic_obs,
                prop_obs_dim=num_critic_prop_obs,
                task_obs_dim=num_critic_task_obs,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_layers=num_layers,
            )
            # Output layer for value
            self.critic_output = nn.Linear(embed_dim, 1)
        else:
            # Value function
            critic_layers = []
            critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
            critic_layers.append(activation)
            for l in range(len(critic_hidden_dims)):
                if l == len(critic_hidden_dims) - 1:
                    critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                else:
                    critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                    critic_layers.append(activation)
            self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

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

    def update_distribution(self, observations):
        transformer_output = self.actor_transformer(observations)
        mean = self.actor_output(transformer_output)
        std = self.std.to(mean.device)
        # if torch.isnan(mean).any() or torch.isinf(mean).any():
        #     print("loc 张量包含无效值！")
        # if torch.isnan(observations).any() or torch.isinf(observations).any():
        #     print("obs 张量包含无效值！")
        self.distribution = Normal(mean, mean * 0.0 + std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        transformer_output = self.actor_transformer(observations)
        actions_mean = self.actor_output(transformer_output)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        if self.use_transformer_critic:
            transformer_output = self.critic_transformer(critic_observations)
            value = self.critic_output(transformer_output)
        else:
            value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
