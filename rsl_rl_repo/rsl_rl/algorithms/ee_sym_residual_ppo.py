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

import torch
import torch.nn as nn
import torch.optim as optim
import math

from rsl_rl.storage import RolloutStorage


class EESymResidualPPO:

    def __init__(
        self,
        actor_critic,
        use_flip=True,
        fix_waist=True,
        freq_control=False,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        symmetry_scale=1e-3,
        adaptation_loss_coef=0.0,
        fabric=None,
    ):

        self.device = device
        self.fabric = fabric
        self.use_flip = use_flip

        self.fix_waist = fix_waist
        self.freq_control = freq_control

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()
        self.transition_sym = RolloutStorage.Transition()
        self.symmetry_scale = symmetry_scale
        self.adaptation_loss_coef = adaptation_loss_coef
        self.enable_adaptation_loss = (
            adaptation_loss_coef > 0.0
            and hasattr(self.actor_critic, "compute_adaptation_loss")
            and getattr(self.actor_critic, "use_adaptation", False)
        )
        self.mean_adaptation_loss = 0.0
        self.latest_adaptation_metrics = {}
        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.lower_dof_indices = self.actor_critic.lower_dof_indices
        self.upper_dof_indices = self.actor_critic.upper_dof_indices

    @staticmethod
    def _flip_wrist_residual(observer_r_wrist: torch.Tensor) -> torch.Tensor:
        """Mirror wrist residuals for symmetric rollouts.

        Expected layout: [lx, ly, lz, rx, ry, rz] or
        [lx, ly, lz, rx, ry, rz, wrist_force_active].
        Mirror across sagittal plane:
        - swap left/right
        - flip Y sign
        - keep the optional active flag unchanged
        """
        if observer_r_wrist is None:
            return None
        if observer_r_wrist.numel() == 0:
            return observer_r_wrist
        if observer_r_wrist.shape[-1] not in (6, 7):
            return observer_r_wrist
        wrist_active = observer_r_wrist[:, 6:7] if observer_r_wrist.shape[-1] == 7 else None
        left = observer_r_wrist[:, 0:3]
        right = observer_r_wrist[:, 3:6]
        left_f = left.clone()
        right_f = right.clone()
        left_f[:, 1] *= -1.0
        right_f[:, 1] *= -1.0
        flipped = torch.cat([right_f, left_f], dim=-1)
        if wrist_active is not None:
            flipped = torch.cat([flipped, wrist_active], dim=-1)
        return flipped
        
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        residual_actions, base_action, force_residual_pred = self.actor_critic.act(obs)
        self.transition.actions = residual_actions.detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        
        # Symmetrical part (no changes needed here)
        obs_sym = self.flip_g1_actor_obs(obs)
        critic_obs_sym = self.flip_g1_critic_obs(critic_obs)
        actions_sym, sym_base_action, sym_force_residual_pred = self.actor_critic.act(obs_sym)
        self.transition_sym.actions = actions_sym.detach()
        self.transition_sym.values = self.actor_critic.evaluate(critic_obs_sym).detach()
        self.transition_sym.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition_sym.actions
        ).detach()
        self.transition_sym.action_mean = self.actor_critic.action_mean.detach()
        self.transition_sym.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition_sym.observations = obs_sym
        self.transition_sym.critic_observations = critic_obs_sym
        
        # --- 恢复残差学习: 下半身为 base + res, 上半身为 res ---
        # 创建一个与残差动作维度相同的零张量
        final_action = self.transition.actions.clone()
        # 将 base_action 加到残差动作的下半身部分
        final_action[..., self.lower_dof_indices] += base_action
        
        # Store force_residual_pred for passing to env.step()
        self.last_force_residual_pred = force_residual_pred.detach()
        
        return final_action, force_residual_pred

    def process_env_step(self, rewards, dones, infos, next_critic_obs, observer_r_wrist=None, **kwargs):
        # self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # next_critic_obs_sym = self.flip_g1_critic_obs(next_critic_obs)
        # self.transition_sym.next_critic_observations = next_critic_obs_sym.clone()
        self.transition_sym.rewards = rewards.clone()
        self.transition_sym.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
            self.transition_sym.rewards += self.gamma * torch.squeeze(
                self.transition_sym.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Store observer residuals directly into transitions (so they get shuffled with obs/actions).
        if observer_r_wrist is None and isinstance(infos, dict) and "observer_r_wrist" in infos:
            observer_r_wrist = infos["observer_r_wrist"]
        if observer_r_wrist is None:
            observer_r_wrist = torch.zeros(rewards.shape[0], 6, dtype=torch.float, device=self.device)
        else:
            observer_r_wrist = observer_r_wrist.to(self.device)

        wrist_force_active = None
        if isinstance(infos, dict) and "wrist_force_active" in infos:
            wrist_force_active = infos["wrist_force_active"].to(self.device)
        if wrist_force_active is None:
            wrist_force_active = torch.zeros(rewards.shape[0], 1, dtype=torch.float, device=self.device)
        elif wrist_force_active.ndim == 1:
            wrist_force_active = wrist_force_active.unsqueeze(-1)

        adaptation_signal = torch.cat([observer_r_wrist, wrist_force_active], dim=-1)
        self.transition.observer_r_wrist = adaptation_signal
        self.transition_sym.observer_r_wrist = self._flip_wrist_residual(adaptation_signal)
        
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.storage.add_transitions(self.transition_sym)
        self.transition.clear()
        self.transition_sym.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        # mean_estimation_loss = 0
        # mean_swap_loss = 0
        mean_actor_sym_loss = 0
        mean_critic_sym_loss = 0
        mean_adaptation_loss = 0
        adaptation_metric_totals = {}
        mean_std_lower = 0
        mean_std_upper = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            observer_r_wrist_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.fabric is not None and self.fabric.world_size > 1:
                        kl_t = torch.tensor([kl_mean.item()], device=self.device)
                        kl_t = self.fabric.all_reduce(kl_t, reduce_op="mean")
                        kl_mean = kl_t[0]

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Estimator Update
            # if self.use_flip:
            flipped_obs_batch = self.flip_g1_actor_obs(obs_batch)
            #     flipped_next_critic_obs_batch = self.flip_g1_critic_obs(next_critic_obs_batch)
            #     estimator_update_obs_batch = torch.cat((obs_batch, flipped_obs_batch), dim=0)
            #     estimator_update_next_critic_obs_batch = torch.cat(
            #         (next_critic_obs_batch, flipped_next_critic_obs_batch), dim=0
            #     )
            # else:
            #     estimator_update_obs_batch = obs_batch
            #     estimator_update_next_critic_obs_batch = next_critic_obs_batch
            # estimation_loss, swap_loss = self.actor_critic.update_estimator(
            #     estimator_update_obs_batch, estimator_update_next_critic_obs_batch, lr=self.learning_rate
            # )

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            if self.use_flip:
                flipped_critic_obs_batch = self.flip_g1_critic_obs(critic_obs_batch)
                actor_sym_loss = self.symmetry_scale * torch.mean(
                    torch.sum(
                        torch.square(
                            self.actor_critic.act_inference(flipped_obs_batch)[0]
                            - self.flip_g1_actions(self.actor_critic.act_inference(obs_batch)[0])
                        ),
                        dim=-1,
                    )
                )
                critic_sym_loss = self.symmetry_scale * torch.mean(
                    torch.square(
                        self.actor_critic.evaluate(flipped_critic_obs_batch)
                        - self.actor_critic.evaluate(critic_obs_batch).detach()
                    )
                )
                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                    + actor_sym_loss
                    + critic_sym_loss
                )
            else:
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            adaptation_loss = obs_batch.new_tensor(0.0)
            adaptation_metrics = {}
            if self.enable_adaptation_loss:
                adaptation_loss, adaptation_metrics = self.actor_critic.compute_adaptation_loss(
                    obs_batch, critic_obs_batch, observer_r_wrist_batch=observer_r_wrist_batch
                )
                mean_adaptation_loss += adaptation_loss.item()
                for key, value in adaptation_metrics.items():
                    adaptation_metric_totals[key] = adaptation_metric_totals.get(key, 0.0) + value.item()

            loss = loss + self.adaptation_loss_coef * adaptation_loss

            # Gradient step
            self.optimizer.zero_grad()
            if self.fabric is not None:
                self.fabric.backward(loss)
            else:
                loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            # mean_estimation_loss += estimation_loss
            # mean_swap_loss += swap_loss
            if self.use_flip:
                mean_actor_sym_loss += actor_sym_loss.item()
                mean_critic_sym_loss += critic_sym_loss.item()
            mean_std_lower += self.actor_critic.std_lower.mean().item()
            mean_std_upper += self.actor_critic.std_upper.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        # mean_estimation_loss /= num_updates
        # mean_swap_loss /= num_updates
        if self.use_flip:
            mean_actor_sym_loss /= num_updates
            mean_critic_sym_loss /= num_updates
        mean_std_lower /= num_updates
        mean_std_upper /= num_updates
        if self.enable_adaptation_loss and num_updates > 0:
            mean_adaptation_loss /= num_updates
            self.latest_adaptation_metrics = {
                key: value / num_updates for key, value in adaptation_metric_totals.items()
            }
            self.mean_adaptation_loss = mean_adaptation_loss
        else:
            self.mean_adaptation_loss = 0.0
            self.latest_adaptation_metrics = {}
        self.storage.clear()

        if self.use_flip:
            return (
                mean_value_loss,
                mean_surrogate_loss,
                mean_actor_sym_loss,
                mean_critic_sym_loss,
                mean_std_lower,
                mean_std_upper,
            )
        else:
            return mean_value_loss, mean_surrogate_loss, 0, 0, mean_std_lower, mean_std_upper

    def flip_vector_fix_waist(self, proprioceptive_obs):

        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        flipped_proprioceptive_obs[:, :, 0] = proprioceptive_obs[:, :, 0]  # x command
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1]  # y command
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2]  # yaw command
        flipped_proprioceptive_obs[:, :, 3] = proprioceptive_obs[:, :, 3]  # height command
        flipped_proprioceptive_obs[:, :, 4] = -proprioceptive_obs[:, :, 4]  # base ang vel roll
        flipped_proprioceptive_obs[:, :, 5] = proprioceptive_obs[:, :, 5]  # base ang vel pitch
        flipped_proprioceptive_obs[:, :, 6] = -proprioceptive_obs[:, :, 6]  # base ang vel yaw
        flipped_proprioceptive_obs[:, :, 7] = proprioceptive_obs[:, :, 7]  # projected gravity x
        flipped_proprioceptive_obs[:, :, 8] = -proprioceptive_obs[:, :, 8]  # projected gravity y
        flipped_proprioceptive_obs[:, :, 9] = proprioceptive_obs[:, :, 9]  # projected gravity z

        # joint pos
        flipped_proprioceptive_obs[:, :, 10] = proprioceptive_obs[:, :, 16]  # lower
        flipped_proprioceptive_obs[:, :, 11] = -proprioceptive_obs[:, :, 17]
        flipped_proprioceptive_obs[:, :, 12] = -proprioceptive_obs[:, :, 18]
        flipped_proprioceptive_obs[:, :, 13] = proprioceptive_obs[:, :, 19]
        flipped_proprioceptive_obs[:, :, 14] = proprioceptive_obs[:, :, 20]
        flipped_proprioceptive_obs[:, :, 15] = -proprioceptive_obs[:, :, 21]
        flipped_proprioceptive_obs[:, :, 16] = proprioceptive_obs[:, :, 10]
        flipped_proprioceptive_obs[:, :, 17] = -proprioceptive_obs[:, :, 11]
        flipped_proprioceptive_obs[:, :, 18] = -proprioceptive_obs[:, :, 12]
        flipped_proprioceptive_obs[:, :, 19] = proprioceptive_obs[:, :, 13]
        flipped_proprioceptive_obs[:, :, 20] = proprioceptive_obs[:, :, 14]
        flipped_proprioceptive_obs[:, :, 21] = -proprioceptive_obs[:, :, 15]

        flipped_proprioceptive_obs[:, :, 22] = -proprioceptive_obs[:, :, 22]  # waist

        flipped_proprioceptive_obs[:, :, 23] = proprioceptive_obs[:, :, 30]  # left shoulder
        flipped_proprioceptive_obs[:, :, 24] = -proprioceptive_obs[:, :, 31]
        flipped_proprioceptive_obs[:, :, 25] = -proprioceptive_obs[:, :, 32]
        flipped_proprioceptive_obs[:, :, 26] = proprioceptive_obs[:, :, 33]  # elbow
        flipped_proprioceptive_obs[:, :, 27] = -proprioceptive_obs[:, :, 34]  # wrist
        flipped_proprioceptive_obs[:, :, 28] = proprioceptive_obs[:, :, 35]
        flipped_proprioceptive_obs[:, :, 29] = -proprioceptive_obs[:, :, 36]

        flipped_proprioceptive_obs[:, :, 30] = proprioceptive_obs[:, :, 23]  # right shoulder
        flipped_proprioceptive_obs[:, :, 31] = -proprioceptive_obs[:, :, 24]
        flipped_proprioceptive_obs[:, :, 32] = -proprioceptive_obs[:, :, 25]
        flipped_proprioceptive_obs[:, :, 33] = proprioceptive_obs[:, :, 26]  # elbow
        flipped_proprioceptive_obs[:, :, 34] = -proprioceptive_obs[:, :, 27]  # wrist
        flipped_proprioceptive_obs[:, :, 35] = proprioceptive_obs[:, :, 28]
        flipped_proprioceptive_obs[:, :, 36] = -proprioceptive_obs[:, :, 29]

        # joint vel
        flipped_proprioceptive_obs[:, :, 10 + 27] = proprioceptive_obs[:, :, 16 + 27]  # lower
        flipped_proprioceptive_obs[:, :, 11 + 27] = -proprioceptive_obs[:, :, 17 + 27]
        flipped_proprioceptive_obs[:, :, 12 + 27] = -proprioceptive_obs[:, :, 18 + 27]
        flipped_proprioceptive_obs[:, :, 13 + 27] = proprioceptive_obs[:, :, 19 + 27]
        flipped_proprioceptive_obs[:, :, 14 + 27] = proprioceptive_obs[:, :, 20 + 27]
        flipped_proprioceptive_obs[:, :, 15 + 27] = -proprioceptive_obs[:, :, 21 + 27]
        flipped_proprioceptive_obs[:, :, 16 + 27] = proprioceptive_obs[:, :, 10 + 27]
        flipped_proprioceptive_obs[:, :, 17 + 27] = -proprioceptive_obs[:, :, 11 + 27]
        flipped_proprioceptive_obs[:, :, 18 + 27] = -proprioceptive_obs[:, :, 12 + 27]
        flipped_proprioceptive_obs[:, :, 19 + 27] = proprioceptive_obs[:, :, 13 + 27]
        flipped_proprioceptive_obs[:, :, 20 + 27] = proprioceptive_obs[:, :, 14 + 27]
        flipped_proprioceptive_obs[:, :, 21 + 27] = -proprioceptive_obs[:, :, 15 + 27]

        flipped_proprioceptive_obs[:, :, 22 + 27] = -proprioceptive_obs[:, :, 22 + 27]  # waist

        flipped_proprioceptive_obs[:, :, 23 + 27] = proprioceptive_obs[:, :, 30 + 27]  # left shoulder
        flipped_proprioceptive_obs[:, :, 24 + 27] = -proprioceptive_obs[:, :, 31 + 27]
        flipped_proprioceptive_obs[:, :, 25 + 27] = -proprioceptive_obs[:, :, 32 + 27]
        flipped_proprioceptive_obs[:, :, 26 + 27] = proprioceptive_obs[:, :, 33 + 27]  # elbow
        flipped_proprioceptive_obs[:, :, 27 + 27] = -proprioceptive_obs[:, :, 34 + 27]  # wrist
        flipped_proprioceptive_obs[:, :, 28 + 27] = proprioceptive_obs[:, :, 35 + 27]
        flipped_proprioceptive_obs[:, :, 29 + 27] = -proprioceptive_obs[:, :, 36 + 27]

        flipped_proprioceptive_obs[:, :, 30 + 27] = proprioceptive_obs[:, :, 23 + 27]  # right shoulder
        flipped_proprioceptive_obs[:, :, 31 + 27] = -proprioceptive_obs[:, :, 24 + 27]
        flipped_proprioceptive_obs[:, :, 32 + 27] = -proprioceptive_obs[:, :, 25 + 27]
        flipped_proprioceptive_obs[:, :, 33 + 27] = proprioceptive_obs[:, :, 26 + 27]  # elbow
        flipped_proprioceptive_obs[:, :, 34 + 27] = -proprioceptive_obs[:, :, 27 + 27]  # wrist
        flipped_proprioceptive_obs[:, :, 35 + 27] = proprioceptive_obs[:, :, 28 + 27]
        flipped_proprioceptive_obs[:, :, 36 + 27] = -proprioceptive_obs[:, :, 29 + 27]

        # joint target
        flipped_proprioceptive_obs[:, :, 10 + 54] = proprioceptive_obs[:, :, 16 + 54]  # lower
        flipped_proprioceptive_obs[:, :, 11 + 54] = -proprioceptive_obs[:, :, 17 + 54]
        flipped_proprioceptive_obs[:, :, 12 + 54] = -proprioceptive_obs[:, :, 18 + 54]
        flipped_proprioceptive_obs[:, :, 13 + 54] = proprioceptive_obs[:, :, 19 + 54]
        flipped_proprioceptive_obs[:, :, 14 + 54] = proprioceptive_obs[:, :, 20 + 54]
        flipped_proprioceptive_obs[:, :, 15 + 54] = -proprioceptive_obs[:, :, 21 + 54]
        flipped_proprioceptive_obs[:, :, 16 + 54] = proprioceptive_obs[:, :, 10 + 54]
        flipped_proprioceptive_obs[:, :, 17 + 54] = -proprioceptive_obs[:, :, 11 + 54]
        flipped_proprioceptive_obs[:, :, 18 + 54] = -proprioceptive_obs[:, :, 12 + 54]
        flipped_proprioceptive_obs[:, :, 19 + 54] = proprioceptive_obs[:, :, 13 + 54]
        flipped_proprioceptive_obs[:, :, 20 + 54] = proprioceptive_obs[:, :, 14 + 54]
        flipped_proprioceptive_obs[:, :, 21 + 54] = -proprioceptive_obs[:, :, 15 + 54]

        return flipped_proprioceptive_obs

    def flip_vector_waist(self, proprioceptive_obs):
        """
        Flip observations for symmetry augmentation (sagittal plane mirroring).
        New observation structure (fix_waist=False, freq_control=False):
        - 0-1: lin_command (x, y)
        - 2: ang_command (yaw)
        - 3: height_command
        - 4-7: feet_pose_command (left x,y, right x,y)
        - 8-13: wrist_pose_command (left x,y,z, right x,y,z)
        - 14-21: wrist_quat_command (left quat 4d, right quat 4d)
        - 22-27: wrist_force_command (left x,y,z, right x,y,z)
        - 28-30: head_pose_command (x, y, z)
        - 31-36: imu_info (ang_vel 3d, projected_gravity 3d)
        - 37-65: dof_pos (29 dims: 12 lower + 1 waist_roll + 2 waist_pitch/yaw + 14 upper)
        - 66-94: dof_vel (29 dims, same structure)
        - 95-123: action_actual (29 dims, same structure)
        - 124-137: dof_residual (14 dims: left 7 + right 7)
        - 138+: privileged obs for critic (wrist_forces, wrist_pos, wrist_virtual_target, base_lin_vel)
        """
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        
        # 1. lin_command (0-1): x不变, y取反
        flipped_proprioceptive_obs[:, :, 0] = proprioceptive_obs[:, :, 0]  # x
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1]  # y
        
        # 2. ang_command (2): yaw取反
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2]  # yaw
        
        # 3. height_command (3): 不变
        flipped_proprioceptive_obs[:, :, 3] = proprioceptive_obs[:, :, 3]
        
        # 4. feet_pose_command (4-7): 交换左右, y取反
        # left x,y -> right x,-y; right x,y -> left x,-y
        flipped_proprioceptive_obs[:, :, 4] = proprioceptive_obs[:, :, 6]  # left x <- right x
        flipped_proprioceptive_obs[:, :, 5] = -proprioceptive_obs[:, :, 7]  # left y <- -right y
        flipped_proprioceptive_obs[:, :, 6] = proprioceptive_obs[:, :, 4]  # right x <- left x
        flipped_proprioceptive_obs[:, :, 7] = -proprioceptive_obs[:, :, 5]  # right y <- -left y
        
        # 5. wrist_pose_command (8-13): 交换左右, y取反
        # left x,y,z -> right x,-y,z; right x,y,z -> left x,-y,z
        flipped_proprioceptive_obs[:, :, 8] = proprioceptive_obs[:, :, 11]  # left x <- right x
        flipped_proprioceptive_obs[:, :, 9] = -proprioceptive_obs[:, :, 12]  # left y <- -right y
        flipped_proprioceptive_obs[:, :, 10] = proprioceptive_obs[:, :, 13]  # left z <- right z
        flipped_proprioceptive_obs[:, :, 11] = proprioceptive_obs[:, :, 8]  # right x <- left x
        flipped_proprioceptive_obs[:, :, 12] = -proprioceptive_obs[:, :, 9]  # right y <- -left y
        flipped_proprioceptive_obs[:, :, 13] = proprioceptive_obs[:, :, 10]  # right z <- left z
        
        # 6. wrist_quat_command (14-21): 交换左右, 四元数镜像处理
        # 四元数 (w, x, y, z) 在镜像时需要: 交换左右, 翻转 y 和 z 分量
        # left quat (14-17) -> right quat with flipped y,z
        flipped_proprioceptive_obs[:, :, 14] = proprioceptive_obs[:, :, 18]  # w
        flipped_proprioceptive_obs[:, :, 15] = proprioceptive_obs[:, :, 19]  # x
        flipped_proprioceptive_obs[:, :, 16] = -proprioceptive_obs[:, :, 20]  # y取反
        flipped_proprioceptive_obs[:, :, 17] = -proprioceptive_obs[:, :, 21]  # z取反
        # right quat (18-21) -> left quat with flipped y,z
        flipped_proprioceptive_obs[:, :, 18] = proprioceptive_obs[:, :, 14]  # w
        flipped_proprioceptive_obs[:, :, 19] = proprioceptive_obs[:, :, 15]  # x
        flipped_proprioceptive_obs[:, :, 20] = -proprioceptive_obs[:, :, 16]  # y取反
        flipped_proprioceptive_obs[:, :, 21] = -proprioceptive_obs[:, :, 17]  # z取反
        
        # 7. wrist_force_command (22-27): 交换左右, y取反
        flipped_proprioceptive_obs[:, :, 22] = proprioceptive_obs[:, :, 25]  # left x <- right x
        flipped_proprioceptive_obs[:, :, 23] = -proprioceptive_obs[:, :, 26]  # left y <- -right y
        flipped_proprioceptive_obs[:, :, 24] = proprioceptive_obs[:, :, 27]  # left z <- right z
        flipped_proprioceptive_obs[:, :, 25] = proprioceptive_obs[:, :, 22]  # right x <- left x
        flipped_proprioceptive_obs[:, :, 26] = -proprioceptive_obs[:, :, 23]  # right y <- -left y
        flipped_proprioceptive_obs[:, :, 27] = proprioceptive_obs[:, :, 24]  # right z <- left z
        
        # 8. head_pose_command (28-30): x不变, y取反, z不变
        flipped_proprioceptive_obs[:, :, 28] = proprioceptive_obs[:, :, 28]  # x
        flipped_proprioceptive_obs[:, :, 29] = -proprioceptive_obs[:, :, 29]  # y
        flipped_proprioceptive_obs[:, :, 30] = proprioceptive_obs[:, :, 30]  # z
        
        # 9. imu_info (31-36): ang_vel (roll取反, pitch不变, yaw取反), projected_gravity (x不变, y取反, z不变)
        flipped_proprioceptive_obs[:, :, 31] = -proprioceptive_obs[:, :, 31]  # ang_vel roll
        flipped_proprioceptive_obs[:, :, 32] = proprioceptive_obs[:, :, 32]  # ang_vel pitch
        flipped_proprioceptive_obs[:, :, 33] = -proprioceptive_obs[:, :, 33]  # ang_vel yaw
        flipped_proprioceptive_obs[:, :, 34] = proprioceptive_obs[:, :, 34]  # projected_gravity x
        flipped_proprioceptive_obs[:, :, 35] = -proprioceptive_obs[:, :, 35]  # projected_gravity y
        flipped_proprioceptive_obs[:, :, 36] = proprioceptive_obs[:, :, 36]  # projected_gravity z
        
        # 10. dof_pos (37-65): 29维, 根据关节类型翻转
        # 下肢: 0-11 (左0-5, 右6-11), 交换左右, roll和yaw取反
        # 腰部: 12-14 (waist_roll, waist_pitch, waist_yaw), roll和yaw取反, pitch不变
        # 上肢: 15-28 (左15-21, 右22-28), 交换左右, roll和yaw取反
        base_idx = 37
        # Lower body: swap left (0-5) and right (6-11)
        flipped_proprioceptive_obs[:, :, base_idx + 0] = proprioceptive_obs[:, :, base_idx + 6]   # left hip pitch <- right
        flipped_proprioceptive_obs[:, :, base_idx + 1] = -proprioceptive_obs[:, :, base_idx + 7]  # left hip roll
        flipped_proprioceptive_obs[:, :, base_idx + 2] = -proprioceptive_obs[:, :, base_idx + 8]  # left hip yaw
        flipped_proprioceptive_obs[:, :, base_idx + 3] = proprioceptive_obs[:, :, base_idx + 9]   # left knee
        flipped_proprioceptive_obs[:, :, base_idx + 4] = proprioceptive_obs[:, :, base_idx + 10]  # left ankle pitch
        flipped_proprioceptive_obs[:, :, base_idx + 5] = -proprioceptive_obs[:, :, base_idx + 11]  # left ankle roll
        flipped_proprioceptive_obs[:, :, base_idx + 6] = proprioceptive_obs[:, :, base_idx + 0]   # right hip pitch <- left
        flipped_proprioceptive_obs[:, :, base_idx + 7] = -proprioceptive_obs[:, :, base_idx + 1]  # right hip roll
        flipped_proprioceptive_obs[:, :, base_idx + 8] = -proprioceptive_obs[:, :, base_idx + 2]  # right hip yaw
        flipped_proprioceptive_obs[:, :, base_idx + 9] = proprioceptive_obs[:, :, base_idx + 3]   # right knee
        flipped_proprioceptive_obs[:, :, base_idx + 10] = proprioceptive_obs[:, :, base_idx + 4]  # right ankle pitch
        flipped_proprioceptive_obs[:, :, base_idx + 11] = -proprioceptive_obs[:, :, base_idx + 5]  # right ankle roll
        # Waist: 12-14 (waist_roll, waist_pitch, waist_yaw)
        flipped_proprioceptive_obs[:, :, base_idx + 12] = -proprioceptive_obs[:, :, base_idx + 12]  # waist_roll
        flipped_proprioceptive_obs[:, :, base_idx + 13] = proprioceptive_obs[:, :, base_idx + 13]  # waist_pitch
        flipped_proprioceptive_obs[:, :, base_idx + 14] = -proprioceptive_obs[:, :, base_idx + 14]  # waist_yaw
        # Upper body: swap left (15-21) and right (22-28)
        flipped_proprioceptive_obs[:, :, base_idx + 15] = proprioceptive_obs[:, :, base_idx + 22]  # left shoulder pitch <- right
        flipped_proprioceptive_obs[:, :, base_idx + 16] = -proprioceptive_obs[:, :, base_idx + 23]  # left shoulder roll
        flipped_proprioceptive_obs[:, :, base_idx + 17] = -proprioceptive_obs[:, :, base_idx + 24]  # left shoulder yaw
        flipped_proprioceptive_obs[:, :, base_idx + 18] = proprioceptive_obs[:, :, base_idx + 25]  # left elbow
        flipped_proprioceptive_obs[:, :, base_idx + 19] = -proprioceptive_obs[:, :, base_idx + 26]  # left wrist roll
        flipped_proprioceptive_obs[:, :, base_idx + 20] = proprioceptive_obs[:, :, base_idx + 27]  # left wrist pitch
        flipped_proprioceptive_obs[:, :, base_idx + 21] = -proprioceptive_obs[:, :, base_idx + 28]  # left wrist yaw
        flipped_proprioceptive_obs[:, :, base_idx + 22] = proprioceptive_obs[:, :, base_idx + 15]  # right shoulder pitch <- left
        flipped_proprioceptive_obs[:, :, base_idx + 23] = -proprioceptive_obs[:, :, base_idx + 16]  # right shoulder roll
        flipped_proprioceptive_obs[:, :, base_idx + 24] = -proprioceptive_obs[:, :, base_idx + 17]  # right shoulder yaw
        flipped_proprioceptive_obs[:, :, base_idx + 25] = proprioceptive_obs[:, :, base_idx + 18]  # right elbow
        flipped_proprioceptive_obs[:, :, base_idx + 26] = -proprioceptive_obs[:, :, base_idx + 19]  # right wrist roll
        flipped_proprioceptive_obs[:, :, base_idx + 27] = proprioceptive_obs[:, :, base_idx + 20]  # right wrist pitch
        flipped_proprioceptive_obs[:, :, base_idx + 28] = -proprioceptive_obs[:, :, base_idx + 21]  # right wrist yaw
        
        # 11. dof_vel (66-94): 29维, 与dof_pos相同的翻转逻辑
        base_idx = 66
        # Lower body
        flipped_proprioceptive_obs[:, :, base_idx + 0] = proprioceptive_obs[:, :, base_idx + 6]
        flipped_proprioceptive_obs[:, :, base_idx + 1] = -proprioceptive_obs[:, :, base_idx + 7]
        flipped_proprioceptive_obs[:, :, base_idx + 2] = -proprioceptive_obs[:, :, base_idx + 8]
        flipped_proprioceptive_obs[:, :, base_idx + 3] = proprioceptive_obs[:, :, base_idx + 9]
        flipped_proprioceptive_obs[:, :, base_idx + 4] = proprioceptive_obs[:, :, base_idx + 10]
        flipped_proprioceptive_obs[:, :, base_idx + 5] = -proprioceptive_obs[:, :, base_idx + 11]
        flipped_proprioceptive_obs[:, :, base_idx + 6] = proprioceptive_obs[:, :, base_idx + 0]
        flipped_proprioceptive_obs[:, :, base_idx + 7] = -proprioceptive_obs[:, :, base_idx + 1]
        flipped_proprioceptive_obs[:, :, base_idx + 8] = -proprioceptive_obs[:, :, base_idx + 2]
        flipped_proprioceptive_obs[:, :, base_idx + 9] = proprioceptive_obs[:, :, base_idx + 3]
        flipped_proprioceptive_obs[:, :, base_idx + 10] = proprioceptive_obs[:, :, base_idx + 4]
        flipped_proprioceptive_obs[:, :, base_idx + 11] = -proprioceptive_obs[:, :, base_idx + 5]
        # Waist
        flipped_proprioceptive_obs[:, :, base_idx + 12] = -proprioceptive_obs[:, :, base_idx + 12]
        flipped_proprioceptive_obs[:, :, base_idx + 13] = proprioceptive_obs[:, :, base_idx + 13]
        flipped_proprioceptive_obs[:, :, base_idx + 14] = -proprioceptive_obs[:, :, base_idx + 14]
        # Upper body
        flipped_proprioceptive_obs[:, :, base_idx + 15] = proprioceptive_obs[:, :, base_idx + 22]
        flipped_proprioceptive_obs[:, :, base_idx + 16] = -proprioceptive_obs[:, :, base_idx + 23]
        flipped_proprioceptive_obs[:, :, base_idx + 17] = -proprioceptive_obs[:, :, base_idx + 24]
        flipped_proprioceptive_obs[:, :, base_idx + 18] = proprioceptive_obs[:, :, base_idx + 25]
        flipped_proprioceptive_obs[:, :, base_idx + 19] = -proprioceptive_obs[:, :, base_idx + 26]
        flipped_proprioceptive_obs[:, :, base_idx + 20] = proprioceptive_obs[:, :, base_idx + 27]
        flipped_proprioceptive_obs[:, :, base_idx + 21] = -proprioceptive_obs[:, :, base_idx + 28]
        flipped_proprioceptive_obs[:, :, base_idx + 22] = proprioceptive_obs[:, :, base_idx + 15]
        flipped_proprioceptive_obs[:, :, base_idx + 23] = -proprioceptive_obs[:, :, base_idx + 16]
        flipped_proprioceptive_obs[:, :, base_idx + 24] = -proprioceptive_obs[:, :, base_idx + 17]
        flipped_proprioceptive_obs[:, :, base_idx + 25] = proprioceptive_obs[:, :, base_idx + 18]
        flipped_proprioceptive_obs[:, :, base_idx + 26] = -proprioceptive_obs[:, :, base_idx + 19]
        flipped_proprioceptive_obs[:, :, base_idx + 27] = proprioceptive_obs[:, :, base_idx + 20]
        flipped_proprioceptive_obs[:, :, base_idx + 28] = -proprioceptive_obs[:, :, base_idx + 21]
        
        # 12. action_actual (95-123): 29维, 与dof_pos相同的翻转逻辑
        base_idx = 95
        # Lower body
        flipped_proprioceptive_obs[:, :, base_idx + 0] = proprioceptive_obs[:, :, base_idx + 6]
        flipped_proprioceptive_obs[:, :, base_idx + 1] = -proprioceptive_obs[:, :, base_idx + 7]
        flipped_proprioceptive_obs[:, :, base_idx + 2] = -proprioceptive_obs[:, :, base_idx + 8]
        flipped_proprioceptive_obs[:, :, base_idx + 3] = proprioceptive_obs[:, :, base_idx + 9]
        flipped_proprioceptive_obs[:, :, base_idx + 4] = proprioceptive_obs[:, :, base_idx + 10]
        flipped_proprioceptive_obs[:, :, base_idx + 5] = -proprioceptive_obs[:, :, base_idx + 11]
        flipped_proprioceptive_obs[:, :, base_idx + 6] = proprioceptive_obs[:, :, base_idx + 0]
        flipped_proprioceptive_obs[:, :, base_idx + 7] = -proprioceptive_obs[:, :, base_idx + 1]
        flipped_proprioceptive_obs[:, :, base_idx + 8] = -proprioceptive_obs[:, :, base_idx + 2]
        flipped_proprioceptive_obs[:, :, base_idx + 9] = proprioceptive_obs[:, :, base_idx + 3]
        flipped_proprioceptive_obs[:, :, base_idx + 10] = proprioceptive_obs[:, :, base_idx + 4]
        flipped_proprioceptive_obs[:, :, base_idx + 11] = -proprioceptive_obs[:, :, base_idx + 5]
        # Waist
        flipped_proprioceptive_obs[:, :, base_idx + 12] = -proprioceptive_obs[:, :, base_idx + 12]
        flipped_proprioceptive_obs[:, :, base_idx + 13] = proprioceptive_obs[:, :, base_idx + 13]
        flipped_proprioceptive_obs[:, :, base_idx + 14] = -proprioceptive_obs[:, :, base_idx + 14]
        # Upper body
        flipped_proprioceptive_obs[:, :, base_idx + 15] = proprioceptive_obs[:, :, base_idx + 22]
        flipped_proprioceptive_obs[:, :, base_idx + 16] = -proprioceptive_obs[:, :, base_idx + 23]
        flipped_proprioceptive_obs[:, :, base_idx + 17] = -proprioceptive_obs[:, :, base_idx + 24]
        flipped_proprioceptive_obs[:, :, base_idx + 18] = proprioceptive_obs[:, :, base_idx + 25]
        flipped_proprioceptive_obs[:, :, base_idx + 19] = -proprioceptive_obs[:, :, base_idx + 26]
        flipped_proprioceptive_obs[:, :, base_idx + 20] = proprioceptive_obs[:, :, base_idx + 27]
        flipped_proprioceptive_obs[:, :, base_idx + 21] = -proprioceptive_obs[:, :, base_idx + 28]
        flipped_proprioceptive_obs[:, :, base_idx + 22] = proprioceptive_obs[:, :, base_idx + 15]
        flipped_proprioceptive_obs[:, :, base_idx + 23] = -proprioceptive_obs[:, :, base_idx + 16]
        flipped_proprioceptive_obs[:, :, base_idx + 24] = -proprioceptive_obs[:, :, base_idx + 17]
        flipped_proprioceptive_obs[:, :, base_idx + 25] = proprioceptive_obs[:, :, base_idx + 18]
        flipped_proprioceptive_obs[:, :, base_idx + 26] = -proprioceptive_obs[:, :, base_idx + 19]
        flipped_proprioceptive_obs[:, :, base_idx + 27] = proprioceptive_obs[:, :, base_idx + 20]
        flipped_proprioceptive_obs[:, :, base_idx + 28] = -proprioceptive_obs[:, :, base_idx + 21]
        
        # 13. dof_residual (124-137): 14维, 左7+右7, 交换左右, roll和yaw取反
        base_idx = 124
        # Left arm 7 joints -> Right arm 7 joints
        flipped_proprioceptive_obs[:, :, base_idx + 0] = proprioceptive_obs[:, :, base_idx + 7]   # left shoulder pitch <- right
        flipped_proprioceptive_obs[:, :, base_idx + 1] = -proprioceptive_obs[:, :, base_idx + 8]  # left shoulder roll
        flipped_proprioceptive_obs[:, :, base_idx + 2] = -proprioceptive_obs[:, :, base_idx + 9]  # left shoulder yaw
        flipped_proprioceptive_obs[:, :, base_idx + 3] = proprioceptive_obs[:, :, base_idx + 10]  # left elbow
        flipped_proprioceptive_obs[:, :, base_idx + 4] = -proprioceptive_obs[:, :, base_idx + 11]  # left wrist roll
        flipped_proprioceptive_obs[:, :, base_idx + 5] = proprioceptive_obs[:, :, base_idx + 12]  # left wrist pitch
        flipped_proprioceptive_obs[:, :, base_idx + 6] = -proprioceptive_obs[:, :, base_idx + 13]  # left wrist yaw
        # Right arm 7 joints -> Left arm 7 joints
        flipped_proprioceptive_obs[:, :, base_idx + 7] = proprioceptive_obs[:, :, base_idx + 0]   # right shoulder pitch <- left
        flipped_proprioceptive_obs[:, :, base_idx + 8] = -proprioceptive_obs[:, :, base_idx + 1]  # right shoulder roll
        flipped_proprioceptive_obs[:, :, base_idx + 9] = -proprioceptive_obs[:, :, base_idx + 2]  # right shoulder yaw
        flipped_proprioceptive_obs[:, :, base_idx + 10] = proprioceptive_obs[:, :, base_idx + 3]  # right elbow
        flipped_proprioceptive_obs[:, :, base_idx + 11] = -proprioceptive_obs[:, :, base_idx + 4]  # right wrist roll
        flipped_proprioceptive_obs[:, :, base_idx + 12] = proprioceptive_obs[:, :, base_idx + 5]  # right wrist pitch
        flipped_proprioceptive_obs[:, :, base_idx + 13] = -proprioceptive_obs[:, :, base_idx + 6]  # right wrist yaw
        
        # 14. Privileged observations for critic (138+): wrist_forces, wrist_pos, wrist_virtual_target, base_lin_vel
        if proprioceptive_obs.shape[-1] > 138:
            # wrist_forces (138-143): 交换左右, y取反
            base_idx = 138
            flipped_proprioceptive_obs[:, :, base_idx + 0] = proprioceptive_obs[:, :, base_idx + 3]  # left x <- right x
            flipped_proprioceptive_obs[:, :, base_idx + 1] = -proprioceptive_obs[:, :, base_idx + 4]  # left y <- -right y
            flipped_proprioceptive_obs[:, :, base_idx + 2] = proprioceptive_obs[:, :, base_idx + 5]  # left z <- right z
            flipped_proprioceptive_obs[:, :, base_idx + 3] = proprioceptive_obs[:, :, base_idx + 0]  # right x <- left x
            flipped_proprioceptive_obs[:, :, base_idx + 4] = -proprioceptive_obs[:, :, base_idx + 1]  # right y <- -left y
            flipped_proprioceptive_obs[:, :, base_idx + 5] = proprioceptive_obs[:, :, base_idx + 2]  # right z <- left z
            
            # wrist_pos (144-149): 交换左右, y取反
            base_idx = 144
            flipped_proprioceptive_obs[:, :, base_idx + 0] = proprioceptive_obs[:, :, base_idx + 3]  # left x <- right x
            flipped_proprioceptive_obs[:, :, base_idx + 1] = -proprioceptive_obs[:, :, base_idx + 4]  # left y <- -right y
            flipped_proprioceptive_obs[:, :, base_idx + 2] = proprioceptive_obs[:, :, base_idx + 5]  # left z <- right z
            flipped_proprioceptive_obs[:, :, base_idx + 3] = proprioceptive_obs[:, :, base_idx + 0]  # right x <- left x
            flipped_proprioceptive_obs[:, :, base_idx + 4] = -proprioceptive_obs[:, :, base_idx + 1]  # right y <- -left y
            flipped_proprioceptive_obs[:, :, base_idx + 5] = proprioceptive_obs[:, :, base_idx + 2]  # right z <- left z
            
            # wrist_virtual_target (150-155): 交换左右, y取反
            base_idx = 150
            flipped_proprioceptive_obs[:, :, base_idx + 0] = proprioceptive_obs[:, :, base_idx + 3]  # left x <- right x
            flipped_proprioceptive_obs[:, :, base_idx + 1] = -proprioceptive_obs[:, :, base_idx + 4]  # left y <- -right y
            flipped_proprioceptive_obs[:, :, base_idx + 2] = proprioceptive_obs[:, :, base_idx + 5]  # left z <- right z
            flipped_proprioceptive_obs[:, :, base_idx + 3] = proprioceptive_obs[:, :, base_idx + 0]  # right x <- left x
            flipped_proprioceptive_obs[:, :, base_idx + 4] = -proprioceptive_obs[:, :, base_idx + 1]  # right y <- -left y
            flipped_proprioceptive_obs[:, :, base_idx + 5] = proprioceptive_obs[:, :, base_idx + 2]  # right z <- left z
            
            # base_lin_vel (156-158): x不变, y取反, z不变
            base_idx = 156
            flipped_proprioceptive_obs[:, :, base_idx + 0] = proprioceptive_obs[:, :, base_idx + 0]  # x
            flipped_proprioceptive_obs[:, :, base_idx + 1] = -proprioceptive_obs[:, :, base_idx + 1]  # y
            flipped_proprioceptive_obs[:, :, base_idx + 2] = proprioceptive_obs[:, :, base_idx + 2]  # z
        
        return flipped_proprioceptive_obs

    def flip_vector_fix_waist_freq(self, proprioceptive_obs):

        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        flipped_proprioceptive_obs[:, :, 0] = proprioceptive_obs[:, :, 0]  # x command
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1]  # y command
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2]  # yaw command
        flipped_proprioceptive_obs[:, :, 3] = proprioceptive_obs[:, :, 3]  # height command
        # Add frequency command here
        flipped_proprioceptive_obs[:, :, 4] = proprioceptive_obs[:, :, 4]  # frequency command
        flipped_proprioceptive_obs[:, :, 4 + 1] = -proprioceptive_obs[:, :, 4 + 1]  # base ang vel roll
        flipped_proprioceptive_obs[:, :, 5 + 1] = proprioceptive_obs[:, :, 5 + 1]  # base ang vel pitch
        flipped_proprioceptive_obs[:, :, 6 + 1] = -proprioceptive_obs[:, :, 6 + 1]  # base ang vel yaw
        flipped_proprioceptive_obs[:, :, 7 + 1] = proprioceptive_obs[:, :, 7 + 1]  # projected gravity x
        flipped_proprioceptive_obs[:, :, 8 + 1] = -proprioceptive_obs[:, :, 8 + 1]  # projected gravity y
        flipped_proprioceptive_obs[:, :, 9 + 1] = proprioceptive_obs[:, :, 9 + 1]  # projected gravity z

        flipped_proprioceptive_obs[:, :, 4 + 1 + 6] = -proprioceptive_obs[:, :, 4 + 1 + 6]  # base ang vel roll
        flipped_proprioceptive_obs[:, :, 5 + 1 + 6] = proprioceptive_obs[:, :, 5 + 1 + 6]  # base ang vel pitch
        flipped_proprioceptive_obs[:, :, 6 + 1 + 6] = -proprioceptive_obs[:, :, 6 + 1 + 6]  # base ang vel yaw
        flipped_proprioceptive_obs[:, :, 7 + 1 + 6] = proprioceptive_obs[:, :, 7 + 1 + 6]  # projected gravity x
        flipped_proprioceptive_obs[:, :, 8 + 1 + 6] = -proprioceptive_obs[:, :, 8 + 1 + 6]  # projected gravity y
        flipped_proprioceptive_obs[:, :, 9 + 1 + 6] = proprioceptive_obs[:, :, 9 + 1 + 6]  # projected gravity z

        # joint pos
        flipped_proprioceptive_obs[:, :, 10 + 1 + 6] = proprioceptive_obs[:, :, 16 + 1 + 6]  # lower
        flipped_proprioceptive_obs[:, :, 11 + 1 + 6] = -proprioceptive_obs[:, :, 17 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 12 + 1 + 6] = -proprioceptive_obs[:, :, 18 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 13 + 1 + 6] = proprioceptive_obs[:, :, 19 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 14 + 1 + 6] = proprioceptive_obs[:, :, 20 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 15 + 1 + 6] = -proprioceptive_obs[:, :, 21 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 16 + 1 + 6] = proprioceptive_obs[:, :, 10 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 17 + 1 + 6] = -proprioceptive_obs[:, :, 11 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 18 + 1 + 6] = -proprioceptive_obs[:, :, 12 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 19 + 1 + 6] = proprioceptive_obs[:, :, 13 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 20 + 1 + 6] = proprioceptive_obs[:, :, 14 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 21 + 1 + 6] = -proprioceptive_obs[:, :, 15 + 1 + 6]

        flipped_proprioceptive_obs[:, :, 22 + 1 + 6] = -proprioceptive_obs[:, :, 22 + 1 + 6]  # waist

        flipped_proprioceptive_obs[:, :, 23 + 1 + 6] = proprioceptive_obs[:, :, 30 + 1 + 6]  # left shoulder
        flipped_proprioceptive_obs[:, :, 24 + 1 + 6] = -proprioceptive_obs[:, :, 31 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 25 + 1 + 6] = -proprioceptive_obs[:, :, 32 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 26 + 1 + 6] = proprioceptive_obs[:, :, 33 + 1 + 6]  # elbow
        flipped_proprioceptive_obs[:, :, 27 + 1 + 6] = -proprioceptive_obs[:, :, 34 + 1 + 6]  # wrist
        flipped_proprioceptive_obs[:, :, 28 + 1 + 6] = proprioceptive_obs[:, :, 35 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 29 + 1 + 6] = -proprioceptive_obs[:, :, 36 + 1 + 6]

        flipped_proprioceptive_obs[:, :, 30 + 1 + 6] = proprioceptive_obs[:, :, 23 + 1 + 6]  # right shoulder
        flipped_proprioceptive_obs[:, :, 31 + 1 + 6] = -proprioceptive_obs[:, :, 24 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 32 + 1 + 6] = -proprioceptive_obs[:, :, 25 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 33 + 1 + 6] = proprioceptive_obs[:, :, 26 + 1 + 6]  # elbow
        flipped_proprioceptive_obs[:, :, 34 + 1 + 6] = -proprioceptive_obs[:, :, 27 + 1 + 6]  # wrist
        flipped_proprioceptive_obs[:, :, 35 + 1 + 6] = proprioceptive_obs[:, :, 28 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 36 + 1 + 6] = -proprioceptive_obs[:, :, 29 + 1 + 6]

        # joint vel
        flipped_proprioceptive_obs[:, :, 10 + 1 + 27 + 6] = proprioceptive_obs[:, :, 16 + 1 + 27 + 6]  # lower
        flipped_proprioceptive_obs[:, :, 11 + 1 + 27 + 6] = -proprioceptive_obs[:, :, 17 + 1 + 27 + 6]
        flipped_proprioceptive_obs[:, :, 12 + 1 + 27 + 6] = -proprioceptive_obs[:, :, 18 + 1 + 27 + 6]
        flipped_proprioceptive_obs[:, :, 13 + 1 + 27 + 6] = proprioceptive_obs[:, :, 19 + 1 + 27 + 6]
        flipped_proprioceptive_obs[:, :, 14 + 1 + 27 + 6] = proprioceptive_obs[:, :, 20 + 1 + 27 + 6]
        flipped_proprioceptive_obs[:, :, 15 + 1 + 27 + 6] = -proprioceptive_obs[:, :, 21 + 1 + 27 + 6]
        flipped_proprioceptive_obs[:, :, 16 + 1 + 27 + 6] = proprioceptive_obs[:, :, 10 + 1 + 27 + 6]
        flipped_proprioceptive_obs[:, :, 17 + 1 + 27 + 6] = -proprioceptive_obs[:, :, 11 + 1 + 27 + 6]
        flipped_proprioceptive_obs[:, :, 18 + 1 + 27 + 6] = -proprioceptive_obs[:, :, 12 + 1 + 27 + 6]
        flipped_proprioceptive_obs[:, :, 19 + 1 + 27 + 6] = proprioceptive_obs[:, :, 13 + 1 + 27 + 6]
        flipped_proprioceptive_obs[:, :, 20 + 1 + 27 + 6] = proprioceptive_obs[:, :, 14 + 1 + 27 + 6]
        flipped_proprioceptive_obs[:, :, 21 + 1 + 27 + 6] = -proprioceptive_obs[:, :, 15 + 1 + 27 + 6]

        flipped_proprioceptive_obs[:, :, 22 + 1 + 27 + 6] = -proprioceptive_obs[:, :, 22 + 1 + 27 + 6]  # waist

        flipped_proprioceptive_obs[:, :, 23 + 1 + 27 + 6] = proprioceptive_obs[:, :, 30 + 1 + 27 + 6]  # left shoulder
        flipped_proprioceptive_obs[:, :, 24 + 1 + 27 + 6] = -proprioceptive_obs[:, :, 31 + 1 + 27 + 6]
        flipped_proprioceptive_obs[:, :, 25 + 1 + 27 + 6] = -proprioceptive_obs[:, :, 32 + 1 + 27 + 6]
        flipped_proprioceptive_obs[:, :, 26 + 1 + 27 + 6] = proprioceptive_obs[:, :, 33 + 1 + 27 + 6]  # elbow
        flipped_proprioceptive_obs[:, :, 27 + 1 + 27 + 6] = -proprioceptive_obs[:, :, 34 + 1 + 27 + 6]  # wrist
        flipped_proprioceptive_obs[:, :, 28 + 1 + 27 + 6] = proprioceptive_obs[:, :, 35 + 1 + 27 + 6]
        flipped_proprioceptive_obs[:, :, 29 + 1 + 27 + 6] = -proprioceptive_obs[:, :, 36 + 1 + 27 + 6]

        flipped_proprioceptive_obs[:, :, 30 + 1 + 27 + 6] = proprioceptive_obs[:, :, 23 + 1 + 27 + 6]  # right shoulder
        flipped_proprioceptive_obs[:, :, 31 + 1 + 27 + 6] = -proprioceptive_obs[:, :, 24 + 1 + 27 + 6]
        flipped_proprioceptive_obs[:, :, 32 + 1 + 27 + 6] = -proprioceptive_obs[:, :, 25 + 1 + 27 + 6]
        flipped_proprioceptive_obs[:, :, 33 + 1 + 27 + 6] = proprioceptive_obs[:, :, 26 + 1 + 27 + 6]  # elbow
        flipped_proprioceptive_obs[:, :, 34 + 1 + 27 + 6] = -proprioceptive_obs[:, :, 27 + 1 + 27 + 6]  # wrist
        flipped_proprioceptive_obs[:, :, 35 + 1 + 27 + 6] = proprioceptive_obs[:, :, 28 + 1 + 27 + 6]
        flipped_proprioceptive_obs[:, :, 36 + 1 + 27 + 6] = -proprioceptive_obs[:, :, 29 + 1 + 27 + 6]

        # joint target
        flipped_proprioceptive_obs[:, :, 10 + 1 + 54 + 6] = proprioceptive_obs[:, :, 16 + 1 + 54 + 6]  # lower
        flipped_proprioceptive_obs[:, :, 11 + 1 + 54 + 6] = -proprioceptive_obs[:, :, 17 + 1 + 54 + 6]
        flipped_proprioceptive_obs[:, :, 12 + 1 + 54 + 6] = -proprioceptive_obs[:, :, 18 + 1 + 54 + 6]
        flipped_proprioceptive_obs[:, :, 13 + 1 + 54 + 6] = proprioceptive_obs[:, :, 19 + 1 + 54 + 6]
        flipped_proprioceptive_obs[:, :, 14 + 1 + 54 + 6] = proprioceptive_obs[:, :, 20 + 1 + 54 + 6]
        flipped_proprioceptive_obs[:, :, 15 + 1 + 54 + 6] = -proprioceptive_obs[:, :, 21 + 1 + 54 + 6]
        flipped_proprioceptive_obs[:, :, 16 + 1 + 54 + 6] = proprioceptive_obs[:, :, 10 + 1 + 54 + 6]
        flipped_proprioceptive_obs[:, :, 17 + 1 + 54 + 6] = -proprioceptive_obs[:, :, 11 + 1 + 54 + 6]
        flipped_proprioceptive_obs[:, :, 18 + 1 + 54 + 6] = -proprioceptive_obs[:, :, 12 + 1 + 54 + 6]
        flipped_proprioceptive_obs[:, :, 19 + 1 + 54 + 6] = proprioceptive_obs[:, :, 13 + 1 + 54 + 6]
        flipped_proprioceptive_obs[:, :, 20 + 1 + 54 + 6] = proprioceptive_obs[:, :, 14 + 1 + 54 + 6]
        flipped_proprioceptive_obs[:, :, 21 + 1 + 54 + 6] = -proprioceptive_obs[:, :, 15 + 1 + 54 + 6]

        # gait
        flipped_proprioceptive_obs[:, :, 22 + 1 + 54 + 6] = proprioceptive_obs[:, :, 23 + 1 + 54 + 6]
        flipped_proprioceptive_obs[:, :, 23 + 1 + 54 + 6] = proprioceptive_obs[:, :, 22 + 1 + 54 + 6]

        return flipped_proprioceptive_obs

    def flip_vector_waist_freq(self, proprioceptive_obs):

        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        flipped_proprioceptive_obs[:, :, 0] = proprioceptive_obs[:, :, 0]  # x command
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1]  # y command
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2]  # yaw command
        flipped_proprioceptive_obs[:, :, 3] = proprioceptive_obs[:, :, 3]  # height command
        # Add frequency command here
        flipped_proprioceptive_obs[:, :, 4] = proprioceptive_obs[:, :, 4]  # frequency command
        flipped_proprioceptive_obs[:, :, 4 + 1] = -proprioceptive_obs[:, :, 4 + 1]  # base ang vel roll
        flipped_proprioceptive_obs[:, :, 5 + 1] = proprioceptive_obs[:, :, 5 + 1]  # base ang vel pitch
        flipped_proprioceptive_obs[:, :, 6 + 1] = -proprioceptive_obs[:, :, 6 + 1]  # base ang vel yaw
        flipped_proprioceptive_obs[:, :, 7 + 1] = proprioceptive_obs[:, :, 7 + 1]  # projected gravity x
        flipped_proprioceptive_obs[:, :, 8 + 1] = -proprioceptive_obs[:, :, 8 + 1]  # projected gravity y
        flipped_proprioceptive_obs[:, :, 9 + 1] = proprioceptive_obs[:, :, 9 + 1]  # projected gravity z

        flipped_proprioceptive_obs[:, :, 4 + 1 + 6] = -proprioceptive_obs[:, :, 4 + 1 + 6]  # base ang vel roll
        flipped_proprioceptive_obs[:, :, 5 + 1 + 6] = proprioceptive_obs[:, :, 5 + 1 + 6]  # base ang vel pitch
        flipped_proprioceptive_obs[:, :, 6 + 1 + 6] = -proprioceptive_obs[:, :, 6 + 1 + 6]  # base ang vel yaw
        flipped_proprioceptive_obs[:, :, 7 + 1 + 6] = proprioceptive_obs[:, :, 7 + 1 + 6]  # projected gravity x
        flipped_proprioceptive_obs[:, :, 8 + 1 + 6] = -proprioceptive_obs[:, :, 8 + 1 + 6]  # projected gravity y
        flipped_proprioceptive_obs[:, :, 9 + 1 + 6] = proprioceptive_obs[:, :, 9 + 1 + 6]  # projected gravity z

        # joint pos
        flipped_proprioceptive_obs[:, :, 10 + 1 + 6] = proprioceptive_obs[:, :, 16 + 1 + 6]  # lower
        flipped_proprioceptive_obs[:, :, 11 + 1 + 6] = -proprioceptive_obs[:, :, 17 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 12 + 1 + 6] = -proprioceptive_obs[:, :, 18 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 13 + 1 + 6] = proprioceptive_obs[:, :, 19 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 14 + 1 + 6] = proprioceptive_obs[:, :, 20 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 15 + 1 + 6] = -proprioceptive_obs[:, :, 21 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 16 + 1 + 6] = proprioceptive_obs[:, :, 10 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 17 + 1 + 6] = -proprioceptive_obs[:, :, 11 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 18 + 1 + 6] = -proprioceptive_obs[:, :, 12 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 19 + 1 + 6] = proprioceptive_obs[:, :, 13 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 20 + 1 + 6] = proprioceptive_obs[:, :, 14 + 1 + 6]
        flipped_proprioceptive_obs[:, :, 21 + 1 + 6] = -proprioceptive_obs[:, :, 15 + 1 + 6]

        flipped_proprioceptive_obs[:, :, 22 + 1 + 6] = -proprioceptive_obs[:, :, 22 + 1 + 6]  # waist yaw
        flipped_proprioceptive_obs[:, :, 23 + 1 + 6] = -proprioceptive_obs[:, :, 23 + 1 + 6]  # waist roll
        flipped_proprioceptive_obs[:, :, 24 + 1 + 6] = proprioceptive_obs[:, :, 24 + 1 + 6]  # waist pitch

        flipped_proprioceptive_obs[:, :, 23 + 1 + 2 + 6] = proprioceptive_obs[:, :, 30 + 1 + 2 + 6]  # left shoulder
        flipped_proprioceptive_obs[:, :, 24 + 1 + 2 + 6] = -proprioceptive_obs[:, :, 31 + 1 + 2 + 6]
        flipped_proprioceptive_obs[:, :, 25 + 1 + 2 + 6] = -proprioceptive_obs[:, :, 32 + 1 + 2 + 6]
        flipped_proprioceptive_obs[:, :, 26 + 1 + 2 + 6] = proprioceptive_obs[:, :, 33 + 1 + 2 + 6]  # elbow
        flipped_proprioceptive_obs[:, :, 27 + 1 + 2 + 6] = -proprioceptive_obs[:, :, 34 + 1 + 2 + 6]  # wrist
        flipped_proprioceptive_obs[:, :, 28 + 1 + 2 + 6] = proprioceptive_obs[:, :, 35 + 1 + 2 + 6]
        flipped_proprioceptive_obs[:, :, 29 + 1 + 2 + 6] = -proprioceptive_obs[:, :, 36 + 1 + 2 + 6]

        flipped_proprioceptive_obs[:, :, 30 + 1 + 2 + 6] = proprioceptive_obs[:, :, 23 + 1 + 2 + 6]  # right shoulder
        flipped_proprioceptive_obs[:, :, 31 + 1 + 2 + 6] = -proprioceptive_obs[:, :, 24 + 1 + 2 + 6]
        flipped_proprioceptive_obs[:, :, 32 + 1 + 2 + 6] = -proprioceptive_obs[:, :, 25 + 1 + 2 + 6]
        flipped_proprioceptive_obs[:, :, 33 + 1 + 2 + 6] = proprioceptive_obs[:, :, 26 + 1 + 2 + 6]  # elbow
        flipped_proprioceptive_obs[:, :, 34 + 1 + 2 + 6] = -proprioceptive_obs[:, :, 27 + 1 + 2 + 6]  # wrist
        flipped_proprioceptive_obs[:, :, 35 + 1 + 2 + 6] = proprioceptive_obs[:, :, 28 + 1 + 2 + 6]
        flipped_proprioceptive_obs[:, :, 36 + 1 + 2 + 6] = -proprioceptive_obs[:, :, 29 + 1 + 2 + 6]

        # joint vel
        flipped_proprioceptive_obs[:, :, 10 + 1 + 27 + 2 + 6] = proprioceptive_obs[:, :, 16 + 1 + 27 + 2 + 6]  # lower
        flipped_proprioceptive_obs[:, :, 11 + 1 + 27 + 2 + 6] = -proprioceptive_obs[:, :, 17 + 1 + 27 + 2 + 6]
        flipped_proprioceptive_obs[:, :, 12 + 1 + 27 + 2 + 6] = -proprioceptive_obs[:, :, 18 + 1 + 27 + 2 + 6]
        flipped_proprioceptive_obs[:, :, 13 + 1 + 27 + 2 + 6] = proprioceptive_obs[:, :, 19 + 1 + 27 + 2 + 6]
        flipped_proprioceptive_obs[:, :, 14 + 1 + 27 + 2 + 6] = proprioceptive_obs[:, :, 20 + 1 + 27 + 2 + 6]
        flipped_proprioceptive_obs[:, :, 15 + 1 + 27 + 2 + 6] = -proprioceptive_obs[:, :, 21 + 1 + 27 + 2 + 6]
        flipped_proprioceptive_obs[:, :, 16 + 1 + 27 + 2 + 6] = proprioceptive_obs[:, :, 10 + 1 + 27 + 2 + 6]
        flipped_proprioceptive_obs[:, :, 17 + 1 + 27 + 2 + 6] = -proprioceptive_obs[:, :, 11 + 1 + 27 + 2 + 6]
        flipped_proprioceptive_obs[:, :, 18 + 1 + 27 + 2 + 6] = -proprioceptive_obs[:, :, 12 + 1 + 27 + 2 + 6]
        flipped_proprioceptive_obs[:, :, 19 + 1 + 27 + 2 + 6] = proprioceptive_obs[:, :, 13 + 1 + 27 + 2 + 6]
        flipped_proprioceptive_obs[:, :, 20 + 1 + 27 + 2 + 6] = proprioceptive_obs[:, :, 14 + 1 + 27 + 2 + 6]
        flipped_proprioceptive_obs[:, :, 21 + 1 + 27 + 2 + 6] = -proprioceptive_obs[:, :, 15 + 1 + 27 + 2 + 6]

        flipped_proprioceptive_obs[:, :, 22 + 1 + 27 + 2 + 6] = -proprioceptive_obs[
            :, :, 22 + 1 + 27 + 2 + 6
        ]  # waist yaw
        flipped_proprioceptive_obs[:, :, 23 + 1 + 27 + 2 + 6] = -proprioceptive_obs[
            :, :, 23 + 1 + 27 + 2 + 6
        ]  # waist roll
        flipped_proprioceptive_obs[:, :, 24 + 1 + 27 + 2 + 6] = proprioceptive_obs[
            :, :, 24 + 1 + 27 + 2 + 6
        ]  # waist pitch

        flipped_proprioceptive_obs[:, :, 23 + 1 + 27 + 4 + 6] = proprioceptive_obs[
            :, :, 30 + 1 + 27 + 4 + 6
        ]  # left shoulder
        flipped_proprioceptive_obs[:, :, 24 + 1 + 27 + 4 + 6] = -proprioceptive_obs[:, :, 31 + 1 + 27 + 4 + 6]
        flipped_proprioceptive_obs[:, :, 25 + 1 + 27 + 4 + 6] = -proprioceptive_obs[:, :, 32 + 1 + 27 + 4 + 6]
        flipped_proprioceptive_obs[:, :, 26 + 1 + 27 + 4 + 6] = proprioceptive_obs[:, :, 33 + 1 + 27 + 4 + 6]  # elbow
        flipped_proprioceptive_obs[:, :, 27 + 1 + 27 + 4 + 6] = -proprioceptive_obs[:, :, 34 + 1 + 27 + 4 + 6]  # wrist
        flipped_proprioceptive_obs[:, :, 28 + 1 + 27 + 4 + 6] = proprioceptive_obs[:, :, 35 + 1 + 27 + 4 + 6]
        flipped_proprioceptive_obs[:, :, 29 + 1 + 27 + 4 + 6] = -proprioceptive_obs[:, :, 36 + 1 + 27 + 4 + 6]

        flipped_proprioceptive_obs[:, :, 30 + 1 + 27 + 4 + 6] = proprioceptive_obs[
            :, :, 23 + 1 + 27 + 4 + 6
        ]  # right shoulder
        flipped_proprioceptive_obs[:, :, 31 + 1 + 27 + 4 + 6] = -proprioceptive_obs[:, :, 24 + 1 + 27 + 4 + 6]
        flipped_proprioceptive_obs[:, :, 32 + 1 + 27 + 4 + 6] = -proprioceptive_obs[:, :, 25 + 1 + 27 + 4 + 6]
        flipped_proprioceptive_obs[:, :, 33 + 1 + 27 + 4 + 6] = proprioceptive_obs[:, :, 26 + 1 + 27 + 4 + 6]  # elbow
        flipped_proprioceptive_obs[:, :, 34 + 1 + 27 + 4 + 6] = -proprioceptive_obs[:, :, 27 + 1 + 27 + 4 + 6]  # wrist
        flipped_proprioceptive_obs[:, :, 35 + 1 + 27 + 4 + 6] = proprioceptive_obs[:, :, 28 + 1 + 27 + 4 + 6]
        flipped_proprioceptive_obs[:, :, 36 + 1 + 27 + 4 + 6] = -proprioceptive_obs[:, :, 29 + 1 + 27 + 4 + 6]

        # joint target
        flipped_proprioceptive_obs[:, :, 10 + 1 + 54 + 4 + 6] = proprioceptive_obs[:, :, 16 + 1 + 54 + 4 + 6]  # lower
        flipped_proprioceptive_obs[:, :, 11 + 1 + 54 + 4 + 6] = -proprioceptive_obs[:, :, 17 + 1 + 54 + 4 + 6]
        flipped_proprioceptive_obs[:, :, 12 + 1 + 54 + 4 + 6] = -proprioceptive_obs[:, :, 18 + 1 + 54 + 4 + 6]
        flipped_proprioceptive_obs[:, :, 13 + 1 + 54 + 4 + 6] = proprioceptive_obs[:, :, 19 + 1 + 54 + 4 + 6]
        flipped_proprioceptive_obs[:, :, 14 + 1 + 54 + 4 + 6] = proprioceptive_obs[:, :, 20 + 1 + 54 + 4 + 6]
        flipped_proprioceptive_obs[:, :, 15 + 1 + 54 + 4 + 6] = -proprioceptive_obs[:, :, 21 + 1 + 54 + 4 + 6]
        flipped_proprioceptive_obs[:, :, 16 + 1 + 54 + 4 + 6] = proprioceptive_obs[:, :, 10 + 1 + 54 + 4 + 6]
        flipped_proprioceptive_obs[:, :, 17 + 1 + 54 + 4 + 6] = -proprioceptive_obs[:, :, 11 + 1 + 54 + 4 + 6]
        flipped_proprioceptive_obs[:, :, 18 + 1 + 54 + 4 + 6] = -proprioceptive_obs[:, :, 12 + 1 + 54 + 4 + 6]
        flipped_proprioceptive_obs[:, :, 19 + 1 + 54 + 4 + 6] = proprioceptive_obs[:, :, 13 + 1 + 54 + 4 + 6]
        flipped_proprioceptive_obs[:, :, 20 + 1 + 54 + 4 + 6] = proprioceptive_obs[:, :, 14 + 1 + 54 + 4 + 6]
        flipped_proprioceptive_obs[:, :, 21 + 1 + 54 + 4 + 6] = -proprioceptive_obs[:, :, 15 + 1 + 54 + 4 + 6]

        flipped_proprioceptive_obs[:, :, 22 + 1 + 54 + 4 + 6] = -proprioceptive_obs[
            :, :, 22 + 1 + 54 + 4 + 6
        ]  # waist roll
        flipped_proprioceptive_obs[:, :, 23 + 1 + 54 + 4 + 6] = proprioceptive_obs[
            :, :, 23 + 1 + 54 + 4 + 6
        ]  # waist pitch

        # gait
        flipped_proprioceptive_obs[:, :, 24 + 1 + 54 + 4 + 6] = proprioceptive_obs[:, :, 25 + 1 + 54 + 4 + 6]
        flipped_proprioceptive_obs[:, :, 25 + 1 + 54 + 4 + 6] = proprioceptive_obs[:, :, 24 + 1 + 54 + 4 + 6]
        

        return flipped_proprioceptive_obs

    def flip_g1_actor_obs(self, obs):
        proprioceptive_obs = torch.clone(
            obs[:, : self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length]
        )
        proprioceptive_obs = proprioceptive_obs.view(
            -1, self.actor_critic.actor_history_length, self.actor_critic.num_one_step_obs
        )

        obs_mask = torch.clone(
            obs[
                :,
                self.actor_critic.num_one_step_obs
                * self.actor_critic.actor_history_length : self.actor_critic.num_one_step_obs
                * self.actor_critic.actor_history_length
                + self.actor_critic.actor_history_length * len(self.actor_critic.one_step_obs_dims),
            ]
        )
        flipped_obs_mask = obs_mask  # TODO: flip obs_mask if needed

        if self.actor_critic.actor_use_height:
            # ! Warning: terrain info must be at the end of the observation vector
            terrain_obs = torch.clone(
                obs[:, self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length :]
            )

        if not self.freq_control:
            flipped_proprioceptive_obs = (
                self.flip_vector_fix_waist(proprioceptive_obs)
                if self.fix_waist
                else self.flip_vector_waist(proprioceptive_obs)
            )
        else:
            flipped_proprioceptive_obs = (
                self.flip_vector_fix_waist_freq(proprioceptive_obs)
                if self.fix_waist
                else self.flip_vector_waist_freq(proprioceptive_obs)
            )

        if self.actor_critic.actor_use_height:
            # terrain flip
            height_x_dim = int(math.sqrt(terrain_obs.shape[-1]))
            assert height_x_dim * height_x_dim == terrain_obs.shape[-1], "Terrain observation must be square"
            terrain_obs = terrain_obs.reshape(-1, height_x_dim, height_x_dim)
            flipped_terrain_obs = torch.flip(terrain_obs, dims=[-1])
            flipped_terrain_obs = flipped_terrain_obs.reshape(-1, height_x_dim * height_x_dim)

            return torch.cat(
                [
                    flipped_proprioceptive_obs.view(
                        -1, self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length
                    ).detach(),
                    flipped_obs_mask.detach(),
                    flipped_terrain_obs.detach(),
                ],
                dim=-1,
            )

        else:
            return torch.cat(
                [
                    flipped_proprioceptive_obs.view(
                        -1, self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length
                    ),
                    flipped_obs_mask,
                ],
                dim=-1,
            ).detach()

    def flip_g1_critic_obs(self, critic_obs):
        proprioceptive_obs = torch.clone(
            critic_obs[:, : self.actor_critic.num_one_step_critic_obs * self.actor_critic.critic_history_length]
        )
        proprioceptive_obs = proprioceptive_obs.view(
            -1, self.actor_critic.critic_history_length, self.actor_critic.num_one_step_critic_obs
        )

        critic_obs_mask = torch.clone(
            critic_obs[
                :,
                self.actor_critic.num_one_step_critic_obs
                * self.actor_critic.critic_history_length : self.actor_critic.num_one_step_critic_obs
                * self.actor_critic.critic_history_length
                + self.actor_critic.critic_history_length * len(self.actor_critic.one_step_privileged_obs_dims),
            ]
        )
        flipped_critic_obs_mask = critic_obs_mask  # TODO: flip obs_mask if needed

        if not self.freq_control:
            flipped_proprioceptive_obs = (
                self.flip_vector_fix_waist(proprioceptive_obs)
                if self.fix_waist
                else self.flip_vector_waist(proprioceptive_obs)
            )
        else:
            flipped_proprioceptive_obs = (
                self.flip_vector_fix_waist_freq(proprioceptive_obs)
                if self.fix_waist
                else self.flip_vector_waist_freq(proprioceptive_obs)
            )

        # Note: Privileged observations (wrist_forces, wrist_pos, wrist_virtual_target, base_lin_vel)
        # are now handled in flip_vector_waist/flip_vector_fix_waist methods

        return torch.cat(
            [
                flipped_proprioceptive_obs.view(
                    -1, self.actor_critic.num_one_step_critic_obs * self.actor_critic.critic_history_length
                ),
                flipped_critic_obs_mask,
            ],
            dim=-1,
        ).detach()

    def flip_g1_actions(self, actions):
        flipped_actions = torch.zeros_like(actions)
        flipped_actions[:, 0] = actions[:, 6]  # 0 "left_hip_pitch_joint",
        flipped_actions[:, 1] = -actions[:, 7]  # 1 "left_hip_roll_joint",
        flipped_actions[:, 2] = -actions[:, 8]  # 2 "left_hip_yaw_joint",
        flipped_actions[:, 3] = actions[:, 9]  # 3 "left_knee_joint",
        flipped_actions[:, 4] = actions[:, 10]  # 4 "left_ankle_pitch_joint",
        flipped_actions[:, 5] = -actions[:, 11]  # 5 "left_ankle_roll_joint",
        flipped_actions[:, 6] = actions[:, 0]  # 6 "right_hip_pitch_joint",
        flipped_actions[:, 7] = -actions[:, 1]  # 7 "right_hip_roll_joint",
        flipped_actions[:, 8] = -actions[:, 2]  # 8 "right_hip_yaw_joint",
        flipped_actions[:, 9] = actions[:, 3]  # 9 "right_knee_joint",
        flipped_actions[:, 10] = actions[:, 4]  # 10 "right_ankle_pitch_joint",
        flipped_actions[:, 11] = -actions[:, 5]  # 11 "right_ankle_roll_joint",
        if not self.fix_waist:
            flipped_actions[:, 12] = -actions[:, 12]  # 12 "waist_roll_joint"
            flipped_actions[:, 13] = actions[:, 13]  # 13 "waist_pitch_joint"
            flipped_actions[:, 14] = -actions[:, 14]  # 14 "waist_yaw_joint"
        
        flipped_actions[:, 15] = actions[:, 22]  # 15 "left_shoulder_pitch_joint"
        flipped_actions[:, 16] = -actions[:, 23]  # 16 "left_shoulder_roll_joint"
        flipped_actions[:, 17] = -actions[:, 24]  # 17 "left_shoulder_yaw_joint"
        flipped_actions[:, 18] = actions[:, 25]  # 18 "left_elbow_joint"
        flipped_actions[:, 19] = -actions[:, 26]  # 19 "left_wrist_roll_joint"
        flipped_actions[:, 20] = actions[:, 27]  # 20 "left_wrist_pitch_joint"
        flipped_actions[:, 21] = -actions[:, 28]  # 21 "left_wrist_yaw_joint"
        
        flipped_actions[:, 22] = actions[:, 15]  # 22 "right_shoulder_pitch_joint"
        flipped_actions[:, 23] = -actions[:, 16]  # 23 "right_shoulder_roll_joint"
        flipped_actions[:, 24] = -actions[:, 17]  # 24 "right_shoulder_yaw_joint"
        flipped_actions[:, 25] = actions[:, 18]  # 25 "right_elbow_joint"
        flipped_actions[:, 26] = -actions[:, 19]  # 26 "right_wrist_roll_joint"
        flipped_actions[:, 27] = actions[:, 20]  # 27 "right_wrist_pitch_joint"
        flipped_actions[:, 28] = -actions[:, 21]  # 28 "right_wrist_yaw_joint"
        
        
        return flipped_actions.detach()
