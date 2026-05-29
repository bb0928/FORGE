from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import copy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.terrain import Terrain
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_yaw_quat_from_quat, quat_from_euler_xyz
from legged_gym.utils.helpers import class_to_dict

from ..g1.g1_29dof_ee_config import G1RoughEECfg
from ..g1.g1_29dof_ee_residual_config import G1RoughEEResidualCfg
from .ee_residual_observations import EEResidualObservationsMixin
from .ee_residual_observer import EEResidualObserverMixin
from .ee_residual_rewards import EEResidualRewardsMixin
from .ee_residual_wrist_control import EEResidualWristControlMixin
from .ee_residual_math_utils import (
    OnlineAcc,
    compute_arm_mass_matrix,
    euler_from_quaternion,
    quaternion_from_rotation_matrix,
)
import threading
import time

import matplotlib.pyplot as plt
import pickle as pkl
import inspect

import pytorch_kinematics as pk
from pytorch_kinematics.transforms.transform3d import Transform3d



class LeggedRobotEEResidual(
    EEResidualRewardsMixin,
    EEResidualObservationsMixin,
    EEResidualWristControlMixin,
    EEResidualObserverMixin,
    BaseTask,
):
    @staticmethod
    def _build_obs_group_offsets(dims_dict):
        offsets = {}
        cursor = 0
        for name, size in dims_dict.items():
            offsets[name] = (cursor, cursor + size)
            cursor += size
        return offsets

    def __init__(self, cfg: G1RoughEEResidualCfg, sim_params, physics_engine, sim_device, headless):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False  # disable debug visualization for better headless performance
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.fix_waist = self.cfg.control.fix_waist
        self.freq_control = self.cfg.commands.freq_control
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        self.num_one_step_privileged_obs = self.cfg.env.num_one_step_privileged_obs
        self.actor_history_length = self.cfg.env.num_actor_history
        self.critic_history_length = self.cfg.env.num_critic_history
        self.num_height_dim = self.cfg.env.num_height_dim
        self.actor_proprioceptive_obs_length = self.num_one_step_obs * self.actor_history_length
        self.critic_proprioceptive_obs_length = self.num_one_step_privileged_obs * self.critic_history_length
        self.actor_use_height = True if self.num_obs > self.actor_proprioceptive_obs_length else False
        self.num_lower_dof = self.cfg.env.num_lower_actions
        self.num_upper_dof = self.cfg.env.num_upper_actions
        self.actuation_num_action = self.num_lower_dof if self.cfg.model_type == "base" else self.num_lower_dof + self.num_upper_dof
        self.one_step_obs_dims = self.cfg.env.one_step_obs_dims
        self.one_step_privileged_obs_dims = self.cfg.env.one_step_privileged_obs_dims
        self.actor_obs_group_names = list(self.one_step_obs_dims.keys())
        self.actor_obs_group_index = {name: idx for idx, name in enumerate(self.actor_obs_group_names)}
        self.actor_obs_group_offsets = self._build_obs_group_offsets(self.one_step_obs_dims)
        self.privileged_obs_group_names = list(self.one_step_privileged_obs_dims.keys())
        self.privileged_obs_group_index = {name: idx for idx, name in enumerate(self.privileged_obs_group_names)}
        self.privileged_obs_group_offsets = self._build_obs_group_offsets(self.one_step_privileged_obs_dims)
        
        self.base_model_params = self.cfg.env.base_model_env # class_to_dict(self.cfg.env.base_model_env, depth=1)
        self.base_model_pretrained_path = self.base_model_params.model_path
        
        # sanity checks
        assert self.num_one_step_obs == sum([v for v in self.one_step_obs_dims.values()])
        assert self.num_one_step_privileged_obs == sum([v for v in self.one_step_privileged_obs_dims.values()])
        if self.fix_waist:
            self.lower_dof_indices = torch.arange(0, self.num_lower_dof, device=self.device, dtype=torch.long)
            self.upper_dof_indices = torch.arange(
                self.num_lower_dof, self.cfg.env.num_dofs, device=self.device, dtype=torch.long
            )
            assert (
                self.cfg.env.num_actions == 12 and self.cfg.env.num_dofs == 27
            ), "Fixing waist is only supported for 12 lower body actions and 27 total dofs"
        else:
            self.lower_dof_indices = torch.cat(
                (
                    torch.arange(0, 12, device=self.device, dtype=torch.long),
                    torch.arange(13, 15, device=self.device, dtype=torch.long),
                )
            )
            self.upper_dof_indices = torch.cat(
                (
                    torch.arange(12, 13, device=self.device, dtype=torch.long),
                    torch.arange(15, self.cfg.env.num_dofs, device=self.device, dtype=torch.long),
                )
            )
            #     self.cfg.env.num_actions == 14 and self.cfg.env.num_dofs == 29
            # ), "Fixing waist is only supported for 14 (lower body actions + 2 dof waist actions) and 29 total dofs"
        # Per-joint curriculum cap: non-wrist upper joints use max_upper_ratio, wrist joints use max_wrist_ratio
        non_wrist_cap = getattr(self.cfg.domain_rand, "max_upper_ratio", 0.8)
        wrist_cap = getattr(self.cfg.domain_rand, "max_wrist_ratio", 1.0)
        self._upper_curriculum_cap = torch.full((len(self.upper_dof_indices),), non_wrist_cap, device=self.device)
        for i, dof_idx in enumerate(self.upper_dof_indices):
            if "wrist" in self.dof_names[dof_idx]:
                self._upper_curriculum_cap[i] = wrist_cap

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        # Determine wrist force command slice before allocating buffers
        self._calculate_force_command_indices()
        self._calculate_wrist_quat_command_indices()
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        self.current_iteration = 0  # initialize global training-iteration counter
        self.left_acc_lin_vel = OnlineAcc(self.dt, self.dt * 2).to(self.device)
        self.left_acc_ang_vel = OnlineAcc(self.dt, self.dt * 2).to(self.device)
        self.right_acc_lin_vel = OnlineAcc(self.dt, self.dt * 2).to(self.device)
        self.right_acc_ang_vel = OnlineAcc(self.dt, self.dt * 2).to(self.device)

        # Initialize force control parameters
        if hasattr(self.cfg.control, 'wrist_force_kp'):
            self.wrist_force_kp = self.cfg.control.wrist_force_kp
        else:
            self.wrist_force_kp = 200.0  # Default value


    def set_iteration(self, iteration: int):
        """Set the current global training iteration."""
        self.current_iteration = iteration


    def _calculate_force_command_indices(self):
        """Identify the slice for the wrist-force command block."""
        self.wrist_force_command_slice = None
        self.left_wrist_force_command_slice = None
        self.right_wrist_force_command_slice = None
        self.wrist_force_command_half = None

        cursor = 0
        for key, length in self.cfg.commands.commands_dim_len_dict.items():
            if key == "wrist_force_command":
                if length < 6:
                    raise ValueError("wrist_force_command must provide at least 6 dims")
                start = cursor
                end = cursor + length
                self.wrist_force_command_slice = slice(start, end)
                # assume first half corresponds to left wrist, next half to right
                half = length // 2
                self.wrist_force_command_half = half
                self.left_wrist_force_command_slice = slice(start, start + half)
                self.right_wrist_force_command_slice = slice(start + half, end)
                break
            cursor += length

    def _calculate_wrist_quat_command_indices(self):
        """Identify the slice for the wrist quaternion targets block."""
        self.wrist_quat_command_slice = None
        cursor = 0
        for key, length in self.cfg.commands.commands_dim_len_dict.items():
            if key == "wrist_quat":
                start = cursor
                end = cursor + length
                self.wrist_quat_command_slice = slice(start, end)
                break
            cursor += length



    def step(self, actions, force_residual_pred=None):

        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor or tuple/list): Tensor of shape (num_envs, num_actions_per_env)
                OR tuple/list: (policy_actions, force_residual_pred)
            force_residual_pred (torch.Tensor, optional): Force residual prediction in Base-Yaw Frame [num_envs, 6]
                If actions is a tuple/list, this parameter is ignored and extracted from actions instead.
        """
        
        # Handle tuple/list input: (policy_actions, force_residual_pred)
        if isinstance(actions, (tuple, list)) and len(actions) == 2:
            actions, force_residual_pred = actions[0], actions[1]
        
        # Store force residual prediction (in Base-Yaw Frame) for use in control loop
        # Model outputs scaled values (same scale as teacher target), need to unscale for use
        if force_residual_pred is not None:
            # Ensure shape matches [num_envs, 6] (left+right wrist forces)
            if force_residual_pred.shape == (self.num_envs, 6):
                # Unscale: model output is scaled by obs_scales.wrist_force (0.01), divide to get real force
                self.last_force_correction[:] = force_residual_pred / self.obs_scales.wrist_force
            else:
                # If shape doesn't match, zero it (should not happen in normal operation)
                self.last_force_correction[:] = 0.0
        else:
            # If no prediction provided, use zero correction (fallback behavior)
            self.last_force_correction[:] = 0.0

        # !!! WARNING: in fix waist == False case, # actions = 14
        # !!! WARNING: [..., 12: waist yaw, 13: waist roll, 14: waist pitch, ...], where yaw is controlled by the user input, and roll & pitch are controlled by the policy. MUST be very careful with the order of actions!!!
        clip_actions = self.cfg.normalization.clip_actions

        # Update current_upper_actions for all environments (both policy and non-policy mode)
        # Only generate random_upper_actions at interval steps to avoid unnecessary computation
        if self.common_step_counter % self.cfg.domain_rand.upper_interval == 0:
            # Generate new random upper actions only when needed (at interval steps)
            # This is the ONLY data producer for all environments
            self._generate_random_upper_actions()
            
            # Calculate delta for smooth interpolation over the interval
            # IMPORTANT: Calculate delta for ALL environments to ensure current_upper_actions
            # updates smoothly for policy mode (used for FK targets), even though only
            # non-policy mode environments use it directly for execution
            self.delta_upper_actions = (
                self.random_upper_actions - self.current_upper_actions
            ) / self.cfg.domain_rand.upper_interval
        
        # Apply smooth interpolation every step for ALL environments
        # This ensures current_upper_actions is updated for policy mode environments
        # (needed for FK targets in observations), even though execution uses policy output
        self.current_upper_actions = self.current_upper_actions + self.delta_upper_actions
        # Compute FK targets when interval elapses (reuses common_step_counter)
        # The upper actions remain smooth; we just throttle wrist_pos_target updates
        if self.common_step_counter % self.cfg.domain_rand.wrist_target_interval == 0:
            self._assign_wrist_targets_from_fk(torch.arange(self.num_envs, device=self.device), self.current_upper_actions)

        concat_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        concat_actions[:, self.lower_dof_indices] = (
            actions if self.cfg.model_type == "base" else actions[..., : len(self.lower_dof_indices)]
        )
        if self.cfg.model_type == "base":
            upper_values = self.current_upper_actions
        else:
            policy_upper = actions[..., -len(self.upper_dof_indices):]
            upper_values = policy_upper
        concat_actions[:, self.upper_dof_indices] = upper_values
        actions = concat_actions

        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.origin_actions[:] = self.actions[:]
        self.delayed_actions = (
            self.actions.clone().view(1, self.num_envs, self.num_actions).repeat(self.cfg.control.decimation, 1, 1)
        )
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)

        # Randomize Joint Injections
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(
                self.cfg.domain_rand.joint_injection_range[0],
                self.cfg.domain_rand.joint_injection_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            ) * self.torque_limits.unsqueeze(0)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):

            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # upper-body with position control; lower-body with force control;
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            # Apply non-contact wrist forces in world space (mirrors b2z1 behaviour)
            self.gym.apply_rigid_body_force_tensors(
                self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.GLOBAL_SPACE
            )
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        termination_ids, termination_priveleged_obs = self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        # Pass adaptation supervision through infos
        if hasattr(self, 'observer_r_wrist'):
            self.extras["observer_r_wrist"] = self.observer_r_wrist.clone()
        else:
            self.extras["observer_r_wrist"] = torch.zeros(
                self.num_envs, 6, dtype=torch.float, device=self.device
            )
        self.extras["wrist_force_active"] = (
            torch.norm(self.wrist_forces.reshape(self.num_envs, -1), dim=1, keepdim=True) > 1e-6
        ).to(dtype=torch.float)
        return (
            torch.cat([self.obs_buf, self.obs_mask_buf], dim=-1) if hasattr(self, "obs_mask_buf") else self.obs_buf,
            (
                torch.cat([self.privileged_obs_buf, self.privileged_obs_mask_buf], dim=-1)
                if hasattr(self, "privileged_obs_mask_buf")
                else self.privileged_obs_buf
            ),
            self.rew_buf,
            self.reset_buf,
            self.extras,
            termination_ids,
            termination_priveleged_obs,
        )

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # Refresh DOF force tensor to get updated measured joint torques
        self.gym.refresh_dof_force_tensor(self.sim)
        # Refresh mass matrix tensor to get updated M(q) for current configuration
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Push wrist forces (similar to b2z1's push_gripper)
        if self.current_iteration >= self.cfg.commands.force_start_step:
            self._push_wrist(torch.arange(self.num_envs, device=self.device))

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.torso_quat = self.rigid_body_states[:, self.torso_body_index, 3:7]
        relative_quat = quat_mul(
            quat_conjugate(get_yaw_quat_from_quat(self.base_quat)), self.torso_quat
        )  # handle torso rotation
        self.roll, self.pitch, self.yaw = euler_from_quaternion(relative_quat)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.torso_ang_vel[:] = quat_rotate_inverse(
            self.rigid_body_states[:, self.torso_imu_index, 3:7], self.rigid_body_states[:, self.torso_imu_index, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        self.feet_pos[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 3:7]
        self.feet_vel[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self.wrist_pos[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.wrist_indices, 0:3]
        self.wrist_quat[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.wrist_indices, 3:7]
        self.wrist_vel[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.wrist_indices, 7:10]

        self.head_pos[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.head_index, 0:3]
        self.head_quat[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.head_index, 3:7]
        self.head_vel[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.head_index, 7:10]

        # Update wrist force-sensor readings (privileged teacher keeps GT)
        # Note: self.forces has shape [num_envs, num_bodies, 3]
        # self.wrist_indices are indices of wrist rigid bodies
        self.wrist_forces[:] = self.forces[:, self.wrist_indices, :]

        # Update observer first to get F_hat for control feedback; keep GT forces for teacher
        if hasattr(self.cfg, 'observer') and self.cfg.observer.enable:
            self._update_momentum_observer()

        # Update virtual targets: prefer observer force estimate, fallback to GT
        self._update_wrist_virtual_targets()

        # compute contact related quantities
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        self.first_contacts = (self.feet_air_time >= self.dt) * self.contact_filt
        self.feet_air_time += self.dt
        feet_height, feet_height_var, feet_height_raw = self._get_feet_heights()

        self.feet_max_height = torch.maximum(self.feet_max_height, feet_height)

        # compute joint power
        joint_power = torch.abs(self.torques * self.dof_vel).unsqueeze(1)
        self.joint_powers = torch.cat((self.joint_powers[:, 1:], joint_power), dim=1)

        self._post_physics_step_callback()

        # Log the current threshold value to extras for Tensorboard visualization
        self.extras["wrist_tracking_threshold"] = self.wrist_tracking_kill_threshold

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        termination_privileged_obs = (
            torch.cat([termination_privileged_obs, self.privileged_obs_mask_buf[env_ids]], dim=-1)
            if hasattr(self, "privileged_obs_mask_buf")
            else termination_privileged_obs
        )  # append privileged obs mask before reset
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)



        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # reset contact related quantities
        self.feet_air_time *= ~self.contact_filt
        self.feet_max_height *= ~self.contact_filt

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, termination_privileged_obs


    def check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 10.0, dim=1
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.gravity_termination_buf = torch.any(
            torch.norm(self.projected_gravity[:, 0:2], dim=-1, keepdim=True) > 0.8, dim=1
        )

        _, _, feet_height_raw = self._get_feet_heights()


        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.gravity_termination_buf
        self.reset_buf |= torch.any(feet_height_raw < -0.15, dim=1)
        
        # Population-Based Dynamic Termination Curriculum for wrist tracking
        # Compute current wrist tracking error (sum of left and right L2 norms)
        left_wrist_pos = self.rigid_body_states[:, self.left_wrist_indices, :3].clone().mean(dim=1)
        right_wrist_pos = self.rigid_body_states[:, self.right_wrist_indices, :3].clone().mean(dim=1)
        current_wrist_error = torch.norm(left_wrist_pos - self.wrist_pos_target[:, :3], dim=1) + torch.norm(
            right_wrist_pos - self.wrist_pos_target[:, 3:6], dim=1
        )
        
        # Population-Based Dynamic Termination Curriculum (only after start_iter)
        if self.current_iteration >= self.wrist_termination_start_iter:
            is_wrist_mode = (self.is_pos_ctrl_mode | self.is_force_ctrl_mode).squeeze(1)
            warmup_passed = self.episode_length_buf > 300
            
            # Compute height-tracking error: only check wrist tracking after height is near target
            # Reason: wrist target is computed from target torso height; if robot has not reached it yet,
            # wrist error may be caused by height mismatch and should not trigger termination
            current_base_height = self.root_states[:, 2]  # current base height
            target_base_height = self.commands[:, 4]  # target base height
            height_error = torch.abs(current_base_height - target_base_height)
            height_converged = height_error < 0.1  # consider converged when height error is below 10 cm
            
            wrist_tracking_termination = (
                (current_wrist_error > self.wrist_tracking_kill_threshold)
                & is_wrist_mode
                & warmup_passed
                & height_converged  # only check wrist tracking when height is near target
            )
            self.reset_buf |= wrist_tracking_termination

    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # update terrain curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)
        # update action curriculum for specific dofs
        if self.cfg.env.action_curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_action_curriculum(env_ids)

        self.refresh_actor_rigid_shape_props(env_ids)

        # reset robot states
        self._reset_root_states(env_ids)
        self._reset_dofs(env_ids)

        # resample commands (this will also set upper_policy_mask based on is_wrist_pos)
        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_tau[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.joint_powers[env_ids] = 0.0
        self.random_upper_actions[env_ids] = 0.0
        self.current_upper_actions[env_ids] = 0.0
        self.delta_upper_actions[env_ids] = 0.0
        self.wrist_joint_target[env_ids] = self.default_dof_pos[0, self.wrist_joint_dof_indices]
        # Note: upper_policy_mask is set in _resample_commands based on mode
        # Do not override it here, as it should match the current command mode
        # is_wrist_force is already set in _resample_commands; do not overwrite here
        reset_roll, reset_pitch, reset_yaw = euler_from_quaternion(self.base_quat[env_ids])
        self.roll[env_ids] = reset_roll
        self.pitch[env_ids] = reset_pitch
        self.yaw[env_ids] = reset_yaw
        self.reset_buf[env_ids] = 1
        self.base_err_int[env_ids, :] = 0.0

        if self.obs_delay_feet_enable and self.obs_delay_feet_max > 0:
            self.feet_cmd_history[env_ids] = 0.0

        self.last_heights[env_ids, :] = (
            self.cfg.rewards.base_height_target - 1.0
        ) * self.obs_scales.height_measurements

        # Reset force command buffers (similar to b2z1)
        if hasattr(self.cfg.commands, 'push_wrist_stators') and self.cfg.commands.push_wrist_stators:
            self.current_Fxyz_wrist_cmd[env_ids, :] = 0.0
            self.force_target_wrist_cmd[env_ids, :] = 0.0
            self.push_end_time_wrist_cmd[env_ids] = 0.
            self.push_duration_wrist_cmd[env_ids] = 0.
            self.selected_env_ids_wrist_cmd[env_ids] = 0
            self.freed_envs_wrist_cmd[env_ids] = False
            self.force_target_wrist_ext[env_ids, :] = 0.0
            self.push_end_time_wrist_ext[env_ids] = 0.
            self.push_duration_wrist_ext[env_ids] = 0.
            self.selected_env_ids_wrist_ext[env_ids] = 0
            self.freed_envs_wrist_ext[env_ids] = False
            self.forces[env_ids, self.left_wrist_handle, :3] = 0.0
            self.forces[env_ids, self.right_wrist_handle, :3] = 0.0
            
            # Reset force commands in commands array
            if self.wrist_force_command_slice is not None:
                self.commands[env_ids, self.wrist_force_command_slice] = 0.0
                
            # Reset force control state for reward calculation
            if hasattr(self, 'left_wrist_forces_total'):
                self.left_wrist_forces_total[env_ids] = 0.
            if hasattr(self, 'right_wrist_forces_total'):
                self.right_wrist_forces_total[env_ids] = 0.
            if hasattr(self, 'forces_cmd_global'):
                self.forces_cmd_global[env_ids] = 0.
            if hasattr(self, 'last_force_correction'):
                self.last_force_correction[env_ids] = 0.
        
        # Sample wrist z offset once per episode for smoothness
        # Only sample for environments that will use wrist_pos mode
        # Note: _resample_commands has been called above, so is_wrist_pos is already set
        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        # Removed wrist_z_offset sampling logic: no longer need implicit dz-based target adjustment
        # Use explicit Height Command directly; wrist_z_offset stays zero
        self.wrist_z_offset[env_ids_tensor] = 0.0
        
        # Randomize force gains for reset environments
        if self.cfg.commands.randomize_wrist_force_gains:
                self.wrist_force_kps[env_ids, :] = torch_rand_float(
                    self.cfg.commands.wrist_force_kp_range[0],
                    self.cfg.commands.wrist_force_kp_range[1],
                    (len(env_ids), 3),
                    device=self.device
                )
                self.wrist_force_kds[env_ids, :] = self.wrist_force_kps[env_ids, :] * self.cfg.commands.wrist_prop_kd

        # reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(
                self.cfg.domain_rand.kp_range[0],
                self.cfg.domain_rand.kp_range[1],
                (len(env_ids), self.num_actions),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(
                self.cfg.domain_rand.kd_range[0],
                self.cfg.domain_rand.kd_range[1],
                (len(env_ids), self.num_actions),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(
                self.cfg.domain_rand.actuation_offset_range[0],
                self.cfg.domain_rand.actuation_offset_range[1],
                (len(env_ids), self.num_dof),
                device=self.device,
            ) * self.torque_limits.unsqueeze(0)

        if self.add_noise:
            self.height_noise_offset[env_ids] = torch_rand_float(
                self.cfg.noise.height_measurements.offset_range[0],
                self.cfg.noise.height_measurements.offset_range[1],
                (len(env_ids), 1),
                device=self.device,
            )

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(
                self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt
            )
            self.episode_sums[key][env_ids] = 0.0
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            # self.extras["episode"]["height_curriculum_ratio"] = self.height_curriculum_ratio
        if self.cfg.env.action_curriculum:
            self.extras["episode"]["action_curriculum_ratio"] = self.action_curriculum_ratio
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
        self.gait_indices[env_ids] = 0

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if torch.isnan(rew).any():
                import ipdb

                ipdb.set_trace()
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

            # self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)


    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def create_cameras(self):
        """Creates camera for each robot"""
        self.camera_params = gymapi.CameraProperties()
        self.camera_params.width = self.cfg.camera.width
        self.camera_params.height = self.cfg.camera.height
        self.camera_params.horizontal_fov = self.cfg.camera.horizontal_fov
        self.camera_params.enable_tensors = True
        self.cameras = []
        for env_handle in self.envs:
            camera_handle = self.gym.create_camera_sensor(env_handle, self.camera_params)
            torso_handle = self.gym.get_actor_rigid_body_handle(env_handle, 0, self.torso_index)
            camera_offset = gymapi.Vec3(self.cfg.camera.offset[0], self.cfg.camera.offset[1], self.cfg.camera.offset[2])
            camera_rotation = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 1, 0),
                np.deg2rad(self.cfg.camera.angle_randomization * (2 * np.random.random() - 1) + self.cfg.camera.angle),
            )
            self.gym.attach_camera_to_body(
                camera_handle,
                env_handle,
                torso_handle,
                gymapi.Transform(camera_offset, camera_rotation),
                gymapi.FOLLOW_TRANSFORM,
            )
            self.cameras.append(camera_handle)


    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(
                    friction_range[0], friction_range[1], (self.num_envs, 1), device=self.device
                )

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id == 0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(
                    restitution_range[0], restitution_range[1], (self.num_envs, 1), device=self.device
                )

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props

    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(
                self.cfg.domain_rand.friction_range[0],
                self.cfg.domain_rand.friction_range[1],
                (len(env_ids), 1),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(
                self.cfg.domain_rand.restitution_range[0],
                self.cfg.domain_rand.restitution_range[1],
                (len(env_ids), 1),
                device=self.device,
            )

        for env_id in env_ids:
            env_handle = self.envs[env_id]
            actor_handle = self.actor_handles[env_id]
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)

            for i in range(len(rigid_shape_props)):
                if self.cfg.domain_rand.randomize_friction:
                    rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                if self.cfg.domain_rand.randomize_restitution:
                    rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.hard_dof_pos_limits = torch.zeros(
                self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id == 0:
            sum = 0
            for i, p in enumerate(props):
                sum += p.mass
                print(f"Mass of body {i}: {p.mass} (before randomization)")
            print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[self.torso_body_index].mass = (
                self.default_rigid_body_mass[self.torso_body_index] + self.payload[env_id, 0]
            )
            props[self.left_hand_index].mass = (
                self.default_rigid_body_mass[self.left_hand_index] + self.hand_payload[env_id, 0]
            )
            props[self.right_hand_index].mass = (
                self.default_rigid_body_mass[self.right_hand_index] + self.hand_payload[env_id, 1]
            )

            if self.backpack_payload is not None:
                props[self.backpack_index].mass = (
                    self.default_rigid_body_mass[self.backpack_index] + self.backpack_payload[env_id, 0]
                )

        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = self.default_com + gymapi.Vec3(
                self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2]
            )
        if self.cfg.domain_rand.randomize_body_displacement:
            props[self.torso_body_index].com = self.default_body_com + gymapi.Vec3(
                self.body_displacement[env_id, 0], self.body_displacement[env_id, 1], self.body_displacement[env_id, 2]
            )

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (
            (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        # TODO backup the rotation before resampling
        self.root_pos_before_reset[env_ids, :3] = self.root_states[env_ids, :3].clone()
        self._resample_commands(env_ids)
        if self.freq_control:
            self._step_contact_targets()
        
        # Compute ang vel command from heading error when heading_command is True
        # Only compute if commands[2] is zero (not set by user/teleop)
        # For vel tracking mode: compute heading from linear velocity direction to make robot walk straight
        if self.cfg.commands.heading_command:
            # Check if commands[2] is zero (not set by user)
            mask = torch.abs(self.commands[:, 2]) < 1e-6
            
            if torch.any(mask):
                # Get current yaw angle
                roll, pitch, yaw = euler_from_quaternion(self.base_quat)
                yaw = yaw.squeeze(-1)  # [num_envs]
                
                # For vel tracking: compute desired heading from linear velocity direction
                # This makes robot walk straight in the direction of linear velocity
                lin_vel_norm = torch.norm(self.commands[:, :2], dim=1)
                vel_tracking_mask = (lin_vel_norm > 0.1) & mask  # Only for vel tracking mode
                
                # Compute desired heading: use linear velocity direction for vel tracking, otherwise use commands[3]
                desired_heading = torch.zeros_like(yaw)
                if torch.any(vel_tracking_mask):
                    # For vel tracking: heading = atan2(vy, vx)
                    desired_heading[vel_tracking_mask] = torch.atan2(
                        self.commands[vel_tracking_mask, 1], 
                        self.commands[vel_tracking_mask, 0]
                    )
                if torch.any(~vel_tracking_mask & mask):
                    # For non-vel tracking: use commands[3] (randomly sampled heading)
                    desired_heading[~vel_tracking_mask & mask] = self.commands[~vel_tracking_mask & mask, 3]
                
                # Compute heading error (normalize to [-pi, pi])
                heading_error = desired_heading - yaw
                heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
                
                # Compute desired angular velocity using proportional control
                heading_kp = 2.0  # Proportional gain for heading control
                ang_vel_from_heading = heading_kp * heading_error
                
                # Clamp to ang_vel_yaw range
                ang_vel_from_heading = torch.clamp(
                    ang_vel_from_heading,
                    self.command_ranges["ang_vel_yaw"][0],
                    self.command_ranges["ang_vel_yaw"][1]
                )
                
                # Only update commands[2] for environments where it's zero
                self.commands[mask, 2] = ang_vel_from_heading[mask]

        if self.cfg.terrain.measure_heights and self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.measured_heights = self._get_heights()

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()


        


    def discretize_speed(self, speeds: torch.Tensor, K: float) -> torch.Tensor:
        half = K * 0.5
        return torch.floor((speeds + half) / K) * K



    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        set_x = torch.rand(len(env_ids), 1).to(self.device)
        is_head_pos = False
        # Three-mode mutually exclusive sampling: 50 / 25 / 25
        is_walking    = set_x < 0.50
        is_pos_ctrl   = (set_x >= 0.50) & (set_x < 0.75)
        is_force_ctrl = set_x >= 0.75

        # If push_wrist_stators=False, force-control infra is unavailable; downgrade 25% branch into position-control
        if not (hasattr(self.cfg.commands, 'push_wrist_stators') and self.cfg.commands.push_wrist_stators):
            is_pos_ctrl   = is_pos_ctrl | is_force_ctrl
            is_force_ctrl = torch.zeros_like(is_force_ctrl)

        is_operation = is_pos_ctrl | is_force_ctrl

        self.is_walking_mode[env_ids]    = is_walking
        self.is_pos_ctrl_mode[env_ids]   = is_pos_ctrl
        self.is_force_ctrl_mode[env_ids] = is_force_ctrl
        self.is_feet_pos[env_ids]        = False
        self.is_head_pos[env_ids]        = False
        self.upper_policy_mask[env_ids, 0] = True

        # Force command: sample only in force-control mode (_push_wrist physical force is still gated by push_wrist_stators)
        if self.wrist_force_command_slice is not None:
            force_cmd_range = self.cfg.commands.max_push_force_xyz_wrist_cmd
            total_dims = (self.wrist_force_command_half or 3) * 2
            rand_force = torch_rand_float(
                force_cmd_range[0], force_cmd_range[1], (len(env_ids), total_dims), device=self.device
            )
            self.commands[env_ids, self.wrist_force_command_slice] = \
                rand_force * is_force_ctrl.float().expand(-1, total_dims)

        
        self.commands[env_ids, 0] = (
            torch_rand_float(
                self.command_ranges["lin_vel_x"][0],
                self.command_ranges["lin_vel_x"][1],
                (len(env_ids), 1),
                device=self.device,
            )
            * is_walking
        ).squeeze(1)
        self.command_ratio[env_ids, 0] = (self.commands[env_ids, 0] / self.command_ranges["lin_vel_x"][1]) * (
            self.commands[env_ids, 0] > 0
        ) + (self.commands[env_ids, 0] / self.command_ranges["lin_vel_x"][0]) * (self.commands[env_ids, 0] < 0)
        self.commands[env_ids, 1] = (
            torch_rand_float(
                self.command_ranges["lin_vel_y"][0],
                self.command_ranges["lin_vel_y"][1],
                (len(env_ids), 1),
                device=self.device,
            )
            * (is_walking * (self.commands[env_ids, 0] < 1.2).unsqueeze(1))  # ignore y vel when x vel is large
        ).squeeze(1)
        self.command_ratio[env_ids, 1] = (self.commands[env_ids, 1] / self.command_ranges["lin_vel_y"][1]) * (
            self.commands[env_ids, 1] > 0
        ) + (self.commands[env_ids, 1] / self.command_ranges["lin_vel_y"][0]) * (self.commands[env_ids, 1] < 0)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = (
                torch_rand_float(
                    self.command_ranges["heading"][0],
                    self.command_ranges["heading"][1],
                    (len(env_ids), 1),
                    device=self.device,
                )
                * is_walking
            ).squeeze(1)
            self.command_ratio[env_ids, 3] = (self.commands[env_ids, 3] / self.command_ranges["heading"][1]) * (
                self.commands[env_ids, 3] > 0
            ) + (self.commands[env_ids, 3] / self.command_ranges["heading"][0]) * (self.commands[env_ids, 3] < 0)
            # Height command sampling: random in operation mode, fixed to base_height_target in walking mode (error naturally 0)
            self.commands[env_ids, 4] = (
                torch_rand_float(
                    self.command_ranges["height"][0],
                    self.command_ranges["height"][1],
                    (len(env_ids), 1),
                    device=self.device,
                )
                * is_operation
            ).squeeze(
                1
            ) + self.cfg.rewards.base_height_target  # height
            self.command_ratio[env_ids, 4] = (self.commands[env_ids, 4] / self.command_ranges["height"][1]) * (
                self.commands[env_ids, 4] > 0
            ) + (self.commands[env_ids, 4] / self.command_ranges["height"][0]) * (self.commands[env_ids, 4] < 0)
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)
        else:
            self.commands[env_ids, 2] = (
                torch_rand_float(
                    self.command_ranges["ang_vel_yaw"][0],
                    self.command_ranges["ang_vel_yaw"][1],
                    (len(env_ids), 1),
                    device=self.device,
                )
                * is_walking
            ).squeeze(1)
            self.command_ratio[env_ids, 2] = (self.commands[env_ids, 2] / self.command_ranges["ang_vel_yaw"][1]) * (
                self.commands[env_ids, 2] > 0
            ) + (self.commands[env_ids, 2] / self.command_ranges["ang_vel_yaw"][0]) * (self.commands[env_ids, 2] < 0)
            # Height command sampling: random in operation mode, fixed to base_height_target in walking mode (error naturally 0)
            self.commands[env_ids, 4] = (
                torch_rand_float(
                    self.command_ranges["height"][0],
                    self.command_ranges["height"][1],
                    (len(env_ids), 1),
                    device=self.device,
                )
                * is_operation
            ).squeeze(
                1
            ) + self.cfg.rewards.base_height_target  # height
            self.command_ratio[env_ids, 4] = (self.commands[env_ids, 4] / self.command_ranges["height"][1]) * (
                self.commands[env_ids, 4] > 0
            ) + (self.commands[env_ids, 4] / self.command_ranges["height"][0]) * (self.commands[env_ids, 4] < 0)

            self.commands[env_ids, :3] *= (torch.norm(self.commands[env_ids, :3], dim=1) > 0.1).unsqueeze(1)

        if self.freq_control:
            self.commands[env_ids, 5] = torch_rand_float(
                self.command_ranges["frequency"][0],
                self.command_ranges["frequency"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        left_foot_pos = (
            self.rigid_body_states[:, self.left_foot_indices, :3].clone().mean(dim=1)
            - self.root_pos_before_reset[:, :3].clone()
        ) + self.root_states[:, 0:3].clone()
        right_foot_pos = (
            self.rigid_body_states[:, self.right_foot_indices, :3].clone().mean(dim=1)
            - self.root_pos_before_reset[:, :3].clone()
        ) + self.root_states[:, 0:3].clone()
        rand_angle = torch_rand_float(-np.pi, np.pi, (len(env_ids), 2), device=self.device)
        rand_dist = torch_rand_float(
            self.command_ranges["feet_pos"][0],
            self.command_ranges["feet_pos"][1],
            (len(env_ids), 2),
            device=self.device,
        )
        self.feet_pos_target[env_ids] = (
            torch.cat((left_foot_pos[env_ids, :2], right_foot_pos[env_ids, :2]), dim=1)
            + torch.cat((rand_dist * torch.cos(rand_angle), rand_dist * torch.sin(rand_angle)), dim=1)[:, [0, 2, 1, 3]]
            * is_operation                  # random foothold targets only in operation modes (position + force control)
        )
        # is_feet_pos is already set to False by compatibility mapping; no reassignment here

        # wrist_pos_target is updated every step by _assign_wrist_targets_from_fk() in step(),
        # and written into commands in compute_observations as observations; do not initialize here

        # is_head_pos is already set to False by compatibility mapping; head commands are sampled but inactive
        r = torch_rand_float(
            self.command_ranges["head_pos"][0],
            self.command_ranges["head_pos"][1],
            (len(env_ids), 1),
            device=self.device,
        )
        min_theta, max_theta = self.cfg.commands.ranges.head_pos_sample_theta_range
        theta = torch_rand_float(min_theta, max_theta, (len(env_ids), 1), device=self.device)
        phi = torch_rand_float(-np.pi, np.pi, (len(env_ids), 1), device=self.device)
        dx = r * torch.sin(theta) * torch.cos(phi)
        dy = r * torch.sin(theta) * torch.sin(phi)
        dz = r * torch.cos(theta) # r * torch.ones_like(dy, device=dy.device) #
        head_pos = (
            self.rigid_body_states[:, self.head_index, :3].clone() - self.root_pos_before_reset[:, :3].clone()
        ) + self.root_states[:, 0:3].clone()
        self.head_pos_target[env_ids] = head_pos[env_ids] + torch.cat((dx, dy, dz), dim=1) * is_head_pos

        # Always keep mutual exclusion logic; do not auto-toggle flags by iteration count

    def _compute_torques(self, actions):
        """Compute torques from actions.    
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        self.joint_pos_target = self.default_dof_pos + actions_scaled
        control_type = self.cfg.control.control_type
        if control_type == "P":
            return

        elif control_type == "M":
            # Stable PD discretization, pure torque output, bypass built-in PD
            dt = self.sim_params.dt
            kp = self.p_gains * self.Kp_factors
            kd = self.d_gains * self.Kd_factors
            denom = 1.0 + dt * kd + dt * dt * kp
            kp_hat = kp / denom
            kd_hat = (kd + dt * kp) / denom
            kp_hat = kp
            kd_hat = kd

            tau = kp_hat * (self.joint_pos_target - self.dof_pos) - kd_hat * self.dof_vel
            tau = tau + self.actuation_offset + self.joint_injection
            tau = torch.clip(tau, -self.torque_limits, self.torque_limits)
            
            # Apply EMA smoothing to tau
            tau_ema_alpha = self.cfg.control.tau_ema_alpha
            tau = tau_ema_alpha * self.last_tau + (1.0 - tau_ema_alpha) * tau
            self.last_tau[:] = tau[:]
            
            control = torch.zeros_like(tau)
            control[..., self.lower_dof_indices] = tau[..., self.lower_dof_indices]
            control[..., self.upper_dof_indices] = tau[..., self.upper_dof_indices]
            return control


        else:
            raise NameError(f"Unknown controller type: {control_type}")


    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
        if self.cfg.domain_rand.randomize_initial_joint_pos:
            init_dos_pos = self.default_dof_pos * torch_rand_float(
                self.cfg.domain_rand.initial_joint_pos_scale[0],
                self.cfg.domain_rand.initial_joint_pos_scale[1],
                (len(env_ids), self.num_dof),
                device=self.device,
            )
            init_dos_pos += torch_rand_float(
                self.cfg.domain_rand.initial_joint_pos_offset[0],
                self.cfg.domain_rand.initial_joint_pos_offset[1],
                (len(env_ids), self.num_dof),
                device=self.device,
            )
            self.dof_pos[env_ids] = torch.clip(init_dos_pos, dof_lower, dof_upper)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch.ones((len(env_ids), self.num_dof), device=self.device)

        self.dof_vel[env_ids] = 0.0

        if hasattr(self.cfg, 'observer') and self.cfg.observer.enable:
            if hasattr(self, 'observer_r'):
                self.observer_r[env_ids] = 0.0
            if hasattr(self, 'observer_p_hat'):
                self.observer_p_hat[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _update_terrain_curriculum(self, env_ids):
        """Implements the game-inspired terrain curriculum.
        Robots that travel far enough progress to harder terrains.
        """
        if not self.init_done:
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)

        if self.cfg.terrain.hard_terrain:
            move_up_by_one = (distance >= self.terrain.env_length - 0.5) * ~self.border_buf[env_ids]
        else:
            move_up_by_one = distance >= self.terrain.env_length - 0.5

        self.move_up_counter[env_ids] = (self.move_up_counter[env_ids] + move_up_by_one) * move_up_by_one
        move_up = self.move_up_counter[env_ids] >= 3
        self.move_up_counter[env_ids] *= ~move_up

        self.terrain_levels[env_ids] = self.terrain_levels[env_ids] + 1 * move_up
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )

        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.platform_length[env_ids] = self.terrain_platform_length[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        # TODO backup the rotation before reset
        self.root_pos_before_reset[env_ids, :3] = self.root_states[env_ids, :3].clone()
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )  # xy position within 1m of the center
            self.root_states[env_ids, 2:3] += self._get_init_heights(env_ids)
            self.root_states[env_ids, 2:3] += torch_rand_float(0.03, 0.05, (len(env_ids), 1), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )  # lin vel x/y

        if hasattr(self.cfg.domain_rand, "max_push_vel_z") and self.cfg.domain_rand.max_push_vel_z > 0.0:
            max_vel_z = self.cfg.domain_rand.max_push_vel_z
            self.root_states[:, [9]] += torch_rand_float(  # note we use `add` instead of `assign`
                -max_vel_z, max_vel_z, (self.num_envs, 1), device=self.device
            )  # ang vel z
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 75% of the maximum, increase the range of commands
        if (
            torch.mean(self.episode_sums["tracking_x_vel"][env_ids]) / self.max_episode_length
            > 0.8 * self.reward_scales["tracking_x_vel"]
        ) and (
            torch.mean(self.episode_sums["tracking_y_vel"][env_ids]) / self.max_episode_length
            > 0.8 * self.reward_scales["tracking_y_vel"]
        ):
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.0
            )
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.2, 0.0, self.cfg.commands.max_curriculum
            )

    def update_action_curriculum(self, env_ids):
        """Implements a curriculum of increasing action range

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        walk_mask = self.is_walking_mode[env_ids].squeeze(1)
        if walk_mask.any():
            walk_ids = env_ids[walk_mask]
            walk_sums = self.episode_sums["tracking_x_vel"][walk_ids]
            walk_lens = self.episode_length_buf[walk_ids].float().clamp(min=1)
            mean_walk_reward = torch.mean(walk_sums / walk_lens)
            if mean_walk_reward > 0.8 * self.reward_scales["tracking_x_vel"]:
                self.action_curriculum_ratio += 0.05
                self.action_curriculum_ratio = min(self.action_curriculum_ratio, 1.0)



    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # Initialize tracking error logging (for monitoring only, no adaptive adjustment)
        self._init_tracking_error_logging()
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # Acquire DOF force tensor for momentum observer (actual measured joint torques from simulation)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        # Acquire mass matrix tensor for momentum observer (full M(q) matrix)
        # CRITICAL: No fallback allowed - must have mass matrix support
        mass_matrix_tensor = self.gym.acquire_mass_matrix_tensor(self.sim, "g1")
        if mass_matrix_tensor is None:
            raise RuntimeError("Mass matrix tensor not available in this IsaacGym build. "
                             "This version requires mass matrix support for momentum observer.")
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        # Wrap DOF force tensor: actual joint torques/generalized forces from simulation
        # Shape: [num_envs, num_dof] - represents measured joint forces at each DOF
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)
        # Wrap mass matrix tensor: full M(q) configuration-dependent mass matrix
        # Shape: [num_envs, num_dof, num_dof] - full mass matrix for momentum observer
        self.mass_matrix = gymtorch.wrap_tensor(mass_matrix_tensor).view(self.num_envs, self.num_dof, self.num_dof)
        # CRITICAL: Validate mass matrix shape - no fallback allowed
        assert self.mass_matrix.shape == (self.num_envs, self.num_dof, self.num_dof), \
            f"Mass matrix shape mismatch: expected ({self.num_envs}, {self.num_dof}, {self.num_dof}), "\
            f"got {self.mass_matrix.shape}"
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.torso_quat = self.rigid_body_states[:, self.torso_body_index, 3:7]
        relative_quat = quat_mul(quat_conjugate(get_yaw_quat_from_quat(self.base_quat)), self.torso_quat)
        self.roll, self.pitch, self.yaw = euler_from_quaternion(relative_quat)
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        self.wrist_pos = self.rigid_body_states[:, self.wrist_indices, 0:3]
        self.wrist_quat = self.rigid_body_states[:, self.wrist_indices, 3:7]
        self.wrist_vel = self.rigid_body_states[:, self.wrist_indices, 7:10]
        self.head_pos = self.rigid_body_states[:, self.head_index, 0:3]
        self.head_quat = self.rigid_body_states[:, self.head_index, 3:7]
        self.head_vel = self.rigid_body_states[:, self.head_index, 7:10]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.origin_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_tau = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.commands = torch.zeros(
            self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False
        )  # x vel, y vel, yaw vel, heading
        self.command_ratio = torch.zeros(
            self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )  # TODO change this
        self.feet_air_time = torch.zeros(
            self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False
        )
        self.feet_max_height = torch.zeros(
            self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.first_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.torso_ang_vel = quat_rotate_inverse(
            self.rigid_body_states[:, self.torso_imu_index, 3:7], self.rigid_body_states[:, self.torso_imu_index, 10:13]
        )
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # Note: noise_scale_vec requires num_dof, which is not initialized yet; set it after _create_envs
        self.border_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.move_up_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.base_err_int = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # store integral of base position error

        self.last_heights = torch.zeros(
            self.num_envs,
            len(self.cfg.terrain.measured_points_x) * len(self.cfg.terrain.measured_points_y),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_heights[:, :] = (self.cfg.rewards.base_height_target - 1.0) * self.obs_scales.height_measurements

        if self.cfg.terrain.measure_heights and self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            print(
                f"Joint {self.gym.find_actor_dof_index(self.envs[0], self.actor_handles[0], name, gymapi.IndexDomain.DOMAIN_ACTOR)}: {name}"
            )
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.upper_policy_mask = torch.ones(self.num_envs, 1, dtype=torch.bool, device=self.device)
        self.action_max = (
            self.hard_dof_pos_limits[:, 1].unsqueeze(0) - self.default_dof_pos
        ) / self.cfg.control.action_scale
        self.action_min = (
            self.hard_dof_pos_limits[:, 0].unsqueeze(0) - self.default_dof_pos
        ) / self.cfg.control.action_scale
        self.action_curriculum_ratio = self.cfg.domain_rand.init_upper_ratio
        self.target_heights = torch.ones((self.num_envs), device=self.device) * self.cfg.rewards.base_height_target
        print(f"Action min: {self.action_min}")
        print(f"Action max: {self.action_max}")

        self.random_upper_actions = torch.zeros(
            (self.num_envs, self.num_actions - self.num_lower_dof), device=self.device
        )
        self.current_upper_actions = torch.zeros(
            (self.num_envs, self.num_actions - self.num_lower_dof), device=self.device
        )
        self.delta_upper_actions = torch.zeros(
            (self.num_envs, self.num_actions - self.num_lower_dof), device=self.device
        )
        # randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.Kd_factors = torch.ones(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.joint_injection = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actuation_offset = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.height_noise_offset = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )

        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(
                self.cfg.domain_rand.kp_range[0],
                self.cfg.domain_rand.kp_range[1],
                (self.num_envs, self.num_actions),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(
                self.cfg.domain_rand.kd_range[0],
                self.cfg.domain_rand.kd_range[1],
                (self.num_envs, self.num_actions),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(
                self.cfg.domain_rand.joint_injection_range[0],
                self.cfg.domain_rand.joint_injection_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            ) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(
                self.cfg.domain_rand.actuation_offset_range[0],
                self.cfg.domain_rand.actuation_offset_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            ) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(
                self.cfg.domain_rand.payload_mass_range[0],
                self.cfg.domain_rand.payload_mass_range[1],
                (self.num_envs, 1),
                device=self.device,
            )
            self.hand_payload = torch_rand_float(
                self.cfg.domain_rand.hand_payload_mass_range[0],
                self.cfg.domain_rand.hand_payload_mass_range[1],
                (self.num_envs, 2),
                device=self.device,
            )

            if self.backpack_payload is not None:
                self.backpack_payload = torch_rand_float(
                    self.cfg.domain_rand.backpack_payload_mass_range[0],
                    self.cfg.domain_rand.backpack_payload_mass_range[1],
                    (self.num_envs, 1),
                    device=self.device,
                )

        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(
                self.cfg.domain_rand.com_displacement_range[0],
                self.cfg.domain_rand.com_displacement_range[1],
                (self.num_envs, 3),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_body_displacement:
            self.body_displacement = torch_rand_float(
                self.cfg.domain_rand.body_displacement_range[0],
                self.cfg.domain_rand.body_displacement_range[1],
                (self.num_envs, 3),
                device=self.device,
            )

        # store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )
        # External force buffer (used for non-contact wrist pushes similar to b2z1)
        self.forces = torch.zeros(
            self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # Force-control parameters (for PF mode)
        # Wrist force sensor buffers
        self.wrist_forces = torch.zeros(
            self.num_envs, len(self.wrist_indices), 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        # Force-control stiffness coefficient (for force-to-displacement conversion)
        # Reference B2Z1: gripper_force_kp = 200 N/m
        if hasattr(self.cfg.control, 'wrist_force_kp'):
            self.wrist_force_kp = torch.tensor(
                self.cfg.control.wrist_force_kp, dtype=torch.float, device=self.device, requires_grad=False
            )
        else:
            # Default stiffness coefficient 200 N/m (kept consistent with B2Z1)
            # Physical meaning: 200N causes 1m displacement, or 10N causes 5cm displacement
            self.wrist_force_kp = torch.tensor(200.0, dtype=torch.float, device=self.device, requires_grad=False)

        # Bilateral wrist force-control state (used by reward function)
        self.left_wrist_forces_total = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.right_wrist_forces_total = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.forces_cmd_global = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # Force residual correction buffer (for  gap compensation)
        # Stores predicted force residual in Base-Yaw Frame [num_envs, 6] (left+right wrist)
        self.last_force_correction = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )

        # Force command system (similar to b2z1)
        if hasattr(self.cfg.commands, 'push_wrist_stators') and self.cfg.commands.push_wrist_stators:
            # Wrist force command tensors
            num_force_dims = (self.wrist_force_command_half or 3) * 2
            self.current_Fxyz_wrist_cmd = torch.zeros(
                self.num_envs, num_force_dims, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.force_target_wrist_cmd = torch.zeros(
                self.num_envs, num_force_dims, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.push_duration_wrist_cmd = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.push_end_time_wrist_cmd = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.selected_env_ids_wrist_cmd = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
            )
            self.freed_envs_wrist_cmd = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
            )
            
            # External force tensors (for physics simulation)
            # force_target_wrist_ext needs 6 dimensions (3 per wrist)
            self.force_target_wrist_ext = torch.zeros(
                self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.push_duration_wrist_ext = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.push_end_time_wrist_ext = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.selected_env_ids_wrist_ext = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
            )
            self.freed_envs_wrist_ext = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
            )
            
            # Force gains (similar to b2z1's gripper_force_kps)
            self.wrist_force_kps = torch.zeros(
                self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.wrist_force_kds = torch.zeros(
                self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
            )
            
            # Push interval parameters
            self.push_interval_wrist_cmd = torch.zeros(
                self.num_envs, 1, dtype=torch.long, device=self.device, requires_grad=False
            )
            self.push_interval_wrist_ext = torch.zeros(
                self.num_envs, 1, dtype=torch.long, device=self.device, requires_grad=False
            )
            
            # Initialize force intervals and gains
            self._init_force_parameters()

        # joint powers
        self.joint_powers = torch.zeros(
            self.num_envs, 100, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.obs_mask_buf = torch.ones(
            self.num_envs,
            self.actor_history_length * len(self.one_step_obs_dims),
            device=self.device,
            dtype=torch.float,
        )
        self.privileged_obs_mask_buf = (
            torch.ones(
                self.num_envs,
                self.critic_history_length * len(self.one_step_privileged_obs_dims),
                device=self.device,
                dtype=torch.float,
            )
            if self.num_privileged_obs is not None
            else None
        )
        self.root_pos_before_reset = torch.zeros(
            self.num_envs, 7, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.feet_pos_target = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.is_feet_pos = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device, requires_grad=False)

        # Feet-command observation delay ring buffer (sim2real: SLAM latency)
        dr = self.cfg.domain_rand
        self.obs_delay_feet_enable = getattr(dr, 'obs_delay_feet_enable', False)
        _delay_range = getattr(dr, 'obs_delay_feet_range', [0, 0])
        self.obs_delay_feet_min = _delay_range[0]
        self.obs_delay_feet_max = _delay_range[1]
        if self.obs_delay_feet_enable and self.obs_delay_feet_max > 0:
            feet_cmd_dim = self.one_step_obs_dims.get("feet_pose_command", 4)
            self.feet_cmd_history = torch.zeros(
                self.num_envs, self.obs_delay_feet_max + 1, feet_cmd_dim,
                dtype=torch.float, device=self.device,
            )
            self.feet_cmd_ptr = 0
        # is_height is now a local variable in compute_observations and is no longer persisted
        self.wrist_pos_target = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.virtual_wrist_pos_target = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.wrist_quat_target = torch.zeros(
            self.num_envs, 8, dtype=torch.float, device=self.device, requires_grad=False
        )
        # Target angles of 6 wrist joints (joint space), used in _reward_tracking_wrist_joints
        # Format: [left_roll, left_pitch, left_yaw, right_roll, right_pitch, right_yaw]
        self.wrist_joint_target = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        # Store target elbow positions (computed from FK)
        # Format: [left_x, left_y, left_z, right_x, right_y, right_z], total 6 values
        # Principle: similar to wrist_pos_target, used for elbow position tracking
        self.elbow_pos_target = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.head_pos_target = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.is_head_pos = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool, requires_grad=False)
        # Main three-mode flags (50/25/25 sampling, mutually exclusive)
        self.is_walking_mode    = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool, requires_grad=False)
        self.is_pos_ctrl_mode   = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool, requires_grad=False)
        self.is_force_ctrl_mode = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool, requires_grad=False)
        # Store wrist z-axis offset for base height compensation
        self.wrist_z_offset = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Wrist tracking termination: fixed threshold (no dynamic curriculum)
        self.wrist_tracking_kill_threshold = 0.35  # Fixed threshold
        self.wrist_termination_start_iter = 5000  # Start termination after 2000 iterations
        
        # Fix tracking_wrist_sigma to 0.25 (no adaptive adjustment)
        self.cfg.rewards.tracking_wrist_sigma = 0.25
        
        # Initialize momentum observer buffers
        self._init_observer_buffers()
        
        # Pre-compile arm subchains
        def build_arm_subchains(end_link_name, arm_dof_indices_global):
            subchains = []
            urdf_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            with open(urdf_path, "rb") as f:
                urdf_data = f.read()
            full_chain = pk.build_serial_chain_from_urdf(urdf_data, root_link_name="torso_link", end_link_name=end_link_name)
            active_joints_in_chain = [j.name for j in full_chain.get_joints() if j.joint_type != 'fixed']
            for link_name, link_data in self.link_inertial_dict.items():
                try:
                    sub_chain = pk.build_serial_chain_from_urdf(urdf_data, root_link_name="torso_link", end_link_name=link_name)
                except:
                    continue
                sub_joints = [j.name for j in sub_chain.get_joints() if j.joint_type != 'fixed']
                if not set(sub_joints).issubset(set(active_joints_in_chain)):
                    continue
                sub_dof_indices = [self.dof_names.index(j_name) for j_name in sub_joints]
                subchains.append({
                    'chain': sub_chain.to(dtype=torch.float, device=self.device),
                    'dof_indices': torch.tensor(sub_dof_indices, device=self.device, dtype=torch.long),
                    'mass': link_data['mass'],
                    'com_offset': link_data['com_offset'].to(device=self.device),
                    'name': link_name
                })
            return subchains
        
        self.left_subchains = build_arm_subchains("left_hand_palm_link", self.left_arm_indices_in_dof_tensor)
        self.right_subchains = build_arm_subchains("right_hand_palm_link", self.right_arm_indices_in_dof_tensor)
        import pytorch_kinematics.transforms.transform3d as tf3d
        if not getattr(tf3d.Transform3d, "_patched_by_user", False):
            _original_init = tf3d.Transform3d.__init__
            def _safe_init(self, pos=None, rot=None, device=None, matrix=None, **kwargs):
                # Guard against string inputs that pytorch_kinematics may pass
                if isinstance(pos, str):
                    pos = None
                if isinstance(rot, str):
                    rot = None
                if device is None:
                    if matrix is not None and isinstance(matrix, torch.Tensor):
                        device = matrix.device
                    elif pos is not None and isinstance(pos, torch.Tensor):
                        device = pos.device
                    elif rot is not None and isinstance(rot, torch.Tensor):
                        device = rot.device
                _original_init(self, pos=pos, rot=rot, device=device, matrix=matrix, **kwargs)
            tf3d.Transform3d.__init__ = _safe_init
            tf3d.Transform3d._patched_by_user = True

    
    
        

    
    



    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        )

    def _create_trimesh(self):
        """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg."""
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size  # TODO: what is border size?
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim, self.terrain.vertices.flatten(order="C"), self.terrain.triangles.flatten(order="C"), tm_params
        )
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        )

    def _init_height_points(self):
        """Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """Samples heights of the terrain at required pointsFlast_force around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]
            ) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_states[:, :3]
            ).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()

        if not self.cfg.terrain.hard_terrain:
            points[:, :, 0] += self.terrain.st_distance  # transform to the source terrain observation

        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_init_heights(self, env_ids=None):
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(len(env_ids), self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        # ? No rotation when initializing heights

        y = torch.tensor([-0.25, -0.15, 0.0, 0.15, 0.25], device=self.device, requires_grad=False)
        x = torch.tensor([-0.25, -0.15, 0.0, 0.15, 0.25], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        num_height_points = grid_x.numel()
        sample_points = torch.zeros(self.num_envs, num_height_points, 3, device=self.device, requires_grad=False)
        sample_points[:, :, 0] = grid_x.flatten()
        sample_points[:, :, 1] = grid_y.flatten()

        if env_ids is not None:
            points = sample_points[env_ids] + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = sample_points + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()

        if not self.cfg.terrain.hard_terrain:
            points[:, :, 0] += self.terrain.st_distance  # transform to the source terrain observation

        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return torch.max(heights.view(len(env_ids), -1) * self.terrain.cfg.vertical_scale, dim=-1, keepdim=True).values

    def _add_height_noise(self, heights):
        # extend noise
        heights = heights.reshape(
            self.num_envs, len(self.cfg.terrain.measured_points_x), len(self.cfg.terrain.measured_points_y)
        )
        mean_height = heights[heights < 0].mean()
        valid_index = (heights < 0).float()
        kernel = (
            torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        valid_neighbor = torch.nn.functional.conv2d(valid_index.unsqueeze(1), kernel, padding=1).squeeze(1)
        random_tensor = torch.rand_like(heights)
        valid_neighbor = (
            (valid_neighbor > 0) & (heights >= 0) & (random_tensor < self.cfg.noise.height_measurements.extend_prob)
        )
        extend_noise = torch.normal(
            mean=mean_height + 0.1, std=0.03, size=heights[valid_neighbor].shape, device=self.device
        )
        heights[valid_neighbor] = extend_noise
        heights = heights.reshape(self.num_envs, -1)

        # vertical noise
        noise = torch_rand_float(
            -self.cfg.noise.height_measurements.vertical_scale,
            self.cfg.noise.height_measurements.vertical_scale,
            heights.shape,
            device=self.device,
        )
        offset = self.height_noise_offset
        heights = heights + noise * self.cfg.noise.noise_level + offset

        # repeat
        if torch.rand(1).item() < self.cfg.noise.height_measurements.map_repeat_prob:
            heights = self.last_heights
        self.last_heights = heights

        heights = torch.where((heights > 0.5), 1.0, heights)
        return heights

    def _draw_debug_vis(self):
        """Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        # draw height lines
        if self.cfg.terrain.mesh_type != "trimesh" or not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)


        for i in range(self.num_envs):      
            feet_sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 0, 1))   # blue: feet position target
            for side in range(2):
                x = self.feet_pos_target[i, side * 2 + 0].cpu().numpy()  # + self.root_states[i, 0].cpu().numpy()
                y = self.feet_pos_target[i, side * 2 + 1].cpu().numpy()  # + self.root_states[i, 1].cpu().numpy()
                z = 0
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(feet_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        for i in range(self.num_envs):
            for side in range(2):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                target_pos = (
                    quat_rotate(
                        self.base_quat[[i]],
                        self.commands[[i], 5 + side * 3 : 5 + side * 3 + 3],
                    )
                ).cpu().numpy()[0] + base_pos[:3]
                x = target_pos[0]  # + self.root_states[i, 0].cpu().numpy()
                y = target_pos[1]  # + self.root_states[i, 1].cpu().numpy()
                z = target_pos[2]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                #gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        for i in range(self.num_envs):
            pos_sphere_geom = gymutil.WireframeSphereGeometry(0.03, 6, 6, None, color=(1, 0, 0))    # red: position target
            for side in range(2):
                # wrist_pos_target shape: (num_envs, 6) -> [lx, ly, lz, rx, ry, rz]
                x = self.wrist_pos_target[i, side * 3 + 0].cpu().numpy()
                y = self.wrist_pos_target[i, side * 3 + 1].cpu().numpy()
                z = self.wrist_pos_target[i, side * 3 + 2].cpu().numpy()
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(pos_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        
        self.gym.refresh_rigid_body_state_tensor(self.sim)  # ensure state is refreshed


        left_wrist_pos = self.rigid_body_states[:, self.left_wrist_indices, :3].clone().mean(dim=1)
        right_wrist_pos = self.rigid_body_states[:, self.right_wrist_indices, :3].clone().mean(dim=1)
        for i in range(self.num_envs):
            current_wrist_sphere_geom = gymutil.WireframeSphereGeometry(0.03, 6, 6, None, color=(0, 1, 1))  # cyan: current wrist position
            left_pos = left_wrist_pos[i].cpu().numpy()
            sphere_pose = gymapi.Transform(gymapi.Vec3(left_pos[0], left_pos[1], left_pos[2]), r=None)
            gymutil.draw_lines(current_wrist_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

            right_pos = right_wrist_pos[i].cpu().numpy()
            sphere_pose = gymapi.Transform(gymapi.Vec3(right_pos[0], right_pos[1], right_pos[2]), r=None)
            gymutil.draw_lines(current_wrist_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 6, 6, None, color=(0, 1, 0))
        for i in range(self.num_envs):
            for side in range(2):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                target_pos = (
                    quat_rotate(
                        self.base_quat[[i]],
                        self.commands[[i], 11 + side * 3 : 11 + side * 3 + 3],
                    )
                ).cpu().numpy()[0] + base_pos[:3]
                x = target_pos[0]
                y = target_pos[1]
                z = target_pos[2]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        for i in range(self.num_envs):
            head_sphere_geom = gymutil.WireframeSphereGeometry(0.03, 6, 6, None, color=(1, 1, 0))  # yellow: head target
            x = self.head_pos_target[i, 0].cpu().numpy()
            y = self.head_pos_target[i, 1].cpu().numpy()
            z = self.head_pos_target[i, 2].cpu().numpy()
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(head_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        head_pos = self.rigid_body_states[:, self.head_index, :3]
        for i in range(self.num_envs):
            pos = head_pos[i].cpu().numpy()
            sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
            #gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        for i in range(self.num_envs):
            wrist_target_sphere_geom = gymutil.WireframeSphereGeometry(0.03, 6, 6, None, color=(1, 0, 1))  # magenta: wrist target
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            target_pos = (
                quat_rotate(
                    self.base_quat[[i]],
                    self.commands[[i], 17:20],
                )
            ).cpu().numpy()[
                0
            ] + base_pos[:3]
            x = target_pos[0]
            y = target_pos[1]
            z = target_pos[2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(wrist_target_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)


    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.use_physx_armature = self.cfg.asset.use_physx_armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(self.body_names)
        self.num_dof = len(self.dof_names)

        assert self.fix_waist or ("waist_roll_joint" in self.dof_names and "waist_pitch_joint" in self.dof_names), (
            "The robot asset must have 'waist_roll_joint' and 'waist_pitch_joint' DOFs, "
            "or the cfg.control.fix_waist must be set to True."
        )

        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        left_foot_names = [s for s in self.body_names if self.cfg.asset.left_foot_name in s]
        right_foot_names = [s for s in self.body_names if self.cfg.asset.right_foot_name in s]

        wrist_names = [s for s in self.body_names if self.cfg.asset.wrist_name in s]
        left_wrist_names = [s for s in self.body_names if self.cfg.asset.left_wrist_name in s]
        right_wrist_names = [s for s in self.body_names if self.cfg.asset.right_wrist_name in s]

        head_names = [s for s in self.body_names if self.cfg.asset.head_name in s]

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        self.default_rigid_body_mass = torch.zeros(
            self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False
        )

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []

        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.hand_payload = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )

        if "backpack_link" in self.body_names:
            self.backpack_payload = torch.zeros(
                self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
            )
        else:
            self.backpack_payload = None

        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(
                self.cfg.domain_rand.payload_mass_range[0],
                self.cfg.domain_rand.payload_mass_range[1],
                (self.num_envs, 1),
                device=self.device,
            )
            self.hand_payload = torch_rand_float(
                self.cfg.domain_rand.hand_payload_mass_range[0],
                self.cfg.domain_rand.hand_payload_mass_range[1],
                (self.num_envs, 2),
                device=self.device,
            )

            if self.backpack_payload is not None:
                self.backpack_payload = torch_rand_float(
                    self.cfg.domain_rand.backpack_payload_mass_range[0],
                    self.cfg.domain_rand.backpack_payload_mass_range[1],
                    (self.num_envs, 1),
                    device=self.device,
                )

        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(
                self.cfg.domain_rand.com_displacement_range[0],
                self.cfg.domain_rand.com_displacement_range[1],
                (self.num_envs, 3),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_body_displacement:
            self.body_displacement = torch_rand_float(
                self.cfg.domain_rand.body_displacement_range[0],
                self.cfg.domain_rand.body_displacement_range[1],
                (self.num_envs, 3),
                device=self.device,
            )

        self.torso_body_index = self.body_names.index("torso_link")
        self.left_hand_index = self.body_names.index("left_hand_palm_link")
        self.right_hand_index = self.body_names.index("right_hand_palm_link")

        if "backpack_link" in self.body_names:
            self.backpack_index = self.body_names.index("backpack_link")
        else:
            self.backpack_index = None

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0
            )
            dof_props = self._process_dof_props(dof_props_asset, i)

            # Use torque mode consistently for upper body to avoid stacking built-in PD; switch back in cfg if position mode is needed
            if self.fix_waist:
                dof_props["driveMode"][12:].fill(gymapi.DOF_MODE_EFFORT)
                dof_props["stiffness"][12:] = [0.0] * (len(dof_props["stiffness"]) - 12)
                dof_props["damping"][12:] = [0.0] * (len(dof_props["damping"]) - 12)
            else:
                # Switch waist yaw and arm joints to torque control as well
                dof_props["driveMode"][12] = gymapi.DOF_MODE_EFFORT
                dof_props["stiffness"][12] = 0.0
                dof_props["damping"][12] = 0.0
                dof_props["driveMode"][15:] = gymapi.DOF_MODE_EFFORT
                dof_props["stiffness"][15:] = [0.0] * (len(dof_props["stiffness"]) - 15)
                dof_props["damping"][15:] = [0.0] * (len(dof_props["damping"]) - 15)

            armature = []
            for dof_idx in range(self.num_dof):
                name = self.dof_names[dof_idx]
                armature.append(self.cfg.init_state.joint_armature[name])
            dof_props["armature"] = armature

            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            if i == 0:
                self.default_com = copy.deepcopy(body_props[0].com)
                self.default_body_com = copy.deepcopy(body_props[self.torso_body_index].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass

            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        self.wrist_indices = torch.zeros(len(wrist_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(wrist_names)):
            self.wrist_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], wrist_names[i]
            )

        self.head_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], head_names[0])

        knee_names = self.cfg.asset.knee_names
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], knee_names[i]
            )

        self.left_foot_indices = torch.zeros(
            len(left_foot_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(left_foot_names)):
            self.left_foot_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], left_foot_names[i]
            )

        self.right_foot_indices = torch.zeros(
            len(right_foot_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(right_foot_names)):
            self.right_foot_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], right_foot_names[i]
            )

        self.left_wrist_indices = torch.zeros(
            len(left_wrist_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(left_wrist_names)):
            self.left_wrist_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], left_wrist_names[i]
            )
        self.right_wrist_indices = torch.zeros(
            len(right_wrist_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(right_wrist_names)):
            self.right_wrist_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], right_wrist_names[i]
            )

        # Define indices for elbow links
        elbow_names = ["left_elbow_link", "right_elbow_link"]
        self.elbow_indices = torch.zeros(len(elbow_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(elbow_names)):
            self.elbow_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], elbow_names[i]
            )
        
        self.left_elbow_index = self.elbow_indices[0]
        self.right_elbow_index = self.elbow_indices[1]

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

        self.left_leg_joint_indices = torch.zeros(
            len(self.cfg.asset.left_leg_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.left_leg_joints)):
            self.left_leg_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_leg_joints[i])

        self.right_leg_joint_indices = torch.zeros(
            len(self.cfg.asset.right_leg_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.right_leg_joints)):
            self.right_leg_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_leg_joints[i])

        self.leg_joint_indices = torch.cat((self.left_leg_joint_indices, self.right_leg_joint_indices))

        self.left_hip_joint_indices = torch.zeros(
            len(self.cfg.asset.left_hip_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.left_hip_joints)):
            self.left_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_joints[i])

        self.right_hip_joint_indices = torch.zeros(
            len(self.cfg.asset.right_hip_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.right_hip_joints)):
            self.right_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_joints[i])

        self.hip_joint_indices = torch.cat((self.left_hip_joint_indices, self.right_hip_joint_indices))

        self.hip_pitch_joint_indices = torch.zeros(
            len(self.cfg.asset.hip_pitch_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.hip_pitch_joints)):
            self.hip_pitch_joint_indices[i] = self.dof_names.index(self.cfg.asset.hip_pitch_joints[i])

        self.hip_roll_yaw_joint_indices = torch.zeros(
            len(self.cfg.asset.hip_roll_yaw_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.hip_roll_yaw_joints)):
            self.hip_roll_yaw_joint_indices[i] = self.dof_names.index(self.cfg.asset.hip_roll_yaw_joints[i])

        self.ankle_joint_indices = torch.zeros(
            len(self.cfg.asset.ankle_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.ankle_joints)):
            self.ankle_joint_indices[i] = self.dof_names.index(self.cfg.asset.ankle_joints[i])

        hip_yaw_joint_names = ["left_hip_yaw_joint", "right_hip_yaw_joint"]
        self.hip_yaw_joint_indices = torch.zeros(
            len(hip_yaw_joint_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(hip_yaw_joint_names)):
            self.hip_yaw_joint_indices[i] = self.dof_names.index(hip_yaw_joint_names[i])

        hip_roll_joint_names = ["left_hip_roll_joint", "right_hip_roll_joint"]
        self.hip_roll_joint_indices = torch.zeros(
            len(hip_roll_joint_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(hip_roll_joint_names)):
            self.hip_roll_joint_indices[i] = self.dof_names.index(hip_roll_joint_names[i])

        self.knee_joint_indices = torch.zeros(
            len(self.cfg.asset.knee_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.knee_joints)):
            self.knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.knee_joints[i])

        wrist_joint_names = [
            "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]
        self.wrist_joint_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in wrist_joint_names],
            device=self.device, dtype=torch.long,
        )

        self.upper_body_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], self.cfg.asset.upper_body_link
        )
        self.imu_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], self.cfg.asset.imu_link
        )
        self.torso_imu_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], self.cfg.asset.imu_torso
        )

        self.torso_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "torso_link")

        self.left_wrist_handle = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], "left_hand_palm_link"
        )
        self.right_wrist_handle = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], "right_hand_palm_link"
        )

        if not self.fix_waist:
            self.waist_active_joint_indices = torch.zeros(2, dtype=torch.long, device=self.device, requires_grad=False)
            waist_joints = ["waist_roll_joint", "waist_pitch_joint"]
            for i in range(len(waist_joints)):
                self.waist_active_joint_indices[i] = self.dof_names.index(waist_joints[i])

        # num_dof is initialized now, safe to create noise_scale_vec
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        
        # Initialize pytorch-kinematics chain (starting from torso_link)
        urdf_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        with open(urdf_path, "rb") as f:
            urdf_data = f.read()
        
        # Build left/right wrist kinematic chains (from torso_link to wrist)
        self.left_arm_chain = pk.build_serial_chain_from_urdf(
            urdf_data, 
            root_link_name="torso_link",
            end_link_name="left_hand_palm_link"
        ).to(dtype=torch.float, device=self.device)
        self.left_arm_base_link = "torso_link"
        
        self.right_arm_chain = pk.build_serial_chain_from_urdf(
            urdf_data, 
            root_link_name="torso_link",
            end_link_name="right_hand_palm_link"
        ).to(dtype=torch.float, device=self.device)
        self.right_arm_base_link = "torso_link"
        
        # Parse XML directly to obtain inertial parameters (mass, CoM, inertia tensor)
        # Since PK cannot read them here, parse manually to avoid version constraints
        import xml.etree.ElementTree as ET
        self.link_inertial_dict = {}
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        for link in root.findall('link'):
            name = link.get('name')
            inertial = link.find('inertial')
            if inertial is not None:
                # Parse mass
                mass_elem = inertial.find('mass')
                if mass_elem is not None:
                    mass = float(mass_elem.get('value'))
                    # Parse CoM offset (inertial frame relative to link, with rpy)
                    origin_elem = inertial.find('origin')
                    if origin_elem is not None:
                        xyz_str = origin_elem.get('xyz', '0 0 0')
                        xyz = [float(x) for x in xyz_str.split()]
                        rpy_str = origin_elem.get('rpy', '0 0 0')
                        rpy = [float(x) for x in rpy_str.split()]
                    else:
                        xyz = [0.0, 0.0, 0.0]
                        rpy = [0.0, 0.0, 0.0]

                    # Parse inertia tensor (in inertial frame)
                    inertia_elem = inertial.find('inertia')
                    if inertia_elem is None:
                        continue
                    inertia_inertial = torch.tensor([
                        [float(inertia_elem.get('ixx', '0')), float(inertia_elem.get('ixy', '0')), float(inertia_elem.get('ixz', '0'))],
                        [float(inertia_elem.get('ixy', '0')), float(inertia_elem.get('iyy', '0')), float(inertia_elem.get('iyz', '0'))],
                        [float(inertia_elem.get('ixz', '0')), float(inertia_elem.get('iyz', '0')), float(inertia_elem.get('izz', '0'))],
                    ], device=self.device, dtype=torch.float)

                    # inertial -> link rotation
                    rpy_tensor = torch.tensor(rpy, device=self.device, dtype=torch.float)
                    r, p, y = rpy_tensor
                    cr, sr = torch.cos(r), torch.sin(r)
                    cp, sp = torch.cos(p), torch.sin(p)
                    cy, sy = torch.cos(y), torch.sin(y)
                    R_il = torch.stack([
                        torch.stack([cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr], dim=0),
                        torch.stack([sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr], dim=0),
                        torch.stack([-sp,  cp*sr,               cp*cr             ], dim=0)
                    ], dim=0)

                    # Transform inertia and CoM into link frame
                    inertia_link = R_il @ inertia_inertial @ R_il.transpose(-1, -2)
                    com_offset_link = (R_il @ torch.tensor(xyz, device=self.device, dtype=torch.float).unsqueeze(-1)).squeeze(-1)

                    # Keep only links with mass above threshold, filter dummy links
                    if mass > 1e-3:
                        self.link_inertial_dict[name] = {
                            "mass": mass,
                            "com_offset": com_offset_link,
                            "inertia_tensor": inertia_link,
                            "inertia": {
                                'ixx': inertia_link[0, 0].item(),
                                'ixy': inertia_link[0, 1].item(),
                                'ixz': inertia_link[0, 2].item(),
                                'iyy': inertia_link[1, 1].item(),
                                'iyz': inertia_link[1, 2].item(),
                                'izz': inertia_link[2, 2].item(),
                            },
                        }
                        # Print once for verification (debug-only; can be commented after validation)
                        print(f"🔗 Loaded URDF Data: {name}, Mass={mass}kg")
        
        # Get required joint-name list for the chain
        left_chain_names = [j.name for j in self.left_arm_chain.get_joints()]
        right_chain_names = [j.name for j in self.right_arm_chain.get_joints()]
        
        # Create DOF name-to-index mapping
        dof_name_to_index = {name: i for i, name in enumerate(self.dof_names)}
        
        # Map chain joint names to global DOF indices
        self.left_arm_indices_in_dof_tensor = torch.tensor(
            [dof_name_to_index[name] for name in left_chain_names], 
            device=self.device, dtype=torch.long
        )
        
        self.right_arm_indices_in_dof_tensor = torch.tensor(
            [dof_name_to_index[name] for name in right_chain_names], 
            device=self.device, dtype=torch.long
        )
        
        # Build left/right elbow kinematic chains (from torso_link to elbow_link)
        self.left_elbow_chain = pk.build_serial_chain_from_urdf(
            urdf_data, 
            root_link_name="torso_link",
            end_link_name="left_elbow_link"
        ).to(dtype=torch.float, device=self.device)
        
        self.right_elbow_chain = pk.build_serial_chain_from_urdf(
            urdf_data, 
            root_link_name="torso_link",
            end_link_name="right_elbow_link"
        ).to(dtype=torch.float, device=self.device)
        
        # Get required joint-name list for elbow chain
        left_elbow_chain_names = [j.name for j in self.left_elbow_chain.get_joints()]
        right_elbow_chain_names = [j.name for j in self.right_elbow_chain.get_joints()]
        
        # Map elbow-chain joint names to global DOF indices
        self.left_elbow_indices_in_dof_tensor = torch.tensor(
            [dof_name_to_index[name] for name in left_elbow_chain_names], 
            device=self.device, dtype=torch.long
        )
        
        self.right_elbow_indices_in_dof_tensor = torch.tensor(
            [dof_name_to_index[name] for name in right_elbow_chain_names], 
            device=self.device, dtype=torch.long
        )

    def _get_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.terrain.num_cols),
                rounding_mode="floor",
            ).to(torch.long)
            self.max_terrain_level = self.cfg.terrain.max_terrain_level
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            self.terrain_platform_length = (
                torch.from_numpy(self.terrain.platform_length).to(self.device).to(torch.float)
            )
            self.platform_length = self.terrain_platform_length[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.upper_interval = np.ceil(self.cfg.domain_rand.upper_interval_s / self.dt)
        wrist_target_interval = getattr(self.cfg.domain_rand, "wrist_target_interval", 1)
        self.cfg.domain_rand.wrist_target_interval = max(1, int(np.ceil(wrist_target_interval)))

