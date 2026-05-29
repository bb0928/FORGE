import numpy as np
import torch
from isaacgym.torch_utils import quat_apply, quat_mul, quat_rotate, quat_rotate_inverse, torch_rand_float
from legged_gym.utils.math import quat_from_euler_xyz

from .ee_residual_math_utils import euler_from_quaternion, quaternion_from_rotation_matrix


class EEResidualWristControlMixin:
    def _step_contact_targets(self):
        frequencies = self.commands[:, 5]
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)
        durations = torch.full_like(self.gait_indices, 0.5)
        phases = 0.5
        kappa = 0.07
        foot_indices = [
            self.gait_indices + phases,  # FL
            self.gait_indices,  # FR
        ]
        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(2)], dim=1), 1.0)
        for fi in foot_indices:
            stance = fi < durations
            swing = fi >= durations
            fi[stance] = fi[stance] * (0.5 / durations[stance])
            fi[swing] = 0.5 + (fi[swing] - durations[swing]) * (0.5 / (1 - durations[swing]))

        # self.foot_indices = torch.remainder(torch.stack(foot_indices, dim=1), 1.0)

        self.clock_inputs = torch.stack([torch.sin(2 * np.pi * fi) for fi in foot_indices], dim=1)
        cdf = torch.distributions.Normal(0, kappa).cdf

        def smooth(fi):
            f = torch.remainder(fi, 1.0)
            return cdf(f) * (1 - cdf(f - 0.5)) + cdf(f - 1) * (1 - cdf(f - 1 - 0.5))

        self.desired_contact_states = torch.stack([smooth(fi) for fi in foot_indices], dim=1)

    def _update_wrist_virtual_targets(self):
        half = self.wrist_force_command_half or 3
        total_dims = half * 2

        if (
            self.wrist_force_command_slice is not None
            and self.wrist_force_command_slice.stop <= self.commands.shape[1]
        ):
            forces_cmd = self.commands[:, self.wrist_force_command_slice]
        else:
            forces_cmd = torch.zeros(self.num_envs, total_dims, device=self.device)

        left_cmd = forces_cmd[:, :half]
        right_cmd = forces_cmd[:, half : half * 2]

        # Extract yaw from base_quat (only yaw rotation, like b2z1's base_yaw_quat)
        roll, pitch, yaw = euler_from_quaternion(self.base_quat)
        base_yaw_quat = quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            yaw.squeeze(-1)
        )
        
        # Apply yaw-only rotation (like b2z1)
        self.left_force_cmd_global = quat_apply(base_yaw_quat, left_cmd)
        self.right_force_cmd_global = quat_apply(base_yaw_quat, right_cmd)
        
        self.forces_cmd_global = torch.cat((self.left_force_cmd_global, self.right_force_cmd_global), dim=-1)

        use_hat = hasattr(self, "wrist_force_hat_world") and hasattr(self.cfg, 'observer') and self.cfg.observer.enable
        if use_hat:
            # Get observer estimate in World Frame
            left_external_hat = self.wrist_force_hat_world[:, :3]  # [num_envs, 3]
            right_external_hat = self.wrist_force_hat_world[:, 3:6]  # [num_envs, 3]
            
            # Apply predicted correction: F_final = F_hat + F_correction
            # Correction is in Base-Yaw Frame, need to rotate to World Frame
            F_correction_left_local = self.last_force_correction[:, :3]  # [num_envs, 3]
            F_correction_right_local = self.last_force_correction[:, 3:6]  # [num_envs, 3]
            
            # Rotate correction from Base-Yaw Frame to World Frame
            F_correction_left_world = quat_apply(base_yaw_quat, F_correction_left_local)
            F_correction_right_world = quat_apply(base_yaw_quat, F_correction_right_local)
            
            # NOTE: correction temporarily disabled; impedance uses F_hat only
            left_external = left_external_hat
            right_external = right_external_hat
        else:
            # If observer is disabled, fallback to GT forces (no correction applied)
            left_external = self.forces[:, self.left_wrist_handle, :3]
            right_external = self.forces[:, self.right_wrist_handle, :3]

        # Use F_cmd only if enabled in config (to avoid supervision signal confusion)
        use_f_cmd = getattr(self.cfg.commands, 'use_force_cmd_in_control', False)
        if use_f_cmd:
            self.left_wrist_forces_total = left_external + self.left_force_cmd_global
            self.right_wrist_forces_total = right_external + self.right_force_cmd_global
        else:
            # Temporarily disable F_cmd: use only external force
            self.left_wrist_forces_total = left_external
            self.right_wrist_forces_total = right_external

        if hasattr(self, "wrist_force_kps") and self.wrist_force_kps.shape[0] == self.num_envs:
            force_gains = torch.clamp(self.wrist_force_kps, min=1.0)
        else:
            base_gain = max(float(self.wrist_force_kp), 1.0)
            force_gains = torch.ones(self.num_envs, 3, device=self.device) * base_gain

        left_virtual = self.wrist_pos_target[:, :3] + self.left_wrist_forces_total / force_gains
        right_virtual = self.wrist_pos_target[:, 3:6] + self.right_wrist_forces_total / force_gains
        self.virtual_wrist_pos_target = torch.cat((left_virtual, right_virtual), dim=1)

    def _generate_random_upper_actions(self):
        """Generate random upper body actions using base class logic.
        Called every step regardless of is_wrist_pos.
        """
        capped_ratio = torch.clamp(
            torch.full_like(self._upper_curriculum_cap, self.action_curriculum_ratio),
            max=self._upper_curriculum_cap,
        )  # [len(upper_dof_indices)]
        uu = torch.rand(self.num_envs, len(self.upper_dof_indices), device=self.device)
        self.random_upper_ratio = -1.0 / (20 * (1 - capped_ratio * 0.99)) * torch.log(
            1 - uu + uu * torch.exp(-20 * (1 - capped_ratio * 0.99))
        )
        self.random_joint_ratio = self.random_upper_ratio * torch.rand(
            self.num_envs, len(self.upper_dof_indices), device=self.device
        )
        rand_pos = torch.rand(self.num_envs, len(self.upper_dof_indices), device=self.device) - 0.5
        self.random_upper_actions = (
            (self.action_min[:, self.upper_dof_indices] * (rand_pos >= 0))
            + (self.action_max[:, self.upper_dof_indices] * (rand_pos < 0))
        ) * self.random_joint_ratio
        return self.random_upper_actions

    def _assign_wrist_targets_from_fk(self, env_ids: torch.Tensor, upper_actions: torch.Tensor):
        """Compute wrist position targets from FK using pytorch-kinematics (from torso_link)."""
        if env_ids.numel() == 0:
            return
        env_ids_long = env_ids.long()
        num_envs = env_ids_long.shape[0]
        
        # 1. Convert actions to joint positions
        if self.default_dof_pos.dim() == 1:
            default_upper = self.default_dof_pos[self.upper_dof_indices].unsqueeze(0).expand(num_envs, -1)
        else:
            default_upper = self.default_dof_pos[:, self.upper_dof_indices]
            if default_upper.shape[0] == 1:
                default_upper = default_upper.expand(num_envs, -1)
        
        target_upper_dof_pos = default_upper + upper_actions[env_ids_long] * self.cfg.control.action_scale
        
        current_full_dof = self.dof_pos[env_ids_long].clone()
        
        for i, idx in enumerate(self.upper_dof_indices):
            current_full_dof[:, idx] = target_upper_dof_pos[:, i]
        
        if "left_shoulder_pitch_joint" in self.dof_names:
            left_shoulder_pitch_idx = self.dof_names.index("left_shoulder_pitch_joint")
            current_full_dof[:, left_shoulder_pitch_idx] = torch.clamp(
                current_full_dof[:, left_shoulder_pitch_idx], max=0.0
            )
        if "right_shoulder_pitch_joint" in self.dof_names:
            right_shoulder_pitch_idx = self.dof_names.index("right_shoulder_pitch_joint")
            current_full_dof[:, right_shoulder_pitch_idx] = torch.clamp(
                current_full_dof[:, right_shoulder_pitch_idx], max=0.0
            )
        
        left_arm_dof = current_full_dof[:, self.left_arm_indices_in_dof_tensor]
        right_arm_dof = current_full_dof[:, self.right_arm_indices_in_dof_tensor]
        
        left_elbow_dof = current_full_dof[:, self.left_elbow_indices_in_dof_tensor]
        right_elbow_dof = current_full_dof[:, self.right_elbow_indices_in_dof_tensor]
        
        left_tf = self.left_arm_chain.forward_kinematics(left_arm_dof)
        right_tf = self.right_arm_chain.forward_kinematics(right_arm_dof)
        
        left_elbow_tf = self.left_elbow_chain.forward_kinematics(left_elbow_dof)
        right_elbow_tf = self.right_elbow_chain.forward_kinematics(right_elbow_dof)
        
        left_pos_rel = left_tf.get_matrix()[:, :3, 3]  # [N, 3]
        right_pos_rel = right_tf.get_matrix()[:, :3, 3]  # [N, 3]
        
        left_rot_matrix = left_tf.get_matrix()[:, :3, :3]
        right_rot_matrix = right_tf.get_matrix()[:, :3, :3]
        
        left_quat_rel = quaternion_from_rotation_matrix(left_rot_matrix)  # [N, 4]
        right_quat_rel = quaternion_from_rotation_matrix(right_rot_matrix)  # [N, 4]
        
        torso_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        torso_pos_current = torso_states[env_ids_long, self.torso_body_index, 0:3]  # [N, 3]
        torso_quat_current = torso_states[env_ids_long, self.torso_body_index, 3:7]
        
        torso_to_base_offset = 0.044
        target_torso_z = self.commands[env_ids_long, 4] + torso_to_base_offset
        torso_pos = torch.cat([torso_pos_current[:, :2], target_torso_z.unsqueeze(1)], dim=1)  # [N, 3]
        
        roll, pitch, yaw = euler_from_quaternion(torso_quat_current)
        yaw_flat = yaw.view(-1)
        zeros = torch.zeros_like(yaw_flat)
        upright_torso_quat = quat_from_euler_xyz(zeros, zeros, yaw_flat)  # [N, 4]
        upright_torso_quat = upright_torso_quat.view(-1, 4)
        
        left_pos_world = quat_rotate(upright_torso_quat, left_pos_rel) + torso_pos
        right_pos_world = quat_rotate(upright_torso_quat, right_pos_rel) + torso_pos
        
        left_quat_world = quat_mul(upright_torso_quat, left_quat_rel)  # [N, 4]
        right_quat_world = quat_mul(upright_torso_quat, right_quat_rel)  # [N, 4]
        
        left_elbow_pos_rel = left_elbow_tf.get_matrix()[:, :3, 3]  # [N, 3]
        right_elbow_pos_rel = right_elbow_tf.get_matrix()[:, :3, 3]  # [N, 3]
        
        left_elbow_pos_world = quat_rotate(upright_torso_quat, left_elbow_pos_rel) + torso_pos
        right_elbow_pos_world = quat_rotate(upright_torso_quat, right_elbow_pos_rel) + torso_pos
        
        targets = torch.cat((left_pos_world, right_pos_world), dim=1)  # [N, 6]
        quat_targets = torch.cat((left_quat_world, right_quat_world), dim=1)  # [N, 8]
        elbow_targets = torch.cat((left_elbow_pos_world, right_elbow_pos_world), dim=1)  # [N, 6]
        
        
        self.wrist_pos_target[env_ids_long] = targets
        self.virtual_wrist_pos_target[env_ids_long] = targets
        self.wrist_quat_target[env_ids_long] = quat_targets
        self.elbow_pos_target[env_ids_long] = elbow_targets
        self.wrist_joint_target[env_ids_long] = current_full_dof[:, self.wrist_joint_dof_indices]

    def compute_force_compensation(self, wrist_forces):
        """
        
        
        Args:
            
        Returns:
        """
        
        left_wrist_force_local = quat_rotate_inverse(
            self.base_quat, wrist_forces[:, 0, :]
        )
        right_wrist_force_local = quat_rotate_inverse(
            self.base_quat, wrist_forces[:, 1, :]
        )
        
        left_displacement = left_wrist_force_local / self.wrist_force_kp
        right_displacement = right_wrist_force_local / self.wrist_force_kp
        
        cartesian_compensation = torch.stack([left_displacement, right_displacement], dim=1)
        
        return cartesian_compensation

    def _init_force_parameters(self):
        """Initialize force command parameters based on config"""
        if hasattr(self.cfg.commands, 'push_wrist_stators') and self.cfg.commands.push_wrist_stators:
            # Initialize force gains
            if self.cfg.commands.randomize_wrist_force_gains:
                self.wrist_force_kps[:] = torch_rand_float(
                    self.cfg.commands.wrist_force_kp_range[0],
                    self.cfg.commands.wrist_force_kp_range[1],
                    (self.num_envs, 3),
                    device=self.device
                )
                self.wrist_force_kds[:] = self.wrist_force_kps * self.cfg.commands.wrist_prop_kd
            else:
                self.wrist_force_kps[:] = self.cfg.commands.wrist_force_kp_range[0]
                self.wrist_force_kds[:] = self.cfg.commands.wrist_force_kd_range[0]
                
            # Initialize push intervals
            self.push_interval_wrist_cmd[:, 0] = torch.randint(
                int(self.cfg.commands.push_wrist_interval_s_cmd[0] / self.dt),
                int(self.cfg.commands.push_wrist_interval_s_cmd[1] / self.dt),
                (self.num_envs,),
                device=self.device
            )
            self.push_interval_wrist_ext[:, 0] = torch.randint(
                int(self.cfg.commands.push_wrist_interval_s_ext[0] / self.dt),
                int(self.cfg.commands.push_wrist_interval_s_ext[1] / self.dt),
                (self.num_envs,),
                device=self.device
            )
            
            # Initialize timing parameters
            self.push_duration_wrist_cmd_min = self.cfg.commands.push_wrist_duration_s_cmd[0] / self.dt
            self.push_duration_wrist_cmd_max = self.cfg.commands.push_wrist_duration_s_cmd[1] / self.dt
            self.push_duration_wrist_ext_min = self.cfg.commands.push_wrist_duration_s_ext[0] / self.dt
            self.push_duration_wrist_ext_max = self.cfg.commands.push_wrist_duration_s_ext[1] / self.dt
            self.settling_time_force_wrist = self.cfg.commands.settling_time_force_wrist_s / self.dt

    def _push_wrist(self, env_ids_all):
        """Randomly pushes the wrist stators. Emulates an impulse by setting a randomized wrist force.
        Based on b2z1's _push_gripper implementation.
        """
        total_dims = (self.wrist_force_command_half or 3) * 2
        if not (hasattr(self.cfg.commands, 'push_wrist_stators') and self.cfg.commands.push_wrist_stators):
            return
            
        # CMD force (command-based force)
        new_selected_env_ids_cmd = env_ids_all[(self.episode_length_buf % self.push_interval_wrist_cmd[:, 0]) == 0]
        
        if new_selected_env_ids_cmd.nelement() > 0:
            self.freed_envs_wrist_cmd[new_selected_env_ids_cmd] = torch.rand(
                len(new_selected_env_ids_cmd), dtype=torch.float, device=self.device, requires_grad=False
            ) > self.cfg.commands.wrist_forced_prob_cmd
            
            min_force_cmd = self.cfg.commands.max_push_force_xyz_wrist_cmd[0]
            max_force_cmd = self.cfg.commands.max_push_force_xyz_wrist_cmd[1]

            rand_force = torch_rand_float(
                min_force_cmd, max_force_cmd, (len(new_selected_env_ids_cmd), total_dims), device=self.device
            )
            self.force_target_wrist_cmd[new_selected_env_ids_cmd, :total_dims] = rand_force
            
            push_duration_wrist_cmd = torch_rand_float(
                self.push_duration_wrist_cmd_min, self.push_duration_wrist_cmd_max, 
                (len(new_selected_env_ids_cmd), 1), device=self.device
            ).view(len(new_selected_env_ids_cmd))
            
            push_duration_wrist_cmd = torch.clip(
                push_duration_wrist_cmd, 
                max=(self.push_interval_wrist_cmd[new_selected_env_ids_cmd, 0] - self.settling_time_force_wrist)/2
            ).to(self.device)
            
            self.push_end_time_wrist_cmd[new_selected_env_ids_cmd] = (
                self.episode_length_buf[new_selected_env_ids_cmd] + push_duration_wrist_cmd
            )
            self.push_duration_wrist_cmd[new_selected_env_ids_cmd] = push_duration_wrist_cmd
            self.selected_env_ids_wrist_cmd[new_selected_env_ids_cmd] = 1
                
        # Apply forces to selected envs
        if self.episode_length_buf[self.selected_env_ids_wrist_cmd == 1].nelement() > 0:
            subset_env_ids_selected = env_ids_all[self.selected_env_ids_wrist_cmd == 1]

            # Step 1: apply force from 0 to force_target_wrist_cmd
            env_ids_apply_push_step1 = subset_env_ids_selected[
                self.episode_length_buf[self.selected_env_ids_wrist_cmd == 1] < 
                (self.push_end_time_wrist_cmd[self.selected_env_ids_wrist_cmd == 1]).type(torch.int32)
            ]
            
            if env_ids_apply_push_step1.nelement() > 0:
                push_duration_reshaped = self.push_duration_wrist_cmd[env_ids_apply_push_step1].unsqueeze(-1)
                ramp = torch.clamp(
                    self.episode_length_buf[env_ids_apply_push_step1].unsqueeze(-1)
                    - (self.push_end_time_wrist_cmd[env_ids_apply_push_step1].unsqueeze(-1) - push_duration_reshaped),
                    torch.zeros_like(push_duration_reshaped),
                    push_duration_reshaped,
                )

                self.current_Fxyz_wrist_cmd[env_ids_apply_push_step1, :total_dims] = (
                    self.force_target_wrist_cmd[env_ids_apply_push_step1, :total_dims] / push_duration_reshaped
                ) * ramp

                # Set force commands in commands array
                if self.wrist_force_command_slice is not None:
                    self.commands[env_ids_apply_push_step1, self.wrist_force_command_slice] = (
                        self.current_Fxyz_wrist_cmd[env_ids_apply_push_step1, :total_dims]
                    )
 
            # Step 2: apply force from force_target_wrist_cmd back to 0
            env_ids_apply_push_step2 = subset_env_ids_selected[
                self.episode_length_buf[self.selected_env_ids_wrist_cmd == 1] > 
                (self.push_end_time_wrist_cmd[self.selected_env_ids_wrist_cmd == 1] + self.settling_time_force_wrist).type(torch.int32)
            ]
            
            if env_ids_apply_push_step2.nelement() > 0:
                push_duration_reshaped = self.push_duration_wrist_cmd[env_ids_apply_push_step2].unsqueeze(-1)
                ramp = torch.clamp(
                    self.episode_length_buf[env_ids_apply_push_step2].unsqueeze(-1)
                    - (self.push_end_time_wrist_cmd[env_ids_apply_push_step2].unsqueeze(-1) + self.settling_time_force_wrist),
                    torch.zeros_like(push_duration_reshaped),
                    push_duration_reshaped,
                )
                self.current_Fxyz_wrist_cmd[env_ids_apply_push_step2, :total_dims] = (
                    self.force_target_wrist_cmd[env_ids_apply_push_step2, :total_dims]
                    - (
                        self.force_target_wrist_cmd[env_ids_apply_push_step2, :total_dims] / push_duration_reshaped
                    )
                    * ramp
                )

                # Update force commands
                if self.wrist_force_command_slice is not None:
                    self.commands[env_ids_apply_push_step2, self.wrist_force_command_slice] = (
                        self.current_Fxyz_wrist_cmd[env_ids_apply_push_step2, :total_dims]
                    )
                    
            # Reset the tensors when force period is complete
            env_ids_to_reset = subset_env_ids_selected[
                self.episode_length_buf[self.selected_env_ids_wrist_cmd == 1] >= 
                (self.push_end_time_wrist_cmd[self.selected_env_ids_wrist_cmd == 1] + 
                 self.settling_time_force_wrist + self.push_duration_wrist_cmd[self.selected_env_ids_wrist_cmd == 1]
                ).type(torch.int32)
            ]
            
            if env_ids_to_reset.nelement() > 0:
                self.selected_env_ids_wrist_cmd[env_ids_to_reset] = 0
                self.force_target_wrist_cmd[env_ids_to_reset, :total_dims] = 0.
                self.current_Fxyz_wrist_cmd[env_ids_to_reset, :total_dims] = 0.
                self.push_end_time_wrist_cmd[env_ids_to_reset] = 0.
                self.push_duration_wrist_cmd[env_ids_to_reset] = 0.
                if self.wrist_force_command_slice is not None:
                    self.commands[env_ids_to_reset, self.wrist_force_command_slice] = 0.0
                self.push_interval_wrist_cmd[env_ids_to_reset, 0] = torch.randint(
                    int(self.cfg.commands.push_wrist_interval_s_cmd[0] / self.dt),
                    int(self.cfg.commands.push_wrist_interval_s_cmd[1] / self.dt),
                    (len(env_ids_to_reset), 1), 
                    device=self.device
                )[:, 0]
                    
        # Handle freed command environments (mirror b2z1 release logic)
        if torch.any(self.freed_envs_wrist_cmd):
            freed_cmd_ids = torch.nonzero(self.freed_envs_wrist_cmd).flatten()
            self.selected_env_ids_wrist_cmd[freed_cmd_ids] = 0
            self.force_target_wrist_cmd[freed_cmd_ids, :total_dims] = 0.
            self.current_Fxyz_wrist_cmd[freed_cmd_ids, :total_dims] = 0.
            self.push_end_time_wrist_cmd[freed_cmd_ids] = 0.
            self.push_duration_wrist_cmd[freed_cmd_ids] = 0.
            if self.wrist_force_command_slice is not None:
                self.commands[freed_cmd_ids, self.wrist_force_command_slice] = 0.0
            self.freed_envs_wrist_cmd[freed_cmd_ids] = False

        # === External force pushes (actual non-contact forces) ===
        new_selected_env_ids_ext = env_ids_all[(self.episode_length_buf % self.push_interval_wrist_ext[:, 0]) == 0]
        if new_selected_env_ids_ext.nelement() > 0:
            env_count = new_selected_env_ids_ext.shape[0]
            self.freed_envs_wrist_ext[new_selected_env_ids_ext] = torch.rand(
                env_count, dtype=torch.float, device=self.device, requires_grad=False
            ) > self.cfg.commands.wrist_forced_prob_ext

            min_force_ext = self.cfg.commands.max_push_force_xyz_wrist_ext[0]
            max_force_ext = self.cfg.commands.max_push_force_xyz_wrist_ext[1]
            self.force_target_wrist_ext[new_selected_env_ids_ext, 0] = torch_rand_float(
                min_force_ext, max_force_ext, (env_count, 1), device=self.device
            ).view(env_count)
            self.force_target_wrist_ext[new_selected_env_ids_ext, 1] = torch_rand_float(
                min_force_ext, max_force_ext, (env_count, 1), device=self.device
            ).view(env_count)
            self.force_target_wrist_ext[new_selected_env_ids_ext, 2] = torch_rand_float(
                min_force_ext, max_force_ext, (env_count, 1), device=self.device
            ).view(env_count)
            self.force_target_wrist_ext[new_selected_env_ids_ext, 3] = torch_rand_float(
                min_force_ext, max_force_ext, (env_count, 1), device=self.device
            ).view(env_count)
            self.force_target_wrist_ext[new_selected_env_ids_ext, 4] = torch_rand_float(
                min_force_ext, max_force_ext, (env_count, 1), device=self.device
            ).view(env_count)
            self.force_target_wrist_ext[new_selected_env_ids_ext, 5] = torch_rand_float(
                min_force_ext, max_force_ext, (env_count, 1), device=self.device
            ).view(env_count)

            push_duration_wrist_ext = torch_rand_float(
                self.push_duration_wrist_ext_min,
                self.push_duration_wrist_ext_max,
                (env_count, 1),
                device=self.device,
            ).view(env_count)
            max_duration = (
                self.push_interval_wrist_ext[new_selected_env_ids_ext, 0] - self.settling_time_force_wrist
            ) / 2
            max_duration = torch.clamp(max_duration, min=1.0)
            push_duration_wrist_ext = torch.clamp(push_duration_wrist_ext, min=1.0)
            push_duration_wrist_ext = torch.minimum(push_duration_wrist_ext, max_duration)

            self.push_end_time_wrist_ext[new_selected_env_ids_ext] = (
                self.episode_length_buf[new_selected_env_ids_ext] + push_duration_wrist_ext
            )
            self.push_duration_wrist_ext[new_selected_env_ids_ext] = push_duration_wrist_ext
            self.selected_env_ids_wrist_ext[new_selected_env_ids_ext] = 1

        if self.episode_length_buf[self.selected_env_ids_wrist_ext == 1].nelement() > 0:
            subset_env_ids_selected = env_ids_all[self.selected_env_ids_wrist_ext == 1]

            env_ids_apply_push_step1 = subset_env_ids_selected[
                self.episode_length_buf[self.selected_env_ids_wrist_ext == 1]
                < (self.push_end_time_wrist_ext[self.selected_env_ids_wrist_ext == 1]).type(torch.int32)
            ]
            if env_ids_apply_push_step1.nelement() > 0:
                push_duration_reshaped = self.push_duration_wrist_ext[env_ids_apply_push_step1].unsqueeze(-1).clamp(min=1.0)
                ramp = torch.clamp(
                    self.episode_length_buf[env_ids_apply_push_step1].unsqueeze(-1)
                    - (self.push_end_time_wrist_ext[env_ids_apply_push_step1].unsqueeze(-1) - push_duration_reshaped),
                    torch.zeros_like(push_duration_reshaped),
                    push_duration_reshaped,
                )
                left_forces = (self.force_target_wrist_ext[env_ids_apply_push_step1, :3] / push_duration_reshaped) * ramp
                right_forces = (self.force_target_wrist_ext[env_ids_apply_push_step1, 3:6] / push_duration_reshaped) * ramp
                self.forces[env_ids_apply_push_step1.long(), self.left_wrist_handle, :3] = left_forces
                self.forces[env_ids_apply_push_step1.long(), self.right_wrist_handle, :3] = right_forces

            env_ids_apply_push_step2 = subset_env_ids_selected[
                self.episode_length_buf[self.selected_env_ids_wrist_ext == 1]
                > (self.push_end_time_wrist_ext[self.selected_env_ids_wrist_ext == 1] + self.settling_time_force_wrist).type(torch.int32)
            ]
            if env_ids_apply_push_step2.nelement() > 0:
                push_duration_reshaped = self.push_duration_wrist_ext[env_ids_apply_push_step2].unsqueeze(-1).clamp(min=1.0)
                ramp = torch.clamp(
                    self.episode_length_buf[env_ids_apply_push_step2].unsqueeze(-1)
                    - (self.push_end_time_wrist_ext[env_ids_apply_push_step2].unsqueeze(-1) + self.settling_time_force_wrist),
                    torch.zeros_like(push_duration_reshaped),
                    push_duration_reshaped,
                )
                left_forces = self.force_target_wrist_ext[env_ids_apply_push_step2, :3] - (
                    self.force_target_wrist_ext[env_ids_apply_push_step2, :3] / push_duration_reshaped
                ) * ramp
                right_forces = self.force_target_wrist_ext[env_ids_apply_push_step2, 3:6] - (
                    self.force_target_wrist_ext[env_ids_apply_push_step2, 3:6] / push_duration_reshaped
                ) * ramp
                self.forces[env_ids_apply_push_step2.long(), self.left_wrist_handle, :3] = left_forces
                self.forces[env_ids_apply_push_step2.long(), self.right_wrist_handle, :3] = right_forces

            env_ids_to_reset = subset_env_ids_selected[
                self.episode_length_buf[self.selected_env_ids_wrist_ext == 1]
                >= (
                    self.push_end_time_wrist_ext[self.selected_env_ids_wrist_ext == 1]
                    + self.settling_time_force_wrist
                    + self.push_duration_wrist_ext[self.selected_env_ids_wrist_ext == 1]
                ).type(torch.int32)
            ]
            if env_ids_to_reset.nelement() > 0:
                self.selected_env_ids_wrist_ext[env_ids_to_reset] = 0
                self.force_target_wrist_ext[env_ids_to_reset, :6] = 0.
                self.push_end_time_wrist_ext[env_ids_to_reset] = 0.
                self.push_duration_wrist_ext[env_ids_to_reset] = 0.
                self.forces[env_ids_to_reset.long(), self.left_wrist_handle, :3] = 0.0
                self.forces[env_ids_to_reset.long(), self.right_wrist_handle, :3] = 0.0
                reset_count = env_ids_to_reset.shape[0]
                self.push_interval_wrist_ext[env_ids_to_reset, 0] = torch.randint(
                    int(self.cfg.commands.push_wrist_interval_s_ext[0] / self.dt),
                    int(self.cfg.commands.push_wrist_interval_s_ext[1] / self.dt),
                    (reset_count, 1),
                    device=self.device,
                )[:, 0]

        if torch.any(self.freed_envs_wrist_ext):
            freed_ext_ids = torch.nonzero(self.freed_envs_wrist_ext).flatten()
            self.selected_env_ids_wrist_ext[freed_ext_ids] = 0
            self.force_target_wrist_ext[freed_ext_ids, :6] = 0.
            self.push_end_time_wrist_ext[freed_ext_ids] = 0.
            self.push_duration_wrist_ext[freed_ext_ids] = 0.
            self.forces[freed_ext_ids.long(), self.left_wrist_handle, :3] = 0.0
            self.forces[freed_ext_ids.long(), self.right_wrist_handle, :3] = 0.0
            self.freed_envs_wrist_ext[freed_ext_ids] = False
