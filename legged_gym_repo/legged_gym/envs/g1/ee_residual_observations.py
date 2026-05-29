import torch
from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_rotate_inverse
from legged_gym.utils.math import quat_from_euler_xyz

from .ee_residual_math_utils import euler_from_quaternion


class EEResidualObservationsMixin:
    def compute_observations(self):
        """Computes observations"""
        is_standing = torch.norm(self.commands[:, :3], dim=1) < 0.1
        processed_clock_inputs = torch.where(
            is_standing.unsqueeze(1).repeat(1, 2), torch.ones_like(self.clock_inputs), self.clock_inputs
        )
        imu_ang_vel = quat_rotate_inverse(
            self.rigid_body_states[:, self.imu_index, 3:7], self.rigid_body_states[:, self.imu_index, 10:13]
        )
        imu_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index, 3:7], self.gravity_vec)
        torso_imu_ang_vel = quat_rotate_inverse(
            self.rigid_body_states[:, self.torso_imu_index, 3:7], self.rigid_body_states[:, self.torso_imu_index, 10:13]
        )
        torso_imu_projected_vel = quat_rotate_inverse(
            self.rigid_body_states[:, self.torso_imu_index, 3:7], self.gravity_vec
        )
        
        # wrist forces for privileged signals (student cannot observe them directly)
        # yaw-only frame handling
        roll, pitch, yaw = euler_from_quaternion(self.base_quat)
        base_quat_world_indep = quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            yaw.squeeze(-1)  # yaw shape is (num_envs, 1), squeeze to (num_envs,)
        )
        
        # Calculate force residual: ΔF = F_gt - F_hat (in Base-Yaw Frame)
        # Note: F_cmd is NOT considered in supervision signal calculation when use_force_cmd_in_control=False
        # The supervision signal directly compares GT force with observer estimate,
        # without accounting for command force compensation effects.
        # GT forces are in World Frame: self.wrist_forces [num_envs, 2, 3]
        F_gt_left_world = self.wrist_forces[:, 0, :]  # [num_envs, 3]
        F_gt_right_world = self.wrist_forces[:, 1, :]  # [num_envs, 3]
        
        # Observer forces are in World Frame: self.wrist_force_hat_world [num_envs, 6]
        use_observer = hasattr(self, "wrist_force_hat_world") and hasattr(self.cfg, 'observer') and self.cfg.observer.enable
        if use_observer:
            F_hat_left_world = self.wrist_force_hat_world[:, :3]  # [num_envs, 3]
            F_hat_right_world = self.wrist_force_hat_world[:, 3:6]  # [num_envs, 3]
        else:
            # If observer is disabled, use zero (so residual equals GT)
            F_hat_left_world = torch.zeros_like(F_gt_left_world)
            F_hat_right_world = torch.zeros_like(F_gt_right_world)
        
        # Compute residual in World Frame (F_cmd not considered in supervision)
        F_diff_left_world = F_gt_left_world - F_hat_left_world
        F_diff_right_world = F_gt_right_world - F_hat_right_world
        
        # Rotate residual to Base-Yaw Frame (same as teacher signal frame)
        left_wrist_force_local = quat_rotate_inverse(base_quat_world_indep, F_diff_left_world)
        right_wrist_force_local = quat_rotate_inverse(base_quat_world_indep, F_diff_right_world)

        offset = 1 if self.freq_control else 0

        # feet pos command obs
        self.commands[:, 5 + offset : 11 + offset] = quat_rotate_inverse(
            self.base_quat.repeat_interleave(2, dim=0),
            torch.cat(
                [self.feet_pos_target.reshape(-1, 2), torch.zeros(self.num_envs * 2, 1, device=self.device)],
                dim=1,
            )
            - self.root_states[:, 0:3].repeat_interleave(2, dim=0),
        ).reshape(-1, 6)

        # wrist pos command obs
        # Position tasks track FK targets, force tasks track virtual targets
        wrist_world_targets = torch.where(
            self.is_force_ctrl_mode,
            self.virtual_wrist_pos_target,
            self.wrist_pos_target,
        )
        # 1) Recompute the yaw quaternion (or reuse an existing one)
        roll, pitch, yaw = euler_from_quaternion(self.base_quat)
        base_yaw_quat = quat_from_euler_xyz(torch.zeros_like(roll), torch.zeros_like(pitch), yaw)
        base_yaw_quat = base_yaw_quat.squeeze(-2)  # reshape to (num_envs, 4)
        base_yaw_conj = quat_conjugate(base_yaw_quat)
        # 2) Use base_yaw_quat instead of base_quat for observation transform
        self.commands[:, 11 + offset : 17 + offset] = quat_rotate_inverse(
            base_yaw_quat.repeat_interleave(2, dim=0),  # updated transform input
            wrist_world_targets.reshape(-1, 3) - self.root_states[:, 0:3].repeat_interleave(2, dim=0),
        ).reshape(-1, 6)

        if self.wrist_quat_command_slice is not None:
            #self.commands[:, self.wrist_quat_command_slice] = self.wrist_quat_target
            left_curr_quat = self.rigid_body_states[:, self.left_wrist_indices, 3:7].clone().mean(dim=1)
            right_curr_quat = self.rigid_body_states[:, self.right_wrist_indices, 3:7].clone().mean(dim=1)
            left_target_quat = self.wrist_quat_target[:, :4]
            right_target_quat = self.wrist_quat_target[:, 4:]
            # Compute error quaternion: q_err = q_curr^-1 * q_target
            left_error_quat = quat_mul(quat_conjugate(left_curr_quat), left_target_quat)
            right_error_quat = quat_mul(quat_conjugate(right_curr_quat), right_target_quat)
            # q and -q represent the same rotation; force non-negative w for continuity
            left_error_quat = torch.where(left_error_quat[:, 3:4] < 0, -left_error_quat, left_error_quat)
            right_error_quat = torch.where(right_error_quat[:, 3:4] < 0, -right_error_quat, right_error_quat)
            self.commands[:, self.wrist_quat_command_slice] = torch.cat((left_error_quat, right_error_quat), dim=-1)


        # head pos command obs
        self.commands[:, 17 + offset : 20 + offset] = quat_rotate_inverse(
            self.base_quat, self.head_pos_target - self.root_states[:, 0:3]
        )

        lin_command = self.commands[:, :2] * self.commands_scale[:2]
        ang_command = self.commands[:, 2:3] * self.commands_scale[2:3]
        height_command = self.commands[:, 4].unsqueeze(1)
        feet_command = self.commands[:, [5 + offset, 6 + offset, 8 + offset, 9 + offset]]
        wrist_command = self.commands[:, 11 + offset : 17 + offset]
        head_command = self.commands[:, 17 + offset : 20 + offset]
        imu_obs = torch.cat((imu_ang_vel * self.obs_scales.ang_vel, imu_projected_gravity), dim=-1)
        dof_pos_obs = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dof_vel_obs = self.dof_vel * self.obs_scales.dof_vel
        action_obs = (
            self.actions[:, self.lower_dof_indices]
            if self.cfg.model_type == "base"
            else torch.cat([self.actions[:, self.lower_dof_indices], self.actions[:, self.upper_dof_indices]], dim=-1)
        )

        if self.wrist_quat_command_slice is not None:
            wrist_quat_command = self.commands[:, self.wrist_quat_command_slice]
        else:
            wrist_quat_command = torch.zeros(
                self.num_envs, 8, device=self.device
            )

        if self.wrist_force_command_slice is not None:
            wrist_force_command = self.commands[:, self.wrist_force_command_slice]
        else:
            wrist_force_command = torch.zeros(
                self.num_envs, 6, device=self.device
            )

        # DOF residual observation (momentum observer output)
        if hasattr(self, 'observer_r') and hasattr(self.cfg, 'observer') and self.cfg.observer.enable:
            dof_residual_obs = self.observer_r * getattr(self.obs_scales, 'dof_residual', 0.1)
        else:
            dof_residual_obs = torch.zeros(
                self.num_envs, 14, dtype=torch.float, device=self.device
            )

        actor_obs_parts = [
            lin_command,
            ang_command,
            height_command,
            feet_command,
            wrist_command,
            wrist_quat_command,
            wrist_force_command,
            head_command,
            imu_obs,
            dof_pos_obs,
            dof_vel_obs,
            action_obs,
            dof_residual_obs,
        ]
        if self.freq_control:
            actor_obs_parts.append(processed_clock_inputs)
        current_obs = torch.cat(actor_obs_parts, dim=-1)
        current_actor_obs = torch.clone(current_obs)

        # --- Feet-command observation delay (sim2real: SLAM latency) ---
        # Only the actor sees the delayed feet_pose_command; critic keeps ground truth.
        if self.obs_delay_feet_enable and self.obs_delay_feet_max > 0:
            fs, fe = self.actor_obs_group_offsets["feet_pose_command"]
            self.feet_cmd_history[:, self.feet_cmd_ptr] = current_obs[:, fs:fe]
            self.feet_cmd_ptr = (self.feet_cmd_ptr + 1) % (self.obs_delay_feet_max + 1)
            delays = torch.randint(
                self.obs_delay_feet_min, self.obs_delay_feet_max + 1,
                (self.num_envs,), device=self.device,
            )
            read_idx = (self.feet_cmd_ptr - 1 - delays) % (self.obs_delay_feet_max + 1)
            current_actor_obs[:, fs:fe] = self.feet_cmd_history[
                torch.arange(self.num_envs, device=self.device), read_idx
            ]

        # Predict original wrist position (relative to base), not virtual target
        # Yaw-only handling, aligned with B2Z1
        roll, pitch, yaw = euler_from_quaternion(self.base_quat)
        base_quat_world_indep = quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            yaw.squeeze(-1)  # yaw shape is (num_envs, 1), squeeze to (num_envs,)
        )
        wrist_pos_rel = quat_rotate_inverse(
            base_quat_world_indep.repeat_interleave(2, dim=0),
            self.wrist_pos.reshape(-1, 3) - self.root_states[:, 0:3].repeat_interleave(2, dim=0),
        ).reshape(-1, 6)

        # Virtual target position in base frame (to track)
        wrist_virtual_target_rel = quat_rotate_inverse(
            base_quat_world_indep.repeat_interleave(2, dim=0),
            self.virtual_wrist_pos_target.reshape(-1, 3) - self.root_states[:, 0:3].repeat_interleave(2, dim=0),
        ).reshape(-1, 6)
        
        # Virtual target tracking error (what critic needs to see)
        wrist_virtual_error = wrist_virtual_target_rel - wrist_pos_rel

        privileged_terms = torch.cat(
            (
                left_wrist_force_local * self.obs_scales.wrist_force,
                right_wrist_force_local * self.obs_scales.wrist_force,
                wrist_pos_rel,
                wrist_virtual_error * self.obs_scales.wrist_virtual_target,
            ),
            dim=-1,
        )

        current_privileged_obs = torch.cat(
            (current_obs, privileged_terms, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1
        )
        if not hasattr(self, "priv_obs_shapes_logged"):
            print(
                f"current_privileged_dim={current_privileged_obs.shape[-1]}, config_privileged={self.num_one_step_privileged_obs}"
            )
            self.priv_obs_shapes_logged = True

        # Build hierarchical observation masks following UniFP-style gating
        obs_mask = torch.ones((self.num_envs, len(self.one_step_obs_dims)), device=self.device, dtype=torch.bool)
        obs_group_indices = {name: idx for idx, name in enumerate(self.one_step_obs_dims.keys())}
        # Derive mode flags from buffers
        is_walking    = self.is_walking_mode.squeeze(1)
        is_operation  = (self.is_pos_ctrl_mode | self.is_force_ctrl_mode).squeeze(1)
        is_force_ctrl = self.is_force_ctrl_mode.squeeze(1)
        all_true  = torch.ones(self.num_envs,  dtype=torch.bool, device=self.device)
        all_false = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Three-mode observation truth table (dictionary form)
        MODE_OBS_MASK = {
            "lin_command":         is_walking,
            "ang_command":         is_walking,
            "height_command":      is_operation,
            "feet_pose_command":   is_operation,
            "wrist_pose_command":  all_true,        # enabled in all three modes
            "head_pose_command":   all_false,       # permanently disabled
            "wrist_force_command": is_force_ctrl,
            "dof_residual":        is_force_ctrl,
        }

        # Extra privileged terms (force-control infrastructure signals)
        MODE_PRIVILEGED_EXTRA = {
            "wrist_forces":          is_force_ctrl,
            "wrist_virtual_target":  is_force_ctrl,
        }

        # Apply to actor observation mask
        for name, mask in MODE_OBS_MASK.items():
            if name in obs_group_indices:
                obs_mask[:, obs_group_indices[name]] = mask

        # upper_policy_mask is already set to True in _resample_commands; do not overwrite here

        self.obs_mask_buf = torch.cat(
            [
                self.obs_mask_buf[:, len(self.one_step_obs_dims) :],
                obs_mask.to(self.obs_mask_buf.dtype),
            ],
            dim=-1,
        )

        if self.add_noise:
            noise_vec = self.noise_scale_vec
            if noise_vec.shape[-1] != current_actor_obs.shape[-1]:
                noise_vec = noise_vec[: current_actor_obs.shape[-1]]
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * noise_vec
        self.obs_buf = torch.cat(
            (
                self.obs_buf[:, self.num_one_step_obs : self.actor_proprioceptive_obs_length],
                current_actor_obs[:, : self.num_one_step_obs],
            ),
            dim=-1,
        )       # drop oldest step observation and append current observation
        self.privileged_obs_buf = torch.cat(
            (
                self.privileged_obs_buf[:, self.num_one_step_privileged_obs : self.critic_proprioceptive_obs_length],
                current_privileged_obs,
            ),
            dim=-1,
        )

        privileged_obs_mask = torch.ones(
            (self.num_envs, len(self.one_step_privileged_obs_dims)), device=self.device, dtype=torch.bool
        )
        privileged_group_indices = {name: idx for idx, name in enumerate(self.one_step_privileged_obs_dims.keys())}

        # Apply to privileged observation mask (share base dictionary with actor; add extra terms separately)
        for name, mask in {**MODE_OBS_MASK, **MODE_PRIVILEGED_EXTRA}.items():
            if name in privileged_group_indices:
                privileged_obs_mask[:, privileged_group_indices[name]] = mask

        self.privileged_obs_mask_buf = torch.cat(
            [
                self.privileged_obs_mask_buf[:, len(self.one_step_privileged_obs_dims) :],
                privileged_obs_mask.to(self.privileged_obs_mask_buf.dtype),
            ],
            dim=-1,
        )

        if (
            self.cfg.terrain.measure_heights
            and self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]
            and self.num_height_dim > 0
        ):
            heights = (
                torch.clip(self.root_states[:, 2].unsqueeze(1) - 1.0 - self.measured_heights, -1, 1.0)
                * self.obs_scales.height_measurements
            )
            # self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
            if self.add_noise:
                heights = self._add_height_noise(heights)
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

    def compute_termination_observations(self, env_ids):
        is_standing = torch.norm(self.commands[:, :3], dim=1) < 0.1
        processed_clock_inputs = torch.where(
            is_standing.unsqueeze(1).repeat(1, 2), torch.ones_like(self.clock_inputs), self.clock_inputs
        )
        """Computes observations"""
        imu_ang_vel = quat_rotate_inverse(
            self.rigid_body_states[:, self.imu_index, 3:7], self.rigid_body_states[:, self.imu_index, 10:13]
        )
        imu_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index, 3:7], self.gravity_vec)
        torso_imu_ang_vel = quat_rotate_inverse(
            self.rigid_body_states[:, self.torso_imu_index, 3:7], self.rigid_body_states[:, self.torso_imu_index, 10:13]
        )
        torso_imu_projected_vel = quat_rotate_inverse(
            self.rigid_body_states[:, self.torso_imu_index, 3:7], self.gravity_vec
        )

        # Yaw-only handling, aligned with B2Z1
        roll, pitch, yaw = euler_from_quaternion(self.base_quat)
        base_quat_world_indep = quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            yaw.squeeze(-1)  # yaw shape is (num_envs, 1), squeeze to (num_envs,)
        )
        left_wrist_force_local = quat_rotate_inverse(base_quat_world_indep, self.wrist_forces[:, 0, :])
        right_wrist_force_local = quat_rotate_inverse(base_quat_world_indep, self.wrist_forces[:, 1, :])

        offset = 1 if self.freq_control else 0

        self.commands[:, 5 + offset : 11 + offset] = quat_rotate_inverse(
            self.base_quat.repeat_interleave(2, dim=0),
            torch.cat(
                [self.feet_pos_target.reshape(-1, 2), torch.zeros(self.num_envs * 2, 1, device=self.device)], dim=1
            )
            - self.root_states[:, 0:3].repeat_interleave(2, dim=0),
        ).reshape(-1, 6)

        self.commands[:, 11 + offset : 17 + offset] = quat_rotate_inverse(
            self.base_quat.repeat_interleave(2, dim=0),
            self.wrist_pos_target.reshape(-1, 3) - self.root_states[:, 0:3].repeat_interleave(2, dim=0),
        ).reshape(-1, 6)

        if self.wrist_quat_command_slice is not None:
            self.commands[:, self.wrist_quat_command_slice] = self.wrist_quat_target

        self.commands[:, 17 + offset : 20 + offset] = quat_rotate_inverse(
            self.base_quat, self.head_pos_target - self.root_states[:, 0:3]
        )

        lin_command = self.commands[:, :2] * self.commands_scale[:2]
        ang_command = self.commands[:, 2:3] * self.commands_scale[2:3]
        height_command = self.commands[:, 4].unsqueeze(1)
        feet_command = self.commands[:, [5 + offset, 6 + offset, 8 + offset, 9 + offset]]
        wrist_command = self.commands[:, 11 + offset : 17 + offset]
        head_command = self.commands[:, 17 + offset : 20 + offset]
        imu_obs = torch.cat((imu_ang_vel * self.obs_scales.ang_vel, imu_projected_gravity), dim=-1)
        dof_pos_obs = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dof_vel_obs = self.dof_vel * self.obs_scales.dof_vel
        action_obs = (
            self.actions[:, self.lower_dof_indices]
            if self.cfg.model_type == "base"
            else torch.cat([self.actions[:, self.lower_dof_indices], self.actions[:, self.upper_dof_indices]], dim=-1)
        )

        if self.wrist_quat_command_slice is not None:
            wrist_quat_command = self.commands[:, self.wrist_quat_command_slice]
        else:
            wrist_quat_command = torch.zeros(
                self.num_envs, 8, device=self.device
            )

        if self.wrist_force_command_slice is not None:
            wrist_force_command = self.commands[:, self.wrist_force_command_slice]
        else:
            wrist_force_command = torch.zeros(
                self.num_envs, 6, device=self.device
            )

        # DOF residual observation (momentum observer output)
        if hasattr(self, 'observer_r') and hasattr(self.cfg, 'observer') and self.cfg.observer.enable:
            dof_residual_obs = self.observer_r * getattr(self.obs_scales, 'dof_residual', 0.1)
        else:
            dof_residual_obs = torch.zeros(
                self.num_envs, 14, dtype=torch.float, device=self.device
            )

        actor_obs_parts = [
            lin_command,
            ang_command,
            height_command,
            feet_command,
            wrist_command,
            wrist_quat_command,
            wrist_force_command,
            head_command,
            imu_obs,
            dof_pos_obs,
            dof_vel_obs,
            action_obs,
            dof_residual_obs,
        ]
        if self.freq_control:
            actor_obs_parts.append(processed_clock_inputs)
        current_obs = torch.cat(actor_obs_parts, dim=-1)

        # Predict original wrist position (relative to base), not virtual target
        # Yaw-only handling, aligned with B2Z1
        roll, pitch, yaw = euler_from_quaternion(self.base_quat)
        base_quat_world_indep = quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            yaw.squeeze(-1)  # yaw shape is (num_envs, 1), squeeze to (num_envs,)
        )
        wrist_pos_rel = quat_rotate_inverse(
            base_quat_world_indep.repeat_interleave(2, dim=0),
            self.wrist_pos.reshape(-1, 3) - self.root_states[:, 0:3].repeat_interleave(2, dim=0),
        ).reshape(-1, 6)

        # Virtual target position in base frame (to track)
        wrist_virtual_target_rel = quat_rotate_inverse(
            base_quat_world_indep.repeat_interleave(2, dim=0),
            self.virtual_wrist_pos_target.reshape(-1, 3) - self.root_states[:, 0:3].repeat_interleave(2, dim=0),
        ).reshape(-1, 6)
        
        # Virtual target tracking error (what critic needs to see)
        wrist_virtual_error = wrist_virtual_target_rel - wrist_pos_rel

        privileged_terms = torch.cat(
            (
                left_wrist_force_local * self.obs_scales.wrist_force,
                right_wrist_force_local * self.obs_scales.wrist_force,
                wrist_pos_rel,
                wrist_virtual_error * self.obs_scales.wrist_virtual_target,
            ),
            dim=-1,
        )

        if self.add_noise:
            noise_vec = self.noise_scale_vec
            if noise_vec.shape[-1] != current_obs.shape[-1]:
                noise_vec = noise_vec[: current_obs.shape[-1]]
            current_obs = current_obs + (2 * torch.rand_like(current_obs) - 1) * noise_vec

        current_privileged_obs = torch.cat(
            (current_obs, privileged_terms, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1
        )
        return torch.cat(
            (
                self.privileged_obs_buf[:, self.num_one_step_privileged_obs : self.critic_proprioceptive_obs_length],
                current_privileged_obs,
            ),
            dim=-1,
        )[env_ids]

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """

        # Use actual observation dimension of current residual model
        actual_obs_dim = self.num_one_step_obs  # use current model observation dim instead of base model
        noise_vec = torch.zeros(actual_obs_dim, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        # Recompute section indices
        env_obs_dims = self.cfg.env.one_step_obs_dims
        cmd_groups = [
            "lin_command",
            "ang_command",
            "height_command",
            "feet_pose_command",
            "wrist_pose_command",
            "wrist_force_command",
            "head_pose_command",
        ]
        cmd_end = sum(env_obs_dims[name] for name in cmd_groups)
        imu_start = cmd_end
        imu_end = imu_start + env_obs_dims["imu_info"]
        dof_pos_start = imu_end
        dof_pos_end = dof_pos_start + env_obs_dims["dof_pos"]
        dof_vel_start = dof_pos_end
        dof_vel_end = dof_vel_start + env_obs_dims["dof_vel"]
        action_start = dof_vel_end
        action_end = action_start + env_obs_dims["action_actual"]

        # Set noise scales
        noise_vec[0:cmd_end] = 0.0  # commands (no noise)
        noise_vec[imu_start:imu_start + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel  # IMU angular velocity
        noise_vec[imu_start + 3:imu_end] = noise_scales.gravity * noise_level  # IMU gravity
        noise_vec[dof_pos_start:dof_pos_end] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[dof_vel_start:dof_vel_end] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[action_start:action_end] = 0.0  # previous actions (no noise)
        return noise_vec
