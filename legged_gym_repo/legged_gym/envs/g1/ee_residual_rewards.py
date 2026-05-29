import torch
from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_rotate_inverse
from legged_gym.utils.math import quat_from_euler_xyz

from .ee_residual_math_utils import euler_from_quaternion


class EEResidualRewardsMixin:
    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    def _get_feet_heights(self, env_ids=None):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices, :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices, :3].clone()
        if self.cfg.terrain.mesh_type == "plane":
            left_foot_height = torch.mean(left_foot_pos[:, :, 2], dim=-1, keepdim=True)
            left_foot_height_var = torch.var(left_foot_pos[:, :, 2], dim=-1, keepdim=True)
            right_foot_height = torch.mean(right_foot_pos[:, :, 2], dim=-1, keepdim=True)
            right_foot_height_var = torch.var(right_foot_pos[:, :, 2], dim=-1, keepdim=True)
            return (
                torch.cat((left_foot_height, right_foot_height), dim=-1),
                torch.cat((left_foot_height_var, right_foot_height_var), dim=-1),
                torch.cat((left_foot_height, right_foot_height), dim=-1),
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            left_points = left_foot_pos[env_ids].clone()
            right_points = right_foot_pos[env_ids].clone()
        else:
            left_points = left_foot_pos.clone()
            right_points = right_foot_pos.clone()

        left_points += self.terrain.cfg.border_size
        right_points += self.terrain.cfg.border_size
        left_points = (left_points / self.terrain.cfg.horizontal_scale).long()
        right_points = (right_points / self.terrain.cfg.horizontal_scale).long()
        left_px = left_points[:, :, 0].view(-1)
        right_px = right_points[:, :, 0].view(-1)
        left_py = left_points[:, :, 1].view(-1)
        right_py = right_points[:, :, 1].view(-1)
        left_px = torch.clip(left_px, 0, self.height_samples.shape[0] - 2)
        right_px = torch.clip(right_px, 0, self.height_samples.shape[0] - 2)
        left_py = torch.clip(left_py, 0, self.height_samples.shape[1] - 2)
        right_py = torch.clip(right_py, 0, self.height_samples.shape[1] - 2)

        left_heights1 = self.height_samples[left_px, left_py]
        left_heights2 = self.height_samples[left_px + 1, left_py]
        left_heights3 = self.height_samples[left_px, left_py + 1]
        left_heights = torch.min(left_heights1, left_heights2)
        left_heights = torch.min(left_heights, left_heights3)
        left_heights = left_heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        left_foot_heights = left_foot_pos[:, :, 2] - left_heights

        right_heights1 = self.height_samples[right_px, right_py]
        right_heights2 = self.height_samples[right_px + 1, right_py]
        right_heights3 = self.height_samples[right_px, right_py + 1]
        right_heights = torch.min(right_heights1, right_heights2)
        right_heights = torch.min(right_heights, right_heights3)
        right_heights = right_heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        right_foot_heights = right_foot_pos[:, :, 2] - right_heights

        feet_heights = torch.cat(
            (torch.mean(left_foot_heights, dim=-1, keepdim=True), torch.mean(right_foot_heights, dim=-1, keepdim=True)),
            dim=-1,
        )
        feet_heights_var = torch.cat(
            (torch.var(left_foot_heights, dim=-1, keepdim=True), torch.var(right_foot_heights, dim=-1, keepdim=True)),
            dim=-1,
        )

        return torch.clip(feet_heights, min=0.0), feet_heights_var, feet_heights

    def _get_tracking_sigma(self, term: str) -> float:
        """Return the configured sigma for a given tracking term.
        Supports several naming conventions used in configs:
        - per-term attribute like `tracking_wrist_sigma`
        - dict `tracking_sigma[term]`
        - scalar `tracking_sigma` (fallback)
        """
        # wrist special-case: config provides `tracking_wrist_sigma`
        if term.startswith("tracking_wrist") and hasattr(self.cfg.rewards, "tracking_wrist_sigma"):
            return float(self.cfg.rewards.tracking_wrist_sigma)

        ts = getattr(self.cfg.rewards, "tracking_sigma", None)
        try:
            if isinstance(ts, dict):
                return float(ts.get(term, list(ts.values())[0]))
        except Exception:
            pass

        # fallback to scalar tracking_sigma
        if ts is not None:
            return float(ts)

        # as a last resort, look for a direct attribute matching the term with _sigma suffix
        attr = term.replace("tracking_", "")
        attr_name = f"{attr}_sigma"
        if hasattr(self.cfg.rewards, attr_name):
            return float(getattr(self.cfg.rewards, attr_name))

        # default fallback
        return 1.0

    def _init_tracking_error_logging(self):
        """Initialize tracking error logging for monitoring (no adaptive adjustment).
        Only records error statistics for logging purposes.
        """
        # Initialize EMA for tracking error logging (alpha for EMA smoothing)
        alpha = 0.001  # EMA coefficient for error tracking
        self._tracking_error_ema = {}
        self._tracking_error_alpha = alpha
        
        # Initialize error EMA for commonly tracked terms
        tracked_terms = ["tracking_wrist_pos"]
        for term in tracked_terms:
            self._tracking_error_ema[term] = 0.0

    def _log_tracking_error(self, error: torch.Tensor, term: str):
        """Log tracking error statistics (for monitoring only, no sigma adjustment).
        
        Args:
            error: Tracking error tensor [num_envs]
            term: Name of the tracking term (e.g., "tracking_wrist_pos")
        """
        if not hasattr(self, "_tracking_error_ema") or term not in self._tracking_error_ema:
            return
        
        # Update EMA of error for logging (mean across envs)
        mean_error = float(error.mean().item())
        self._tracking_error_ema[term] = (
            self._tracking_error_ema[term] * (1 - self._tracking_error_alpha) 
            + mean_error * self._tracking_error_alpha
        )

    def _reward_tracking_x_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :1] - self.base_lin_vel[:, :1]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_y_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, 1:2] - self.base_lin_vel[:, 1:2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square((self.commands[:, 2] - self.torso_ang_vel[:, 2]) / torch.pi)
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel_int(self):
        err = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        self.base_err_int[:, 2] = (1 - self.cfg.rewards.beta_int) * self.base_err_int[
            :, 2
        ] + self.cfg.rewards.beta_int * err * self.dt
        return torch.exp(-self.base_err_int[:, 2] / self.cfg.rewards.tracking_sigma_int)

    def _reward_tracking_feet_pos(self):
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices, :2].clone().mean(dim=1)
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices, :2].clone().mean(dim=1)

        left_error = torch.norm(left_foot_pos - self.feet_pos_target[:, :2], dim=1)
        right_error = torch.norm(right_foot_pos - self.feet_pos_target[:, 2:4], dim=1)

        deadzone = getattr(self.cfg.rewards, 'feet_pos_deadzone', 0.0)
        feet_error = torch.clamp(left_error - deadzone, min=0.0) + torch.clamp(right_error - deadzone, min=0.0)

        return torch.exp(-feet_error / self.cfg.rewards.tracking_sigma) * (self.is_pos_ctrl_mode | self.is_force_ctrl_mode).squeeze(1)

    def _reward_tracking_wrist_pos(self):
        """Reward for tracking wrist position target computed from FK.
        The target is updated every step from random upper actions.
        """
        left_wrist_pos = self.rigid_body_states[:, self.left_wrist_indices, :3].clone().mean(dim=1)
        right_wrist_pos = self.rigid_body_states[:, self.right_wrist_indices, :3].clone().mean(dim=1)

        # Compute error between current wrist position and FK-computed target
        wrist_error = torch.norm(left_wrist_pos - self.wrist_pos_target[:, :3], dim=1) + torch.norm(
            right_wrist_pos - self.wrist_pos_target[:, 3:6], dim=1
        )
        try:
            self._log_tracking_error(wrist_error, "tracking_wrist_pos")
        except Exception:
            pass

        walk_scale = getattr(self.cfg.rewards, 'walk_wrist_scale', 0.2)
        is_walking = self.is_walking_mode.squeeze(1)
        is_force_ctrl = self.is_force_ctrl_mode.squeeze(1)
        mode_scale = torch.where(is_walking, torch.full_like(wrist_error, walk_scale),
                     torch.where(is_force_ctrl, torch.zeros_like(wrist_error),
                                 torch.ones_like(wrist_error)))
        return torch.exp(-wrist_error / (self.cfg.rewards.tracking_wrist_sigma)) * mode_scale

    def _reward_tracking_wrist_joints(self):
        """Joint-space wrist tracking: only wrist_roll/pitch/yaw joints drive this reward.

        Compared to the old task-space quaternion tracking, this formulation
        cannot be "cheated" by shoulder/elbow movement, eliminating the
        conflict with wrist position tracking.
        """
        actual = self.dof_pos[:, self.wrist_joint_dof_indices]  # [N, 6]
        target = self.wrist_joint_target  # [N, 6]
        error = torch.sum((actual - target) ** 2, dim=-1)  # [N]
        sigma = getattr(self.cfg.rewards, "tracking_wrist_joints_sigma",
                        getattr(self.cfg.rewards, "tracking_wrist_sigma", self.cfg.rewards.tracking_sigma))
        reward = torch.exp(-error / sigma)

        walk_scale = getattr(self.cfg.rewards, 'walk_wrist_scale', 0.2)
        is_walking = self.is_walking_mode.squeeze(1)
        return reward * torch.where(is_walking, walk_scale, 1.0)

    def _reward_tracking_elbow_pos(self):
        """Reward for tracking elbow position target computed from FK.
        The target is updated every step from random upper actions.
        Similar to _reward_tracking_wrist_pos but for elbow position.
        """
        left_elbow_pos = self.rigid_body_states[:, self.left_elbow_index, :3].clone()  # [N, 3]
        right_elbow_pos = self.rigid_body_states[:, self.right_elbow_index, :3].clone()  # [N, 3]

        # Compute error between current elbow position and FK-computed target
        elbow_error = torch.norm(left_elbow_pos - self.elbow_pos_target[:, :3], dim=1) + torch.norm(
            right_elbow_pos - self.elbow_pos_target[:, 3:6], dim=1
        )
        
        sigma = getattr(self.cfg.rewards, "tracking_wrist_sigma", self.cfg.rewards.tracking_sigma)
        reward = torch.exp(-elbow_error / sigma)

        walk_scale = getattr(self.cfg.rewards, 'walk_wrist_scale', 0.2)
        is_walking = self.is_walking_mode.squeeze(1)
        return reward * torch.where(is_walking, walk_scale, 1.0)

    def _reward_tracking_head_pos(self):
        head_pos = self.rigid_body_states[:, self.head_index, :3].clone()

        head_error = torch.norm(head_pos - self.head_pos_target, dim=1)
        return torch.exp(-head_error / self.cfg.rewards.tracking_sigma) * (self.is_head_pos).squeeze(1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        is_wrist_mode = (self.is_pos_ctrl_mode | self.is_force_ctrl_mode).squeeze()
        return torch.square(self.base_lin_vel[:, 2]) * (
            ((torch.norm(self.commands[:, :3], dim=1) > 0.1) | self.is_feet_pos.squeeze())
            & ~(self.is_head_pos.squeeze() | is_wrist_mode)
        )

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """
        Penalize non flat base orientation.
        """
        base_penalty = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        
        
        
        return base_penalty

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_tracking_base_height(self):
        base_height_l = self.root_states[:, 2] - self.feet_pos[:, 0, 2]
        base_height_r = self.root_states[:, 2] - self.feet_pos[:, 1, 2]
        base_height = torch.max(base_height_l, base_height_r)
        
        if self.cfg.rewards.tracking_base_height_judge_return0: # judge and return 0
            height_error = torch.abs(base_height - self.commands[:, 4] + self.cfg.asset.ankle_sole_distance)
            return torch.exp(-height_error * 10) * ~(
                self.is_head_pos.squeeze()
            )
        else:
            height_error = torch.abs(base_height - self.commands[:, 4] + self.cfg.asset.ankle_sole_distance) * ~(
                self.is_head_pos.squeeze()
            )
            return torch.exp(-height_error * 10)

    def _reward_deviation_hip_joint(self):
        return torch.sum(
            torch.square(self.dof_pos - self.default_dof_pos)[:, self.hip_joint_indices], dim=-1
        ) * (torch.norm(self.commands[:, :3], dim=1) > 0.1)

    def _reward_deviation_ankle_joint(self):
        is_wrist_mode = (self.is_pos_ctrl_mode | self.is_force_ctrl_mode).squeeze(1)
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.ankle_joint_indices], dim=-1) * (
            ((torch.norm(self.commands[:, :3], dim=1) > 0.1) | self.is_feet_pos.squeeze(1))
            & ~(self.is_head_pos.squeeze(1) | is_wrist_mode)
        )

    def _reward_deviation_knee_joint(self):
        height_error = (
            self.root_states[:, 2] - self.commands[:, 4]
        )  # / (torch.abs(self.root_states[:, 2] - self.commands[:, 4]) + 1e-6)
        knee_action_min = (
            self.default_dof_pos[:, self.knee_joint_indices]
            + self.cfg.control.action_scale * self.action_min[:, self.knee_joint_indices]
        )
        knee_action_max = (
            self.default_dof_pos[:, self.knee_joint_indices]
            + self.cfg.control.action_scale * self.action_max[:, self.knee_joint_indices]
        )
        joint_deviation = (self.dof_pos[:, self.knee_joint_indices] - knee_action_min) / (
            knee_action_max - knee_action_min
        )  # always positive
        # height_error < 0, dof pos lower; else, dof pos larger
        return torch.sum(torch.abs((joint_deviation - 0.5) * height_error.unsqueeze(-1)), dim=-1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0])[:, : self.num_actions].clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1])[:, : self.num_actions].clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * self.first_contacts, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= (torch.norm(self.commands[:, :3], dim=1) > 0.1)
        return rew_airTime

    def _reward_feet_clearance(self):
        cur_feetvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        feetvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            feetvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_feetvel_translated[:, i, :])
        feet_height, feet_height_var, _ = self._get_feet_heights()
        if self.freq_control:
            phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
            height_target = self.cfg.rewards.clearance_height_target * phases
            height_error = torch.square(feet_height - height_target).view(self.num_envs, -1)
            assert False, "Not tested yet"

        else:
            height_error = torch.square(feet_height - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
            feet_leteral_vel = torch.sqrt(torch.sum(torch.square(feetvel_in_body_frame[:, :, :2]), dim=2)).view(
                self.num_envs, -1
            )
            return torch.sum(height_error * feet_leteral_vel, dim=1) * (
                torch.clip((torch.norm(self.commands[:, :3], dim=1) - 0.1) / 0.2, min=0.0, max=1.0)
                + 0.5 * (self.is_feet_pos.squeeze() | self.is_head_pos.squeeze())
            )

    def _reward_feet_distance_lateral(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        foot_leteral_dis = torch.abs(footpos_in_body_frame[:, 0, 1] - footpos_in_body_frame[:, 1, 1])
        return torch.clamp(foot_leteral_dis - self.cfg.rewards.least_feet_distance_lateral, max=0) + torch.clamp(
            -foot_leteral_dis + self.cfg.rewards.most_feet_distance_lateral, max=0
        )

    def _reward_knee_distance_lateral(self):
        cur_knee_pos_translated = self.rigid_body_states[:, self.knee_indices, :3].clone() - self.root_states[
            :, 0:3
        ].unsqueeze(1)
        knee_pos_in_body_frame = torch.zeros(self.num_envs, len(self.knee_indices), 3, device=self.device)
        for i in range(len(self.knee_indices)):
            knee_pos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_knee_pos_translated[:, i, :])
        knee_lateral_dis = torch.abs(knee_pos_in_body_frame[:, 0, 1] - knee_pos_in_body_frame[:, 2, 1]) + torch.abs(
            knee_pos_in_body_frame[:, 1, 1] - knee_pos_in_body_frame[:, 3, 1]
        )
        return torch.clamp(knee_lateral_dis - self.cfg.rewards.least_knee_distance_lateral * 2, max=0) + torch.clamp(
            -knee_lateral_dis + self.cfg.rewards.most_knee_distance_lateral * 2, max=0
        )

    def _reward_feet_ground_parallel(self):
        feet_heights, feet_heights_var, _ = self._get_feet_heights()
        continue_contact = (self.feet_air_time >= 3 * self.dt) * self.contact_filt
        return torch.sum(feet_heights_var * continue_contact, dim=1)

    def _reward_feet_parallel(self):
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices[0:3], :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices[0:3], :3].clone()
        feet_distances = torch.norm(left_foot_pos - right_foot_pos, dim=2)
        feet_distances_var = torch.var(feet_distances, dim=1)
        return feet_distances_var

    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(
            torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1
        )

    def _reward_joint_power(self):
        # Penalize high power
        is_wrist_mode = (self.is_pos_ctrl_mode | self.is_force_ctrl_mode).squeeze(1)
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1) / torch.clip(
            torch.sum(torch.square(self.commands[:, 0:2]), dim=-1)
            + 0.2 * torch.square(self.commands[:, 2])
            + 0.2 * (self.is_feet_pos.squeeze(1) | self.is_head_pos.squeeze(1) | is_wrist_mode),
            min=0.1,
        )

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square((self.torques / self.p_gains.unsqueeze(0))[:, self.lower_dof_indices]), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel[:, self.lower_dof_indices]), dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit)[
                :, self.lower_dof_indices
            ].clip(min=0.0),
            dim=1,
        )

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit)[
                :, self.lower_dof_indices
            ].clip(min=0.0),
            dim=1,
        )

    def _reward_no_fly(self):
        is_wrist_mode = (self.is_pos_ctrl_mode | self.is_force_ctrl_mode).squeeze(1)
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.5
        single_contact = torch.sum(1.0 * contacts, dim=1) == 1
        rew_no_fly = 1.0 * single_contact
        rew_no_fly = torch.max(
            torch.max(
                rew_no_fly,
                1.0
                * (
                    (
                        (torch.norm(self.commands[:, :3], dim=1) < 0.1)
                        & ~(self.is_feet_pos.squeeze(1) | self.is_head_pos.squeeze(1))
                    )
                    | is_wrist_mode
                ),
            ),
            (self.base_lin_vel[:, 0] > 1.2) * 1.0,
        )  # full reward for zero command
        return rew_no_fly

    def _reward_penalize_pelvis_ang_vel(self):
        is_wrist_mode = (self.is_pos_ctrl_mode | self.is_force_ctrl_mode).squeeze(1)
        ang_vel = self.torso_ang_vel.clone()
        ang_vel[:, 2] = ang_vel[:, 2] - self.commands[:, 2]
        return torch.sum(torch.square(ang_vel), dim=-1) * (
            (torch.norm(self.commands[:, :3], dim=1) > 0.1)
            | self.is_feet_pos.squeeze(1)
            | self.is_head_pos.squeeze(1)
            | is_wrist_mode
        )

    def _reward_joint_tracking_error(self):
        return torch.sum(
            torch.square(self.joint_pos_target[:, self.lower_dof_indices] - self.dof_pos[:, self.lower_dof_indices]),
            dim=-1,
        )

    def _reward_feet_slip(self):
        # Penalize feet slipping
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        return torch.sum(torch.norm(self.feet_vel[:, :, :2], dim=2) * contact, dim=1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_contact_momentum(self):
        # encourage soft contacts
        feet_contact_momentum_z = torch.clip(self.feet_vel[:, :, 2], max=0) * torch.clip(
            self.contact_forces[:, self.feet_indices, 2] - 50, min=0
        )
        return torch.sum(feet_contact_momentum_z, dim=1)

    def _reward_action_vanish(self):
        upper_error = torch.clip(
            self.origin_actions[:, self.lower_dof_indices] - self.action_max[:, self.lower_dof_indices], min=0
        )
        lower_error = torch.clip(
            self.action_min[:, self.lower_dof_indices] - self.origin_actions[:, self.lower_dof_indices], min=0
        )
        return torch.sum(upper_error + lower_error, dim=-1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        is_wrist_mode = (self.is_pos_ctrl_mode | self.is_force_ctrl_mode).squeeze(1)
        contacts = torch.sum(self.contact_forces[:, self.feet_indices, 2] < 0.1, dim=-1)  # no contact
        error_sim = contacts
        return error_sim * (
            (
                (torch.norm(self.commands[:, :3], dim=1) < 0.1)
                & ~(self.is_feet_pos.squeeze(1) | self.is_head_pos.squeeze(1))
            )
            | is_wrist_mode
        )

    def _reward_tracking_contacts_shaped_force(self):
        if not self.freq_control:
            return torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        assert False, "not tested yet"

    def _reward_tracking_contacts_shaped_vel(self):
        if not self.freq_control:
            return torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        assert False, "not tested yet"

    def _reward_waist_action(self):
        """
        """
        if self.fix_waist:
            return torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        
        waist_action = self.dof_pos[:, self.waist_active_joint_indices]
        
        is_wrist_mode = (self.is_pos_ctrl_mode | self.is_force_ctrl_mode).squeeze(1)
        is_head_or_wrist = (self.is_head_pos.squeeze(1) | is_wrist_mode)
        
        penalty_coeff = torch.where(is_head_or_wrist, 20.0, 30.0)
        
        rwd = torch.exp(-penalty_coeff * waist_action.abs()[:, 0]) + torch.exp(-penalty_coeff * waist_action.abs()[:, 1])

        waist_pitch = waist_action[:, 1]
        backward_pitch = torch.clamp(-waist_pitch, min=0.0)
        backward_penalty_coeff = 15.0
        rwd += torch.exp(-backward_penalty_coeff * backward_pitch)
        
        return rwd

    def _reward_imu_stand_still(self):
        """
        Penalizes IMU angular velocity when stationary.
        - Walk mode with zero command: scale × 1.0
        - Manipulation modes (pos_ctrl + force_ctrl): scale × imu_manip_scale
        """
        is_manip = (self.is_pos_ctrl_mode | self.is_force_ctrl_mode).squeeze(1)
        imu_ang_vel = quat_rotate_inverse(
            self.rigid_body_states[:, self.imu_index, 3:7], self.rigid_body_states[:, self.imu_index, 10:13]
        )
        torso_imu_ang_vel = quat_rotate_inverse(
            self.rigid_body_states[:, self.torso_imu_index, 3:7], self.rigid_body_states[:, self.torso_imu_index, 10:13]
        )
        walk_still = (torch.norm(self.commands[:, :3], dim=1) < 0.1) & ~(
            self.is_feet_pos.squeeze(1) | self.is_head_pos.squeeze(1) | is_manip
        )
        imu_ang_vel = torch.mean(torch.square(imu_ang_vel), dim=1)
        torso_imu_ang_vel = torch.mean(torch.square(torso_imu_ang_vel), dim=1)
        imu_penalty = imu_ang_vel + torso_imu_ang_vel

        manip_scale = getattr(self.cfg.rewards, 'imu_manip_scale', 3.0)
        mode_scale = torch.where(is_manip, torch.full_like(imu_penalty, manip_scale), torch.ones_like(imu_penalty))
        active = walk_still | is_manip
        return active * imu_penalty * mode_scale

    def _get_virtual_wrist_tracking_terms(self):
        """Return per-hand wrist errors relative to virtual compliance targets."""
        if not hasattr(self, "virtual_wrist_pos_target") or self.virtual_wrist_pos_target is None:
            return None
        if not hasattr(self, "left_wrist_indices") or not hasattr(self, "right_wrist_indices"):
            return None
        if self.virtual_wrist_pos_target.shape[1] < 3:
            return None

        # Use the same position calculation as _reward_tracking_wrist_pos (mean of multiple indices)
        left_actual = self.rigid_body_states[:, self.left_wrist_indices, :3].clone().mean(dim=1)
        right_actual = self.rigid_body_states[:, self.right_wrist_indices, :3].clone().mean(dim=1)
        
        left_target = self.virtual_wrist_pos_target[:, :3]
        if self.virtual_wrist_pos_target.shape[1] >= 6:
            right_target = self.virtual_wrist_pos_target[:, 3:6]
        else:
            right_target = left_target

        left_error = left_actual - left_target
        right_error = right_actual - right_target

        active_mask = (
            self.is_force_ctrl_mode.squeeze(1).float()
            if hasattr(self, "is_wrist_force")
            else torch.ones(self.num_envs, device=self.device)
        )

        return left_error, right_error, active_mask

    def _reward_tracking_wrist_force_world(self):
        """Reward for matching the virtual (force-compensated) target positions."""
        tracking_terms = self._get_virtual_wrist_tracking_terms()
        if tracking_terms is None:
            return torch.zeros(self.num_envs, device=self.device)

        left_error_vec, right_error_vec, active_mask = tracking_terms
        left_error = torch.norm(left_error_vec, dim=1)
        right_error = torch.norm(right_error_vec, dim=1)
        total_error = left_error + right_error
        sigma = getattr(self.cfg.rewards, "tracking_wrist_sigma", self.cfg.rewards.tracking_sigma)
        reward = torch.exp(-total_error / sigma)

        
        return reward * active_mask

    def _reward_wrist_force_penalty(self):
        """Penalize large virtual-target errors (replaces raw force magnitude penalty)."""
        tracking_terms = self._get_virtual_wrist_tracking_terms()
        if tracking_terms is None:
            return torch.zeros(self.num_envs, device=self.device)

        left_error_vec, right_error_vec, active_mask = tracking_terms
        error_threshold = getattr(self.cfg.rewards, "wrist_error_threshold", 0.05)
        left_err_norm = torch.norm(left_error_vec, dim=1)
        right_err_norm = torch.norm(right_error_vec, dim=1)
        total_error = left_err_norm + right_err_norm
        penalty = torch.clamp(total_error - error_threshold, min=0.0)
        return -penalty * 0.01 * active_mask

    def _reward_force_smoothness(self):
        """Encourage smooth evolution of wrist error vectors."""
        tracking_terms = self._get_virtual_wrist_tracking_terms()
        if tracking_terms is None:
            return torch.zeros(self.num_envs, device=self.device)

        left_error_vec, right_error_vec, active_mask = tracking_terms

        if not hasattr(self, "last_left_wrist_error"):
            self.last_left_wrist_error = torch.zeros_like(left_error_vec)
        if not hasattr(self, "last_right_wrist_error"):
            self.last_right_wrist_error = torch.zeros_like(right_error_vec)

        left_change = torch.norm(left_error_vec - self.last_left_wrist_error, dim=1)
        right_change = torch.norm(right_error_vec - self.last_right_wrist_error, dim=1)
        combined_change = 0.5 * (left_change + right_change)

        self.last_left_wrist_error = left_error_vec.clone()
        self.last_right_wrist_error = right_error_vec.clone()

        return torch.exp(-combined_change / 0.05) * active_mask
