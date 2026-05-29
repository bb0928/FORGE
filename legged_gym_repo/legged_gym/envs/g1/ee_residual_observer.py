import torch
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse
from legged_gym.utils.math import quat_from_euler_xyz

from .ee_residual_math_utils import compute_arm_mass_matrix, euler_from_quaternion


class EEResidualObserverMixin:
    def _init_observer_buffers(self):
        """Initialize momentum observer buffers and topology structure"""
        if not hasattr(self.cfg, 'observer') or not self.cfg.observer.enable:
            return
        
        # Define 14 arm joints (7 left + 7 right)
        arm_joint_names = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        
        # Get DOF indices for arm joints
        self.obs_dof_indices = torch.zeros(
            len(arm_joint_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i, joint_name in enumerate(arm_joint_names):
            if joint_name in self.dof_names:
                self.obs_dof_indices[i] = self.dof_names.index(joint_name)
            else:
                raise ValueError(f"Arm joint {joint_name} not found in DOF names")
        
        # Initialize observer state buffers
        # r: residual estimate (external torque disturbance)
        self.observer_r = torch.zeros(
            self.num_envs, len(arm_joint_names), dtype=torch.float, device=self.device, requires_grad=False
        )
        # p_hat: estimated momentum
        self.observer_p_hat = torch.zeros(
            self.num_envs, len(arm_joint_names), dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # observer_r_wrist: wrist force residuals for adaptation supervision [num_envs, 6]
        self.observer_r_wrist = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # Load observer parameters from config
        self.observer_K = self.cfg.observer.gain
        self.observer_Fc = self.cfg.observer.friction_coulomb
        self.observer_eps = self.cfg.observer.friction_epsilon
        self.observer_residual_clip = self.cfg.observer.residual_clip
        
        # NOTE: observer_M_diag removed - now using full mass matrix M(q) from IsaacGym
        # cfg.observer.mass_diag is no longer used (can be kept in config but ignored)
        
        # Build simplified topology structure for gravity computation
        # For each joint, we need to find descendant bodies
        # This is a simplified version - we'll use rigid body states directly
        # Store joint-to-body mapping (simplified: use body indices from Isaac Gym)
        self.simplified_model_structure = {}
        
        # Get body handles for arm links (simplified approach)
        # We'll compute gravity using rigid body states directly in the update function
        # For now, just store the joint indices and we'll compute gravity from FK
        
        # Gravity and mass matrix update frequency control (update every N steps to reduce computation)
        self.observer_gravity_update_freq = 25
        self.observer_gravity_update_counter = 0
        self.observer_g_hat_cache = torch.zeros(
            self.num_envs, len(arm_joint_names), dtype=torch.float, device=self.device, requires_grad=False
        )
        # Mass matrix cache: [num_envs, 14, 14]
        self.observer_M_arm_cache = torch.zeros(
            self.num_envs, 14, 14, dtype=torch.float, device=self.device, requires_grad=False
        )
        
        print(f"Initialized momentum observer for {len(arm_joint_names)} arm joints")
        print(f"Observer gain K={self.observer_K}, Friction Fc={self.observer_Fc}, eps={self.observer_eps}")
        print(f"Gravity and mass matrix update frequency: every {self.observer_gravity_update_freq} steps")

    def _compute_simplified_gravity_and_friction(self):
        """
        Compute simplified gravity and friction torques for arm joints.
        Gravity is updated every N steps to reduce computation overhead.
        
        Returns:
            g_hat: gravity torque estimate [num_envs, 14]
            fric_hat: friction torque estimate [num_envs, 14]
        """
        if not hasattr(self, 'obs_dof_indices'):
            return None, None
        
        num_arm_joints = len(self.obs_dof_indices)
        arm_dof_vel = self.dof_vel[:, self.obs_dof_indices]
        
        # Update gravity only every N steps
        if self.observer_gravity_update_counter % self.observer_gravity_update_freq == 0:
            g_hat = torch.zeros(self.num_envs, num_arm_joints, dtype=torch.float, device=self.device)
            
            if hasattr(self, 'left_arm_chain') and hasattr(self, 'left_arm_indices_in_dof_tensor'):
                left_gravity_torque = self._compute_arm_gravity_torque(is_left=True)
                left_chain_joint_names = [j.name for j in self.left_arm_chain.get_joints()]
                left_chain_name_to_idx = {name: idx for idx, name in enumerate(left_chain_joint_names)}
                for i, obs_idx in enumerate(self.obs_dof_indices):
                    dof_name = self.dof_names[obs_idx]
                    if dof_name in left_chain_name_to_idx:
                        chain_idx = left_chain_name_to_idx[dof_name]
                        g_hat[:, i] = left_gravity_torque[:, chain_idx]

            if hasattr(self, 'right_arm_chain') and hasattr(self, 'right_arm_indices_in_dof_tensor'):
                right_gravity_torque = self._compute_arm_gravity_torque(is_left=False)
                right_chain_joint_names = [j.name for j in self.right_arm_chain.get_joints()]
                right_chain_name_to_idx = {name: idx for idx, name in enumerate(right_chain_joint_names)}
                for i, obs_idx in enumerate(self.obs_dof_indices):
                    dof_name = self.dof_names[obs_idx]
                    if dof_name in right_chain_name_to_idx:
                        chain_idx = right_chain_name_to_idx[dof_name]
                        g_hat[:, i] = right_gravity_torque[:, chain_idx]
            
            self.observer_g_hat_cache = g_hat
        else:
            g_hat = self.observer_g_hat_cache
        
        fric_hat = self.observer_Fc * torch.tanh(arm_dof_vel / self.observer_eps)
        return g_hat, fric_hat

    def _compute_arm_gravity_torque(self, is_left=True):
        batch_size = self.num_envs
        torso_quat = self.rigid_body_states[:, self.torso_body_index, 3:7]
        g_global = torch.tensor([0.0, 0.0, -9.81], device=self.device).repeat(batch_size, 1)
        g_local = quat_rotate_inverse(torso_quat, g_global)
        total_gravity_torque = torch.zeros_like(self.dof_pos)
        subchains = self.left_subchains if is_left else self.right_subchains
        
        for item in subchains:
            chain = item['chain']
            dof_idx = item['dof_indices']
            mass = item['mass']
            com_offset = item['com_offset']
            q_sub = self.dof_pos[:, dof_idx]
            J = chain.jacobian(q_sub, locations=com_offset)
            J_linear = J[:, :3, :]
            F_g = g_local * mass
            tau_sub = -torch.matmul(J_linear.transpose(1, 2), F_g.unsqueeze(-1)).squeeze(-1)
            total_gravity_torque[:, dof_idx] += tau_sub
        
        target_indices = self.left_arm_indices_in_dof_tensor if is_left else self.right_arm_indices_in_dof_tensor
        return total_gravity_torque[:, target_indices]

    def _update_momentum_observer(self):
        """
        Update momentum observer using discrete update law:
        p_hat_{k+1} = p_hat_k + dt * [tau_m - g_hat - tau_fric + r_k]
        r_{k+1} = r_k + dt * K * (M_diag * q_dot - p_hat_k)
        
        tau_m is now obtained from IsaacGym's actual simulated DOF forces (dof_force_tensor)
        instead of PD formula approximation.
        """
        observer_decay = 0.9
        if not hasattr(self, 'obs_dof_indices') or not hasattr(self.cfg, 'observer') or not self.cfg.observer.enable:
            return
        
        # Ensure obs_dof_indices has exactly 14 DOFs (7 left arm + 7 right arm)
        assert self.obs_dof_indices.shape[0] == 14, f"Expected 14 arm DOFs, got {self.obs_dof_indices.shape[0]}"
        
        self.observer_gravity_update_counter += 1
        
        # Get arm joint positions and velocities
        arm_dof_pos = self.dof_pos[:, self.obs_dof_indices]  # [num_envs, 14]
        arm_dof_vel = self.dof_vel[:, self.obs_dof_indices]  # [num_envs, 14]
        
        # Compute gravity and friction
        g_hat, fric_hat = self._compute_simplified_gravity_and_friction()
        if g_hat is None:
            g_hat = torch.zeros_like(arm_dof_vel)
            fric_hat = torch.zeros_like(arm_dof_vel)
        # -------------- BEGIN PD TORQUES -------------------#####################################
        tau_m = torch.zeros_like(arm_dof_vel)
        if hasattr(self, 'p_gains') and hasattr(self, 'd_gains'):

            arm_p_gains = self.p_gains[self.obs_dof_indices]  # [14]
            arm_d_gains = self.d_gains[self.obs_dof_indices]  # [14]
            arm_targets = self.joint_pos_target[:, self.obs_dof_indices]
            tau_m = (arm_p_gains.unsqueeze(0) * (arm_targets - arm_dof_pos) - 
                    arm_d_gains.unsqueeze(0) * arm_dof_vel)
        # -------------- END PD TORQUES -------------------#####################################
        
        # # --------------- BEGIN DOF FORCE TENSOR -------------------#####################################
        # # Use actual measured joint torques from IsaacGym simulation
        # # dof_force_tensor contains the generalized forces at each DOF from the simulator
        
        # # Extract only the arm DOFs for the observer (14 joints: left 7 + right 7)
        
        # # Apply sign correction if needed (default: +1, no correction)
        # # IsaacGym's dof_force_tensor convention may differ from control convention
        # # Set cfg.observer.tau_sign = -1 if direction is reversed
        
        # # Sanity check: tau_m should have correct shape
        # # ---------- END DOF FORCE TENSOR -------------------#####################################
    
        # Extract arm mass matrix subblock M_arm(q) [num_envs, 14, 14]
        # Update mass matrix only every N steps to reduce computation overhead
        # Also update on first call (when cache is all zeros) to ensure proper initialization
        cache_is_zero = (self.observer_M_arm_cache.abs().max() < 1e-6).all()
        should_update = (self.observer_gravity_update_counter % self.observer_gravity_update_freq == 0) or cache_is_zero
        
        if should_update:
            # Use full mass matrix instead of diagonal approximation
            M_arm = compute_arm_mass_matrix(self)  # [num_envs, 14, 14]
            # CRITICAL: Strict validation - no fallback allowed
            assert M_arm.shape[1] == 14 and M_arm.shape[2] == 14, \
                f"M_arm shape error: expected [N, 14, 14], got {M_arm.shape}"
            
            # Numerical safety checks - raise error if matrix is invalid
            if not torch.isfinite(M_arm).all():
                raise RuntimeError("NaN/Inf detected in mass matrix M_arm - simulation unstable")
            
            # Check positive definiteness via diagonal (lightweight check)
            M_arm_diag = torch.diagonal(M_arm, dim1=-2, dim2=-1)  # [num_envs, 14]
            if (M_arm_diag <= 0).any():
                raise RuntimeError(f"Non-positive mass matrix diagonal detected: min={M_arm_diag.min().item():.6f}")
            
            # Cache the computed mass matrix
            self.observer_M_arm_cache = M_arm
        else:
            # Use cached mass matrix
            M_arm = self.observer_M_arm_cache
        

        # Discrete update law
        # p_hat_{k+1} = p_hat_k + dt * [tau_m - g_hat - fric_hat + r_k]
        self.observer_p_hat = self.observer_p_hat + self.dt * (
            tau_m - g_hat - fric_hat + self.observer_r
        )
        
        # r_{k+1} = r_k + dt * K * (M_arm(q) * q_dot - p_hat_k)
        # Use FULL mass matrix multiplication instead of diagonal approximation
        p_ref = torch.bmm(M_arm, arm_dof_vel.unsqueeze(-1)).squeeze(-1)  # [num_envs, 14]
        momentum_error = p_ref - self.observer_p_hat
        self.observer_r = (self.observer_r * observer_decay) + self.dt * self.observer_K * momentum_error
        
        # # Add large noise to observer residual for robustness testing
        # self.observer_r = self.observer_r + observer_noise
        # self.observer_r *= 0.0  # Disable noise for now 
        
        # Clip residual to prevent instability
        self.observer_r = torch.clamp(
            self.observer_r,
            -self.observer_residual_clip,
            self.observer_residual_clip
        )
        
        # Compute end-effector force residuals from torque residuals using Jacobian
        if hasattr(self, 'observer_r') and self.observer_r.shape[1] >= 14:
            tau_res_left = self.observer_r[:, 0:7]   # [N, 7]
            tau_res_right = self.observer_r[:, 7:14]  # [N, 7]
            
            q_left = self.dof_pos[:, self.left_arm_indices_in_dof_tensor]
            q_right = self.dof_pos[:, self.right_arm_indices_in_dof_tensor]
            
            J_left = self.left_arm_chain.jacobian(q_left)
            J_right = self.right_arm_chain.jacobian(q_right)
            
            left_chain_joint_names = [j.name for j in self.left_arm_chain.get_joints()]
            right_chain_joint_names = [j.name for j in self.right_arm_chain.get_joints()]
            
            left_arm_joint_names = [self.dof_names[i] for i in self.obs_dof_indices[:7]]
            right_arm_joint_names = [self.dof_names[i] for i in self.obs_dof_indices[7:14]]
            
            left_arm_indices_in_chain = [i for i, name in enumerate(left_chain_joint_names) if name in left_arm_joint_names]
            right_arm_indices_in_chain = [i for i, name in enumerate(right_chain_joint_names) if name in right_arm_joint_names]
            
            Jv_left = J_left[:, :3, left_arm_indices_in_chain]   # [N, 3, 7]
            Jv_right = J_right[:, :3, right_arm_indices_in_chain]  # [N, 3, 7]
            
            lam = 1e-3
            I3 = torch.eye(3, device=self.device).unsqueeze(0)  # [1, 3, 3]
            
            A_left = Jv_left @ Jv_left.transpose(1, 2) + lam * I3   # [N, 3, 3]
            b_left = Jv_left @ tau_res_left.unsqueeze(-1)            # [N, 3, 1]
            F_left_torso = torch.linalg.solve(A_left, b_left).squeeze(-1)  # [N, 3]
            
            A_right = Jv_right @ Jv_right.transpose(1, 2) + lam * I3
            b_right = Jv_right @ tau_res_right.unsqueeze(-1)
            F_right_torso = torch.linalg.solve(A_right, b_right).squeeze(-1)
            
            torso_quat = self.rigid_body_states[:, self.torso_body_index, 3:7]
            roll, pitch, yaw = euler_from_quaternion(self.base_quat)
            base_yaw_quat = quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                yaw.squeeze(-1)
            )
            
            F_left_world = quat_apply(torso_quat, F_left_torso)
            F_right_world = quat_apply(torso_quat, F_right_torso)
            
            F_left_baseyaw = quat_rotate_inverse(base_yaw_quat, F_left_world)
            F_right_baseyaw = quat_rotate_inverse(base_yaw_quat, F_right_world)
            
            raw_observer_wrist = torch.cat([F_left_baseyaw, F_right_baseyaw], dim=-1)
            scaled_observer_wrist = raw_observer_wrist * self.obs_scales.wrist_force

            
            self.observer_r_wrist = scaled_observer_wrist

            self.wrist_force_hat_world_raw = torch.cat([F_left_world, F_right_world], dim=-1)

            clip_norm = 50.0
            if hasattr(self.cfg, 'observer') and hasattr(self.cfg.observer, 'wrist_force_hat_clip'):
                clip_norm = float(self.cfg.observer.wrist_force_hat_clip)
            def _clip_vec(F):
                norm = torch.norm(F, dim=-1, keepdim=True) + 1e-6
                scale = torch.clamp(clip_norm / norm, max=1.0)
                return F * scale
            left_clip = _clip_vec(F_left_world)
            right_clip = _clip_vec(F_right_world)
            clipped = torch.cat([left_clip, right_clip], dim=-1)

            ema = 0.8
            if hasattr(self.cfg, 'observer') and hasattr(self.cfg.observer, 'wrist_force_hat_ema'):
                ema = float(self.cfg.observer.wrist_force_hat_ema)
            if not hasattr(self, 'wrist_force_hat_world'):
                self.wrist_force_hat_world = torch.zeros_like(clipped)
            self.wrist_force_hat_world = ema * self.wrist_force_hat_world + (1 - ema) * clipped
        else:
            self.observer_r_wrist = torch.zeros(
                self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
            )
