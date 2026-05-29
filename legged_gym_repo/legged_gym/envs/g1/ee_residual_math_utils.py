import numpy as np
import torch


class OnlineAcc(torch.nn.Module):

    def __init__(self, dt: float, tau: float):
        super().__init__()
        alpha = float(np.exp(-dt / tau))
        self.register_buffer("alpha", torch.tensor(alpha))
        self.dt = dt
        self.v_filt_prev = None

    def forward(self, v_now: torch.Tensor):
        if self.v_filt_prev is None:
            self.v_filt_prev = v_now.clone()
            return torch.zeros_like(v_now)

        v_filt = (1.0 - self.alpha) * v_now + self.alpha * self.v_filt_prev
        a_now = (v_filt - self.v_filt_prev) / self.dt

        self.v_filt_prev = v_filt
        return a_now

def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:, 0]
    y = quat_angle[:, 1]
    z = quat_angle[:, 2]
    w = quat_angle[:, 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x.unsqueeze(1), pitch_y.unsqueeze(1), yaw_z.unsqueeze(1)

def quaternion_from_rotation_matrix(rot_matrix):
    """
    
    
    Args:
    
    Returns:
    """
    if rot_matrix.shape[-1] == 4:
        R = rot_matrix[:, :3, :3]
    else:
        R = rot_matrix
    
    R00, R01, R02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    R10, R11, R12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    R20, R21, R22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    
    trace = R00 + R11 + R22
    
    cond0 = trace > 0
    cond1 = (R00 > R11) & (R00 > R22)
    cond2 = R11 > R22
    cond3 = ~cond0 & ~cond1 & ~cond2
    
    # Case 1: trace > 0
    s0 = torch.sqrt(trace + 1.0) * 2  # s = 4 * qw
    qw0 = 0.25 * s0
    qx0 = (R21 - R12) / s0
    qy0 = (R02 - R20) / s0
    qz0 = (R10 - R01) / s0
    
    s1 = torch.sqrt(1.0 + R00 - R11 - R22) * 2  # s = 4 * qx
    qw1 = (R21 - R12) / s1
    qx1 = 0.25 * s1
    qy1 = (R01 + R10) / s1
    qz1 = (R02 + R20) / s1
    
    s2 = torch.sqrt(1.0 + R11 - R00 - R22) * 2  # s = 4 * qy
    qw2 = (R02 - R20) / s2
    qx2 = (R01 + R10) / s2
    qy2 = 0.25 * s2
    qz2 = (R12 + R21) / s2
    
    s3 = torch.sqrt(1.0 + R22 - R00 - R11) * 2  # s = 4 * qz
    qw3 = (R10 - R01) / s3
    qx3 = (R02 + R20) / s3
    qy3 = (R12 + R21) / s3
    qz3 = 0.25 * s3
    
    qw = torch.where(cond0, qw0, torch.where(cond1, qw1, torch.where(cond2, qw2, qw3)))
    qx = torch.where(cond0, qx0, torch.where(cond1, qx1, torch.where(cond2, qx2, qx3)))
    qy = torch.where(cond0, qy0, torch.where(cond1, qy1, torch.where(cond2, qy2, qy3)))
    qz = torch.where(cond0, qz0, torch.where(cond1, qz1, torch.where(cond2, qz2, qz3)))
    
    return torch.stack([qx, qy, qz, qw], dim=-1)

def compute_arm_mass_matrix(env_instance):
    """
    arm-only CRBA:
    M = Σ J_com^T @ I_com @ J_com
    
    """
    if not hasattr(env_instance, 'obs_dof_indices'):
        return torch.zeros(
            (env_instance.num_envs, 14, 14),
            device=env_instance.device
        )

    device = env_instance.device
    B = env_instance.num_envs

    arm_joint_indices = env_instance.obs_dof_indices
    na = len(arm_joint_indices)

    M = torch.zeros((B, na, na), device=device)

    # select chains with explicit base link names (no fallback allowed)
    chains = []
    if hasattr(env_instance, 'left_arm_chain'):
        if not hasattr(env_instance, 'left_arm_base_link'):
            raise ValueError("left_arm_base_link must be defined for base rotation lookup")
        chains.append((env_instance.left_arm_chain,
                       env_instance.left_arm_indices_in_dof_tensor,
                       env_instance.left_arm_base_link))
    if hasattr(env_instance, 'right_arm_chain'):
        if not hasattr(env_instance, 'right_arm_base_link'):
            raise ValueError("right_arm_base_link must be defined for base rotation lookup")
        chains.append((env_instance.right_arm_chain,
                       env_instance.right_arm_indices_in_dof_tensor,
                       env_instance.right_arm_base_link))

    # orientation utility
    def quat_to_rotmat(q):
        x, y, z, w = q.unbind(-1)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        R = torch.empty((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
        R[:, 0, 0] = 1 - 2 * (yy + zz)
        R[:, 0, 1] = 2 * (xy - wz)
        R[:, 0, 2] = 2 * (xz + wy)
        R[:, 1, 0] = 2 * (xy + wz)
        R[:, 1, 1] = 1 - 2 * (xx + zz)
        R[:, 1, 2] = 2 * (yz - wx)
        R[:, 2, 0] = 2 * (xz - wy)
        R[:, 2, 1] = 2 * (yz + wx)
        R[:, 2, 2] = 1 - 2 * (xx + yy)
        return R

    def get_chain_base_rot_by_name(base_name: str):
        idx = env_instance.body_names.index(base_name)
        q_base = env_instance.rigid_body_states[:, idx, 3:7]
        return quat_to_rotmat(q_base)

    for chain, joint_indices, base_name in chains:
        R_base_world = get_chain_base_rot_by_name(base_name)  # [B,3,3]
        q = env_instance.dof_pos[:, joint_indices]    # [B, na_chain]
        fk_all = chain.forward_kinematics(q, end_only=False)

        J_list = []
        R_bl_list = []
        mass_list = []
        com_list = []
        inertia_list = []
        
        valid_links = []
        for link_name in fk_all.keys():
            if link_name in env_instance.link_inertial_dict:
                valid_links.append(link_name)
        
        if not valid_links:
            continue

        for link in valid_links:
            # 1. Jacobian at origin (Base frame) [B, 6, n_chain]
            J_list.append(chain.jacobian(q, link))
            
            # 2. Rotation Base -> Link [B, 3, 3]
            R_bl_list.append(fk_all[link].get_matrix()[:, :3, :3])
            
            # 3. Inertial properties
            inertial = env_instance.link_inertial_dict[link]
            mass_list.append(inertial["mass"])
            com_list.append(inertial["com_offset"]) # [3]
            
            if "inertia_tensor" in inertial:
                inertia_list.append(inertial["inertia_tensor"]) # [3, 3]
            else:
                # Fallback construction
                I_tens = torch.tensor([
                    [inertial["inertia"]["ixx"], inertial["inertia"]["ixy"], inertial["inertia"]["ixz"]],
                    [inertial["inertia"]["ixy"], inertial["inertia"]["iyy"], inertial["inertia"]["iyz"]],
                    [inertial["inertia"]["ixz"], inertial["inertia"]["iyz"], inertial["inertia"]["izz"]],
                ], device=device, dtype=torch.float32)
                inertia_list.append(I_tens)

        J_stack = torch.stack(J_list, dim=1) # [B, N_links, 6, n_chain]
        R_bl_stack = torch.stack(R_bl_list, dim=1) # [B, N_links, 3, 3]
        
        mass_stack = torch.tensor(mass_list, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)
        com_stack = torch.stack(com_list).to(device).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)
        inertia_local_stack = torch.stack(inertia_list).to(device).unsqueeze(0).expand(B, -1, -1, -1)

        
        # 1. World -> Link Rotation
        R_wl_stack = torch.matmul(R_base_world.unsqueeze(1), R_bl_stack) # [B, N_links, 3, 3]
        
        # 2. CoM in World Frame
        r_com_world_stack = torch.matmul(R_wl_stack, com_stack).squeeze(-1) # [B, N_links, 3]
        
        # 3. Jacobian Transform to World & CoM
        Jv_b = J_stack[:, :, :3, :]
        Jw_b = J_stack[:, :, 3:, :]
        
        R_bw_exp = R_base_world.unsqueeze(1)
        Jv_w = torch.matmul(R_bw_exp, Jv_b)
        Jw_w = torch.matmul(R_bw_exp, Jw_b)
        
        # Shift to CoM: Jv_com = Jv_w + w x r
        Jw_w_t = Jw_w.transpose(-1, -2)
        r_exp = r_com_world_stack.unsqueeze(2)
        cross_term = torch.cross(Jw_w_t, r_exp, dim=-1)
        Jv_com = Jv_w + cross_term.transpose(-1, -2)
        
        J_com_stack = torch.cat([Jv_com, Jw_w], dim=2) # [B, N, 6, n]
        
        # 4. Spatial Inertia in World Frame
        I_rot_stack = torch.matmul(
            torch.matmul(R_wl_stack, inertia_local_stack), 
            R_wl_stack.transpose(-1, -2)
        )
        
        I_spatial_stack = torch.zeros((B, len(valid_links), 6, 6), device=device)
        eye_3 = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
        mass_block = mass_stack.unsqueeze(-1).unsqueeze(-1) * eye_3
        I_spatial_stack[:, :, :3, :3] = mass_block
        I_spatial_stack[:, :, 3:, 3:] = I_rot_stack
        
        # 5. CRBA Summation: M_chain = sum( J^T @ I @ J )
        term1 = torch.matmul(J_com_stack.transpose(2, 3), I_spatial_stack)
        M_chain_sum = torch.matmul(term1, J_com_stack).sum(dim=1) # [B, n, n]

        chain_idx_map = []
        M_idx_map = []
        for local_i, global_idx in enumerate(joint_indices):
            match = (arm_joint_indices == global_idx).nonzero(as_tuple=True)[0]
            if len(match) > 0:
                chain_idx_map.append(local_i)
                M_idx_map.append(match.item())
        
        if not chain_idx_map:
            continue
            
        chain_idx_tensor = torch.tensor(chain_idx_map, device=device, dtype=torch.long)
        M_idx_tensor = torch.tensor(M_idx_map, device=device, dtype=torch.long)
        
        grid_chain_x, grid_chain_y = torch.meshgrid(chain_idx_tensor, chain_idx_tensor, indexing='ij')
        grid_M_x, grid_M_y = torch.meshgrid(M_idx_tensor, M_idx_tensor, indexing='ij')
        
        M[:, grid_M_x, grid_M_y] += M_chain_sum[:, grid_chain_x, grid_chain_y]

    return M
