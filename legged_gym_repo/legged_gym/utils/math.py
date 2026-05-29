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
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple


# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0.0, -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.0) / 2.0
    return (upper - lower) * r + lower


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

    return roll_x, pitch_y, yaw_z


def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)


def get_yaw_quat_from_quat(quat_angle):
    roll, pitch, yaw = euler_from_quaternion(quat_angle)
    roll = torch.zeros_like(roll)
    pitch = torch.zeros_like(pitch)
    return quat_from_euler_xyz(roll, pitch, yaw)


def quaternion_from_rotation_matrix(rot_matrix: Tensor) -> Tensor:
    """Convert [N, 3, 3] or [N, 4, 4] rotation matrix to [N, 4] quaternion [x, y, z, w].

    Uses Shepperd's method for numerical stability.
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

    s0 = torch.sqrt(trace + 1.0) * 2
    qw0 = 0.25 * s0
    qx0 = (R21 - R12) / s0
    qy0 = (R02 - R20) / s0
    qz0 = (R10 - R01) / s0

    s1 = torch.sqrt(1.0 + R00 - R11 - R22) * 2
    qw1 = (R21 - R12) / s1
    qx1 = 0.25 * s1
    qy1 = (R01 + R10) / s1
    qz1 = (R02 + R20) / s1

    s2 = torch.sqrt(1.0 + R11 - R00 - R22) * 2
    qw2 = (R02 - R20) / s2
    qx2 = (R01 + R10) / s2
    qy2 = 0.25 * s2
    qz2 = (R12 + R21) / s2

    s3 = torch.sqrt(1.0 + R22 - R00 - R11) * 2
    qw3 = (R10 - R01) / s3
    qx3 = (R02 + R20) / s3
    qy3 = (R12 + R21) / s3
    qz3 = 0.25 * s3

    qw = torch.where(cond0, qw0, torch.where(cond1, qw1, torch.where(cond2, qw2, qw3)))
    qx = torch.where(cond0, qx0, torch.where(cond1, qx1, torch.where(cond2, qx2, qx3)))
    qy = torch.where(cond0, qy0, torch.where(cond1, qy1, torch.where(cond2, qy2, qy3)))
    qz = torch.where(cond0, qz0, torch.where(cond1, qz1, torch.where(cond2, qz2, qz3)))

    return torch.stack([qx, qy, qz, qw], dim=-1)
