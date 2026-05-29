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
from numpy.random import choice
from scipy import interpolate

from legged_gym.utils import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", "plane"]:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[: i + 1]) for i in range(len(cfg.terrain_proportions))]
        self.hard_terrain = cfg.hard_terrain

        self.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = (
            np.zeros((cfg.num_rows, cfg.num_cols, 3))
            if self.hard_terrain
            else np.zeros((2 * cfg.num_rows, cfg.num_cols, 3))
        )
        self.platform_length = (
            np.zeros((cfg.num_rows, cfg.num_cols)) if self.hard_terrain else np.zeros((2 * cfg.num_rows, cfg.num_cols))
        )

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.length_per_env_pixels) + 2 * self.border
        self.tot_rows = (
            int(cfg.num_rows * self.width_per_env_pixels) + 2 * self.border
            if self.hard_terrain
            else int(2 * cfg.num_rows * self.width_per_env_pixels) + 2 * self.border
        )

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        # for soft terrain
        self.st_distance = int(cfg.num_rows * self.width_per_env_pixels)

        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw, self.cfg.horizontal_scale, self.cfg.vertical_scale, self.cfg.slope_treshold
            )

    def randomized_terrain(self):
        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            # source denotes the terrain that is generated (stepping stones)
            # target denotes the terain where the robot actually walks (planes)
            terrain_source, terrain_target, nearest_center = self.make_terrain(choice, difficulty)
            if self.hard_terrain:
                self.add_terrain_to_map(terrain_source, i, j, nearest_center)
            else:
                self.add_terrain_to_map(terrain_target, i, j, nearest_center)
                self.add_terrain_to_map(terrain_source, i + self.cfg.num_rows, j, nearest_center)

    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows + 0.1
                choice = j / self.cfg.num_cols + 0.001
                if self.hard_terrain:
                    terrain, _, platform_length = self.make_terrain(choice, difficulty)
                    self.add_terrain_to_map(terrain, i, j, platform_length)
                else:
                    # source denotes the terrain that is generated (stepping stones)
                    # target denotes the terain where the robot actually walks (planes) in soft terrain, `None` in hard terrain
                    terrain_source, terrain_target, platform_length = self.make_terrain(choice, difficulty)
                    self.add_terrain_to_map(terrain_target, i, j, platform_length)
                    self.add_terrain_to_map(terrain_source, i + self.cfg.num_rows, j, platform_length)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop("type")
        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        assert self.hard_terrain

        subterrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.length_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        slope = difficulty * 0.4
        amplitude = 0.02 * difficulty
        step_height = 0.05 + 0.175 * difficulty
        discrete_obstacles_height = 0.025 + difficulty * 0.15
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        if choice < self.proportions[0]:
            if choice < (self.proportions[0] / 2):
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(subterrain, slope=slope, platform_size=3.0)
        elif choice < self.proportions[1]:
            if choice < (self.proportions[1] + self.proportions[0]) / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(subterrain, slope=slope, platform_size=3.0)
            terrain_utils.random_uniform_terrain(
                subterrain,
                None,
                min_height=-amplitude,
                max_height=amplitude,
                step=0.005,
                downsampled_scale=0.2,
            )
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(
                subterrain,
                step_width=0.25 + 0.1 * (np.random.rand() - 0.5),
                step_height=step_height,
                platform_size=4.0,
            )
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.0
            rectangle_max_size = 2.0
            terrain_utils.discrete_obstacles_terrain(
                subterrain,
                discrete_obstacles_height,
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=3.0,
            )
        elif choice < self.proportions[5]:
            # terrain_utils.stepping_stones_terrain_easy(
            #     subterrain,
            #     stone_size=stepping_stones_size,
            #     stone_distance=0.1,
            #     max_height=0.0,
            #     platform_size=3.0,
            # )
            pass
        elif choice < self.proportions[6]:
            terrain_utils.random_uniform_terrain(
                subterrain,
                None,
                min_height=-amplitude,
                max_height=amplitude,
                step=0.005,
                downsampled_scale=0.2,
            )
        else:
            raise NotImplementedError(f"Terrain type {choice} not implemented.")

        return subterrain, None, -1  # ! -1 is a placeholder for the platform length, which is not used in our project

        # elif choice < self.proportions[6]:
        #     poles_subterrain(subterrain=subterrain, difficulty=difficulty)
        #     self.walkable_field_raw[start_x:end_x, start_y:end_y] = (
        #         subterrain.height_field_raw != 0
        #     )
        # elif choice < self.proportions[7]:
        #     subterrain.terrain_name = "flat"

        #     flat_border = int(4 / self.horizontal_scale)

        #     self.flat_field_raw[
        #         start_x + flat_border : end_x - flat_border,
        #         start_y + flat_border : end_y - flat_border,
        #     ] = 0
        #     # plain walking terrain
        #     pass
        # self.height_field_raw[start_x:end_x, start_y:end_y] = (
        #     subterrain.height_field_raw
        # )

        # self.walkable_field_raw = ndimage.binary_dilation(
        #     self.walkable_field_raw, iterations=3
        # ).astype(int)

    def make_terrain_beamdojo(self, choice, difficulty):  # ! only for BeamDojo
        terrain_source = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.length_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )

        if not self.hard_terrain:
            terrain_target = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=self.cfg.vertical_scale,
                horizontal_scale=self.cfg.horizontal_scale,
            )
        else:
            terrain_target = None

        # TODO: update: [smooth slope, rough slope, stairs up, stairs down, discrete]

        if choice < self.proportions[0]:
            # plane (both)
            platform_length = self.env_length
            terrain_utils.plane(terrain_source, terrain_target)
            terrain_utils.random_uniform_terrain(
                terrain_source, terrain_target, min_height=-0.02, max_height=0.02, step=0.005, downsampled_scale=0.2
            )

        elif choice < self.proportions[1]:
            # stepping stones (soft)
            stepping_stones_size = np.maximum(1.5 * (1.0 - difficulty), 0.25)
            stone_distance = 0.05 * np.ceil(difficulty / 0.2)
            if difficulty > 0.5:
                platform_length = 1.5
            else:
                platform_length = 2.0
            terrain_utils.stepping_stones_terrain_easy(
                terrain_source,
                terrain_target,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                platform_length=platform_length,
                max_height=0.0,
            )
            terrain_utils.random_uniform_terrain(
                terrain_source, terrain_target, min_height=-0.02, max_height=0.02, step=0.005, downsampled_scale=0.2
            )

        elif choice < self.proportions[2]:
            # stepping stones easy (hard)
            stepping_stones_size = np.maximum(1.5 * (1.0 - difficulty), 0.25)
            stone_distance = 0.05 * np.ceil(difficulty / 0.2)
            if difficulty > 0.5:
                platform_length = 1.5
            else:
                platform_length = 2.0
            terrain_utils.stepping_stones_terrain_easy(
                terrain_source,
                terrain_target,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                platform_length=platform_length,
                max_height=0.0,
            )
            terrain_utils.random_uniform_terrain(
                terrain_source, terrain_target, min_height=-0.02, max_height=0.02, step=0.005, downsampled_scale=0.2
            )

        elif choice < self.proportions[3]:
            # stepping stones hard (hard)
            stepping_stones_size = np.maximum(1.5 * (1.0 - difficulty), 0.25)
            stone_distance = np.clip(0.05 * np.ceil(difficulty / 0.2), 0.1, 0.2)
            if difficulty > 0.5:
                platform_length = 1.5
            else:
                platform_length = 2.0
            terrain_utils.stepping_stones_terrain_hard(
                terrain_source,
                terrain_target,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                platform_length=platform_length,
                max_height=0.0,
            )
            terrain_utils.random_uniform_terrain(
                terrain_source, terrain_target, min_height=-0.02, max_height=0.02, step=0.005, downsampled_scale=0.2
            )

        elif choice < self.proportions[4]:
            # balancing beams via stones (hard)
            stone_size = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.25, 0.25, 0.25, 0.25]
            stone_distance_x = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0, 0.0]
            stone_distance_y = [0.2, 0.2, 0.2, 0.2, 0.25, 0.3, 0.35, 0.4, 0.4, 0.4]
            platform_size = [2.0, 1.5]
            platform_length = platform_size[1]
            terrain_utils.balancing_beams_via_stones_terrain(
                terrain_source,
                terrain_target,
                stone_size=stone_size[round(difficulty / 0.1 - 1)],
                stone_distance_x=stone_distance_x[round(difficulty / 0.1 - 1)],
                stone_distance_y=stone_distance_y[round(difficulty / 0.1 - 1)],
                max_height=0.0,
                platform_size=platform_size,
            )
            terrain_utils.random_uniform_terrain(
                terrain_source, terrain_target, min_height=-0.02, max_height=0.02, step=0.005, downsampled_scale=0.2
            )

        elif choice < self.proportions[5]:
            # stepping stones deploy (hard)
            stepping_stones_size = np.maximum(1.5 * (1.0 - difficulty), 0.25)
            stone_distance = np.clip(0.05 * np.ceil(difficulty / 0.2), 0.1, 0.2)
            platform_size = [2.0, 1.5]
            platform_length = platform_size[1]
            terrain_utils.stepping_stones_hard_terrain_deploy(
                terrain_source,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                platform_size=platform_size,
                max_height=0.0,
            )
            terrain_utils.random_uniform_terrain(
                terrain_source, terrain_target, min_height=-0.02, max_height=0.02, step=0.005, downsampled_scale=0.2
            )

        elif choice < self.proportions[6]:
            # real balancing beams (hard)
            beam_width = [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2]
            deviation_threshold = [0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15, 0.15, 0.2]
            if difficulty > 0.5:
                platform_size = [2.0, 1.5]
            else:
                platform_size = [2.0, 2.0]
            platform_length = platform_size[1]
            terrain_utils.balancing_beams_terrain(
                terrain_source,
                terrain_target,
                beam_width=beam_width[round(difficulty / 0.1 - 1)],
                max_height=0.0,
                platform_size=platform_size,
                deviation_threshold=deviation_threshold[round(difficulty / 0.1 - 1)],
            )
            terrain_utils.random_uniform_terrain(
                terrain_source, terrain_target, min_height=-0.02, max_height=0.02, step=0.005, downsampled_scale=0.2
            )

        elif choice < self.proportions[7]:
            # real stepping beams (hard)
            beam_length = 1.0
            beam_width = 0.2
            beam_distance = 0.3
            platform_size = [2.0, 1.5]
            platform_length = platform_size[1]
            terrain_utils.stepping_beams_terrain(
                terrain_source,
                beam_length=1.0,
                beam_width=beam_width,
                beam_distance=beam_distance,
                max_height=0.0,
                platform_size=platform_size,
            )
            terrain_utils.random_uniform_terrain(
                terrain_source, terrain_target, min_height=-0.02, max_height=0.02, step=0.005, downsampled_scale=0.2
            )

        else:
            raise NotImplementedError

        return terrain_source, terrain_target, platform_length

    def add_terrain_to_map(self, terrain, row, col, platform_length):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.width_per_env_pixels
        end_x = self.border + (i + 1) * self.width_per_env_pixels
        start_y = self.border + j * self.length_per_env_pixels
        end_y = self.border + (j + 1) * self.length_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_width
        # if self.hard_terrain:
        env_origin_y = (j + 0.5) * self.env_length
        # else:
        #     env_origin_y = (j + 0.5) * self.env_length
        # x1 = int((self.env_length / 2.0 - 1) / terrain.horizontal_scale)
        # x2 = int((self.env_length / 2.0 + 1) / terrain.horizontal_scale)
        # y1 = int((self.env_width / 2.0 - 1) / terrain.horizontal_scale)
        # y2 = int((self.env_width / 2.0 + 1) / terrain.horizontal_scale)
        # env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        env_origin_z = 0  # ! We caculate the height later with "_get_init_heights"
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.platform_length[i, j] = platform_length


def gap_terrain(terrain, gap_size, platform_size=1.0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x - x2 : center_x + x2, center_y - y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x - x1 : center_x + x1, center_y - y1 : center_y + y1] = 0


def pit_terrain(terrain, depth, platform_size=1.0):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
