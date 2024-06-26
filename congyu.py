import sys, getopt
from collections import deque
import random
import math
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from load import load_arguments, save_arguments, default_arguments

from environments.noise_generation import NoiseGenerator
from environments.obstacle_generation import ObstacleMapGenerator
from environments.terrain_generation import TerrainGenerator
from environments.env_representation import EnvironmentRepresentation, GeneralEnvironmentRepresentation

from networks.simple_q_network import SimpleDeepQNetworkGenerator
from networks.simple_q_network import SimpleDeepQNetworkGenerator2
from networks.network3 import DeepQNetworkGenerator3

from deep_rl.deep_q_agent import DeepQAgent
from deep_rl.double_dqn_agent import DoubleDeepQAgent
from deep_rl.trainer import DeepRLTrainer


class GeneralEnvironmentGenerator:
    def __init__(self, n_dim):
        self.dim = (8, 8)
        self.set_dimension(n_dim)

        # 障碍物
        self.obstacle_generator = ObstacleMapGenerator()
        # 地形
        self.terrain_generator = TerrainGenerator()

        self.loaded_representation = None

    def get_dimension(self):
        return self.dim

    def set_dimension(self, n_dim):
        assert (n_dim[0] == n_dim[1])
        self.dim = n_dim

    def set_obstacle_frequency(self, n_freq):
        self.obstacle_generator.set_frequency(n_freq)

    def set_fill_ratio(self, n_ratio):
        self.obstacle_generator.fill_ratio = n_ratio

    def set_height_frequency(self, n_freq):
        self.terrain_generator.set_frequency(n_freq)

    def load_env_representation(self, env_repr):
        self.loaded_representation = env_repr
        dim_x, dim_y = env_repr.get_obstacle_map().shape
        self.set_dimension((dim_x, dim_y))

    def get_area_minimum(self):
        return self.dim[0] / 4

    def get_interval_and_count(self, extra_spacing):
        freq_x, freq_y = self.terrain_generator.get_frequency()
        min_freq = min(freq_x, freq_y)

        interval = self.dim[0] // min_freq

        count = (2 * extra_spacing) // interval
        if not (2 * extra_spacing) % interval == 0:
            count += 1

        return interval, count

    def get_final_extra_space(self, extra_spacing):
        interval, count = self.get_interval_and_count(extra_spacing)
        return (count * interval) // 2

    def get_final_frequency_terrain(self, extra_spacing):
        interval, count = self.get_interval_and_count(extra_spacing)
        freq_x, freq_y = self.terrain_generator.get_frequency()
        interval_x = self.dim[0] // freq_x
        multiplier_x = interval // interval_x
        interval_y = self.dim[0] // freq_y
        multiplier_y = interval // interval_y
        n_freq = freq_x + count * multiplier_x, freq_y + count * multiplier_y
        return freq_x + count * multiplier_x, freq_y + count * multiplier_y

    def generate_environment(self, extra_spacing=0, agent_radius=1):
        # loaded representation
        if self.loaded_representation is not None:
            return self.loaded_representation
        # compute extra spacing so that the terrain frequency holds
        x_dim, y_dim = self.dim
        final_extra_spacing = self.get_final_extra_space(extra_spacing)
        # create an obstacle map with enough extra spacing
        obstacle_map = None
        nb_tiles = 0
        start_positions = []
        self.obstacle_generator.set_dimension(self.dim)
        while nb_tiles < self.get_area_minimum():
            obstacle_map, nb_tiles, start_positions = self.obstacle_generator.generate_obstacle_map(
                agent_radius)
        extra_dim_x, extra_dim_y = x_dim + 2 * \
            final_extra_spacing, y_dim + 2 * final_extra_spacing
        extra_map = np.ones((extra_dim_x, extra_dim_y))
        extra_map[
            final_extra_spacing: extra_dim_x - final_extra_spacing,
            final_extra_spacing: extra_dim_y - final_extra_spacing
        ] = obstacle_map

        # create a terrain map with enough extra spacing
        t_freq = self.terrain_generator.get_frequency()
        self.terrain_generator.set_frequency(
            self.get_final_frequency_terrain(extra_spacing))

        self.terrain_generator.set_dimension(
            (x_dim + 2*final_extra_spacing, y_dim + 2*final_extra_spacing))
        terrain_map = self.terrain_generator.generate_terrain_map()

        self.terrain_generator.set_frequency(t_freq)

        # create environment representation
        env_representation = GeneralEnvironmentRepresentation(
            extra_map,
            nb_tiles,
            start_positions,
            terrain_map,
            final_extra_spacing
        )

        return env_representation


class GeneralEnvironment:
    MAX_STEP_MULTIPLIER = 2

    MOVE_PUNISH = 0.05
    OBSTACLE_PUNISH = 0.5
    TERR_PUNISH = 0.5

    DISC_REWARD = 1.0
    CC_REWARD = 50.0

    ACTION_BUFFER_LENGTH = 10

    def __init__(self, generator):
        self.agent_size = 1
        self.fov = None
        self.turning = False
        self.terrain_info = False

        self.generator = generator
        self.env_repr = None

        self.done = False
        self.nb_steps = 0
        self.total_covered_tiles = 0
        self.total_reward = 0.0
        self.total_terr_diff = 0.0
        self.total_pos_terr_diff = 0.0

        self.current_position = (0, 0)
        self.visited_tiles = []

        self.angle_count = 0

        self.action_buffer = deque(
            maxlen=GeneralEnvironment.ACTION_BUFFER_LENGTH)

    def set_agent_size(self, n_size):
        assert (n_size >= 1)

        self.agent_size = n_size

    def set_field_of_view(self, n_fov):
        assert (n_fov is None or (n_fov >= 1 and n_fov % 2 == 1))

        self.fov = n_fov

    def activate_turning(self, active):
        self.turning = active

    def activate_terrain(self, active):
        self.terrain_info = active

    def get_nb_actions(self):
        return 3 if self.turning else 4

    def get_state_shape(self):
        dim_x, dim_y = self.generator.get_dimension()
        if self.fov is not None:
            dim_x, dim_y = self.fov, self.fov

        nb_channels = 3
        if self.fov is not None:
            nb_channels -= 1
        if self.terrain_info:
            nb_channels += 1

        return [nb_channels, dim_x, dim_y]

    def get_extra_spacing(self):
        extra_spacing = self.agent_size // 2
        if self.fov is not None and self.turning:
            extra_fov = self.fov // 2
            extra_turning = (self.fov - 1) // 5 + 1
            extra_spacing = max(
                extra_spacing, extra_fov + extra_turning
            )
        elif self.fov is not None:
            extra_spacing = max(
                extra_spacing, self.fov // 2
            )
        elif self.turning:
            extra_spacing = max(
                extra_spacing, (self.generator.get_dimension()[0] - 1) // 5 + 1
            )
        return extra_spacing

    def get_obstacle_map(self):
        return np.copy(
            self.env_repr.get_obstacle_map()
        )

    def get_coverage_map(self):
        return np.copy(
            self.visited_tiles
        )

    def reset(self):
        self.env_repr = self.generator.generate_environment(
            self.get_extra_spacing(),
            int((self.agent_size + 1) / 2)
        )

        self.done = False
        self.nb_steps = 0
        self.total_reward = 0.0
        self.total_terr_diff = 0.0
        self.total_pos_terr_diff = 0.0

        self.current_position = random.sample(
            self.env_repr.get_start_positions(), 1
        )[0]
        self.visited_tiles = np.zeros_like(self.env_repr.get_obstacle_map())

        # cover tiles at initial position
        mask = GeneralEnvironment.get_radius_map(self.agent_size)
        xx, yy = self.get_local_map(
            self.agent_size,
            self.current_position,
            "indices"
        )
        xx_select = xx[mask]
        yy_select = yy[mask]
        self.visited_tiles[(xx_select, yy_select)] = 1
        self.total_covered_tiles = np.sum(self.visited_tiles)

        self.action_buffer = deque(
            maxlen=GeneralEnvironment.ACTION_BUFFER_LENGTH)

        if self.turning:
            self.set_random_angle_count()

        return self.get_state()

    def set_random_angle_count(self):
        assert self.turning

        self.angle_count = np.random.randint(0, 8)
        collision = True

        while collision:
            self.angle_count = np.random.randint(0, 8)
            n_pos = self.get_new_position(1)
            collision = self.verify_collision(n_pos)

    def step(self, action):
        n_position = self.get_new_position(action)

        collision = self.verify_collision(n_position)
        nb_covered_tiles = self.get_nb_covered_tiles(n_position)
        terr_diff = self.get_terrain_difference(n_position)

        full_cc = self.complete_coverage(nb_covered_tiles)
        loop = self.detect_loop(action)

        done = self.get_done_status(collision, full_cc, loop)
        reward = self.get_reward(collision, nb_covered_tiles,
                                 full_cc, terr_diff)

        self.done = done
        self.nb_steps += 1
        self.total_covered_tiles += nb_covered_tiles
        self.total_terr_diff += terr_diff
        self.total_pos_terr_diff += max(0, terr_diff)
        self.total_reward += reward

        self.update_environment(action, n_position)

        state = self.get_state()
        info = self.get_info({
            "collision": collision,
            "nb_covered_tiles": nb_covered_tiles,
            "terr_diff": terr_diff,
            "full_cc": full_cc,
            "loop": loop,
            "done": done,
            "reward": reward
        })

        return state, reward, done, info

    def get_new_position(self, action):
        assert (action < self.get_nb_actions())

        n_position = self.current_position
        if not self.turning:
            if action == 0:
                n_position = (
                    max(0, self.current_position[0] - 1),
                    self.current_position[1]
                )
            elif action == 1:
                max_x = self.generator.get_dimension()[0] - 1
                n_position = (
                    min(max_x, self.current_position[0] + 1),
                    self.current_position[1]
                )
            elif action == 2:
                n_position = (
                    self.current_position[0],
                    max(0, self.current_position[1] - 1)
                )
            elif action == 3:
                max_y = self.generator.get_dimension()[1] - 1
                n_position = (
                    self.current_position[0],
                    min(max_y, self.current_position[1] + 1)
                )
        else:
            if action == 1:
                n_x, n_y = self.current_position

                if 4 > self.angle_count > 0:
                    n_y += 1
                elif 4 < self.angle_count < 8:
                    n_y -= 1

                turned_angle = (self.angle_count + 2) % 8
                if 4 > turned_angle > 0:
                    n_x += 1
                elif 4 < turned_angle < 8:
                    n_x -= 1

                n_position = (n_x, n_y)

        return n_position

    def get_local_map(self, size, position, type, copy=True):
        assert (type in ["obstacle", "terrain", "coverage", "indices"])

        offset = size // 2
        extra = 1 if size % 2 == 1 else 0

        if type == "indices":
            x = np.linspace(
                position[0] - offset, position[0] + offset,
                size, endpoint=True, dtype=int
            )
            y = np.linspace(
                position[1] - offset, position[1] + offset,
                size, endpoint=True, dtype=int
            )
            xx, yy = np.meshgrid(x, y)

            return np.transpose(xx), np.transpose(yy)

        x_pos = position[0] + offset
        y_pos = position[1] + offset

        map = None
        if type == "obstacle":
            map = self.env_repr.get_obstacle_map(offset)

        elif type == "terrain":
            map = self.env_repr.get_terrain_map(offset)

        elif type == "coverage":
            dim_x, dim_y = self.generator.get_dimension()
            n_dim_x, n_dim_y = (dim_x + 2 * offset, dim_y + 2 * offset)
            map = np.zeros((n_dim_x, n_dim_y))
            map[offset:n_dim_x - offset, offset:n_dim_y -
                offset] = self.visited_tiles

        selection = map[
            x_pos - offset: x_pos + offset + extra,
            y_pos - offset: y_pos + offset + extra
        ]
        if copy:
            return np.copy(selection)

        return selection

    def get_turned_local_map(self, size, center, type, angle):
        assert (type in ["obstacle", "terrain", "coverage"])

        extra_space = self.get_extra_spacing()

        x = np.linspace(0.5, size - 0.5, size) - (size / 2)
        y = np.linspace(0.5, size - 0.5, size) - (size / 2)

        xx, yy = np.meshgrid(x, y)
        xx, yy = np.transpose(xx), np.transpose(yy)

        xx_rot = math.cos(angle) * xx - math.sin(angle) * yy + center[0]
        yy_rot = math.sin(angle) * xx + math.cos(angle) * yy + center[1]

        xx_idxs = np.floor(xx_rot).astype(int) + extra_space
        yy_idxs = np.floor(yy_rot).astype(int) + extra_space

        if type == "obstacle":
            obstacle_map = self.env_repr.get_obstacle_map(extra_space)
            return np.copy(obstacle_map[(xx_idxs, yy_idxs)])

        elif type == "terrain":
            terrain_map = self.env_repr.get_terrain_map(extra_space)
            return np.copy(terrain_map[(xx_idxs, yy_idxs)])

        elif type == "coverage":
            dim_x, dim_y = self.generator.get_dimension()
            n_dim_x, n_dim_y = (dim_x + 2 * extra_space,
                                dim_y + 2 * extra_space)
            coverage_map = np.zeros((n_dim_x, n_dim_y))
            coverage_map[
                extra_space:n_dim_x - extra_space,
                extra_space:n_dim_y - extra_space
            ] = self.visited_tiles
            return np.copy(coverage_map[(xx_idxs, yy_idxs)])

    def verify_collision(self, n_position):
        local_obstacle_map = self.get_local_map(
            self.agent_size,
            n_position,
            "obstacle"
        )
        mask = GeneralEnvironment.get_radius_map(self.agent_size)
        if np.sum(local_obstacle_map[mask]) > 0:
            return True

        return False

    def get_nb_covered_tiles(self, n_position):
        local_coverage_map = self.get_local_map(
            self.agent_size,
            n_position,
            "coverage"
        )
        mask = GeneralEnvironment.get_radius_map(self.agent_size)

        nb_covered_tiles = np.sum(mask) - np.sum(local_coverage_map[mask])

        return nb_covered_tiles

    def complete_coverage(self, nb_covered_tiles):
        nb_free_tiles = self.env_repr.get_nb_free_tiles()
        if np.sum(self.visited_tiles) + nb_covered_tiles == nb_free_tiles:
            return True

        return False

    def detect_loop(self, action):
        # check if action buffer contains sufficient actions
        if len(self.action_buffer) < GeneralEnvironment.ACTION_BUFFER_LENGTH:
            return False

        # check if all even actions are the same (otherwise there is no loop)
        even_idxs = np.arange(0, GeneralEnvironment.ACTION_BUFFER_LENGTH, 2)
        even_actions = np.array(self.action_buffer)[even_idxs]
        if not sum(abs(even_actions - even_actions[0])) == 0:
            return False

        # check if all odd actions are the same (otherwise there is no loop)
        odd_idxs = even_idxs + 1
        odd_actions = np.array(self.action_buffer)[odd_idxs]
        if not sum(abs(odd_actions - odd_actions[0])) == 0:
            return False

        # check that the new action is still part of the loop
        if not action == self.action_buffer[0]:
            return False

        # check if the recurrent actions are opposite actions
        # (meaning that a loop is occurring)
        if self.turning:
            if action == 0 and self.action_buffer[1] == 2:
                return True
            if action == 2 and self.action_buffer[1] == 0:
                return True
        else:
            if action == 0 and self.action_buffer[1] == 1:
                return True
            if action == 1 and self.action_buffer[1] == 0:
                return True
            if action == 2 and self.action_buffer[1] == 3:
                return True
            if action == 3 and self.action_buffer[1] == 2:
                return True

        # if all the previous does not hold, then no looping is occuring
        return False

    def get_terrain_difference(self, n_position):
        curr_height = self.env_repr.get_terrain_map()[self.current_position]
        n_height = self.env_repr.get_terrain_map()[n_position]
        return n_height - curr_height

    def get_done_status(self, collision, full_cc, loop):
        # Terminate on collision with obstacle
        if collision:
            return True

        # Terminate when the agent has covered the full environment
        if full_cc:
            return True

        # Terminate when agent ends up in a loop
        if loop:
            return True

        # Terminate when the agent has done too many steps
        nb_free_tiles = self.env_repr.get_nb_free_tiles()
        if self.nb_steps >= GeneralEnvironment.MAX_STEP_MULTIPLIER * nb_free_tiles:
            return True

        return False

    def get_reward(self, collision, nb_covered_tiles, full_cc, terr_diff):
        # standard punishment every step
        reward = -GeneralEnvironment.MOVE_PUNISH

        # extra punishment on collision with obstacle
        if collision:
            reward -= GeneralEnvironment.OBSTACLE_PUNISH
            return reward

        # reward for every newly covered tile.
        reward += GeneralEnvironment.DISC_REWARD * nb_covered_tiles

        # reward for covering the whole area
        if full_cc:
            reward += GeneralEnvironment.CC_REWARD

        # extra punishment for terrain difference
        if self.terrain_info:
            # only give terrain punishment for positive terrain differences
            diff = max(0, terr_diff)
            reward -= GeneralEnvironment.TERR_PUNISH * diff

        return reward

    def update_environment(self, action, n_position):
        # change turning angle if turning is selected
        if self.turning:
            if action == 0:
                self.angle_count = (self.angle_count + 1) % 8
            elif action == 2:
                self.angle_count = (self.angle_count - 1) % 8

        # update coverage map
        mask = GeneralEnvironment.get_radius_map(self.agent_size)
        local_obstacle_map = self.get_local_map(
            self.agent_size,
            n_position,
            "obstacle"
        )
        mask[local_obstacle_map == 1] = False

        xx, yy = self.get_local_map(
            self.agent_size,
            n_position,
            "indices"
        )
        xx_select, yy_select = xx[mask], yy[mask]

        self.visited_tiles[(xx_select, yy_select)] = 1

        # update position
        self.current_position = n_position

        self.action_buffer.append(action)

    def get_state(self):
        # both FOV and turning are not active
        if self.fov is None and not self.turning:
            curr_pos_map = np.zeros_like(self.env_repr.get_obstacle_map())
            curr_pos_map[self.current_position] = 1

            state = np.stack([curr_pos_map, self.visited_tiles,
                              self.env_repr.get_obstacle_map()])
            if self.terrain_info:
                state = np.concatenate(
                    [state, [self.env_repr.get_terrain_map()]])

            return state

        # FOV is active, but turning is not
        elif self.fov is not None and not self.turning:
            local_coverage_map = self.get_local_map(
                self.fov,
                self.current_position,
                "coverage"
            )
            local_obstacle_map = self.get_local_map(
                self.fov,
                self.current_position,
                "obstacle"
            )
            state = np.stack([local_coverage_map, local_obstacle_map])

            if self.terrain_info:
                local_terrain_map = self.get_local_map(
                    self.fov,
                    self.current_position,
                    "terrain"
                )
                state = np.concatenate([state, [local_terrain_map]])

            return state

        # turning is active, but FOV is not
        elif self.turning and self.fov is None:
            angle = ((self.angle_count - 2) % 8) * (math.pi / 4)
            dim_x, dim_y = self.generator.get_dimension()

            # coverage and obstacle map
            coverage_map = self.get_turned_local_map(
                dim_x,
                (dim_x // 2, dim_y // 2),
                "coverage",
                angle
            )
            obstacle_map = self.get_turned_local_map(
                dim_x,
                (dim_x // 2, dim_y // 2),
                "obstacle",
                angle
            )

            # current position map
            curr_pos_map = np.zeros((dim_x, dim_y))

            rel_pos_x = self.current_position[0] + 0.5 - (dim_x / 2)
            rel_pos_y = self.current_position[1] + 0.5 - (dim_y / 2)

            rot_pos_x = rel_pos_x * \
                math.cos(angle) + rel_pos_y * math.sin(angle)
            rot_pos_y = -rel_pos_x * \
                math.sin(angle) + rel_pos_y * math.cos(angle)

            curr_pos_map[
                np.clip(math.floor(rot_pos_x + dim_x / 2), 0, dim_x - 1),
                np.clip(math.floor(rot_pos_y + dim_y / 2), 0, dim_y - 1)
            ] = 1.0

            state = np.stack([curr_pos_map, coverage_map, obstacle_map])

            if self.terrain_info:
                local_terrain_map = self.get_turned_local_map(
                    dim_x,
                    (dim_x // 2, dim_y // 2),
                    "terrain",
                    angle
                )
                state = np.concatenate([state, [local_terrain_map]])
            return state

        # both FOV and turning are active
        else:
            angle = ((self.angle_count - 2) % 8) * (math.pi / 4)
            local_coverage_map = self.get_turned_local_map(
                self.fov,
                self.current_position,
                "coverage",
                angle
            )
            local_obstacle_map = self.get_turned_local_map(
                self.fov,
                self.current_position,
                "obstacle",
                angle
            )
            state = np.stack([local_coverage_map, local_obstacle_map])

            if self.terrain_info:
                local_terrain_map = self.get_turned_local_map(
                    self.fov,
                    self.current_position,
                    "terrain",
                    angle
                )
                state = np.concatenate([state, [local_terrain_map]])

            return state

    def get_info(self, info):
        info.update({
            "current_position": self.current_position,
            "total_reward": self.total_reward,
            "total_terr_diff": self.total_terr_diff,
            "total_pos_terr_diff": self.total_pos_terr_diff,
            "total_covered_tiles": self.total_covered_tiles,
            "nb_steps": self.nb_steps,
            "agent_size": self.agent_size
        })
        if self.turning:
            info.update({
                "angle": self.angle_count * (math.pi / 4)
            })
        if self.fov is not None:
            info.update({
                "fov": self.fov
            })
        return info

    @staticmethod
    def get_radius_map(agent_size):
        x = np.linspace(0.5, agent_size - 0.5, agent_size) - (agent_size / 2)
        yy, xx = np.meshgrid(x, x)

        dists = np.sqrt(xx ** 2 + yy ** 2)
        mask = dists <= (agent_size / 2)

        return mask



def read_arguments(argv):
    
    # 参数解析
    SHORT_OPTIONS = ""
    LONG_OPTIONS = [
        "loadAgent=",
        
        "loadArguments=",

        "disableCuda",

        "dim=",
        "hFreq=",
        "oFreq=",
        "fillRatio=",
        "loadEnv=",

        "agentSize=",
        "fov=",
        "turn",
        "terrain",

        "movePunish=",
        "terrainPunish=",
        "obstaclePunish=",
        "discoverReward=",
        "coverageReward=",
        "maxStepMultiplier=",

        "gamma=",
        "networkGen=",
        "rlAgent=",
        "epsilonDecay=",
        "targetUpdate=",
        "queueLength=",
        "optim=",
        "lr=",

        "nbEpisodes=",
        "printEvery=",
        "saveEvery=",
        "savePath="
    ]

    try:
        options, args = getopt.getopt(argv, SHORT_OPTIONS, LONG_OPTIONS)
    except getopt.GetoptError:
        print("badly formatted command line arguments")


    arguments = default_arguments()

    for option, argument in options:
        if option == "--loadAgent":
            argument_split = argument.split(",")
            arguments.update(load_arguments(argument_split[0], "arguments"))
            arguments["loadPath"] = argument_split[0]
            arguments["loadEpisode"] = int(argument_split[1])

        if option == "--loadArguments":
            argument_split = argument.split(",")
            arguments.update(load_arguments(argument_split[0], argument_split[1]))

        if option == "--disableCuda":
            arguments["cuda"] = False

        if option == "--dim":
            arguments["dim"] = tuple(tuple(map(int, argument.split(","))))

        if option == "--hFreq":
            arguments["hFreq"] = tuple(map(int, argument.split(",")))

        if option == "--oFreq":
            arguments["oFreq"] = tuple(map(int, argument.split(",")))

        if option == "--fillRatio":
            arguments["fillRatio"] = float(argument)

        if option == "--loadEnv":
            arguments["loadEnv"] = tuple(argument.split(","))

        if option == "--agentSize":
            arguments["agentSize"] = int(argument)

        if option == "--fov":
            arguments["fov"] = int(argument)

        if option == "--turn":
            arguments["turn"] = True

        if option == "--terrain":
            arguments["terrain"] = True

        if option == "--movePunish":
            arguments["movePunish"] = float(argument)

        if option == "--terrainPunish":
            arguments["terrainPunish"] = float(argument)

        if option == "--obstaclePunish":
            arguments["obstaclePunish"] = float(argument)

        if option == "--discoverReward":
            arguments["discoverReward"] = float(argument)

        if option == "--coverageReward":
            arguments["coverageReward"] = float(argument)

        if option == "--maxStepMultiplier":
            arguments["maxStepMultiplier"] = int(argument)

        if option == "--gamma":
            arguments["gamma"] = float(argument)
            assert(float(argument) <= 1.0)

        if option == "--networkGen":
            if argument in GENERATORS:
                arguments["networkGen"] = argument
            else:
                raise Exception("TRAIN.py: given network generator is not defined...")

        if option == "--optim":
            if argument in OPTIMIZERS:
                arguments["optim"] = argument
            else:
                raise Exception("TRAIN.py: given optimizer is not defined...")

        if option == "--lr":
            arguments["lr"] = float(argument)

        if option == "--rlAgent":
            if argument in AGENTS:
                arguments["rlAgent"] = argument
            else:
                raise Exception("TRAIN.py: given agent is not defined...")

        if option == "--epsilonDecay":
            arguments["epsilonDecay"] = int(argument)

        if option == "--targetUpdate":
            arguments["targetUpdate"] = int(argument)

        if option == "--queueLength":
            arguments["queueLength"] = int(argument)

        if option == "--nbEpisodes":
            arguments["nbEpisodes"] = int(argument)

        if option == "--printEvery":
            arguments["printEvery"] = int(argument)

        if option == "--saveEvery":
            arguments["saveEvery"] = int(argument)

        if option == "--savePath":
            arguments["savePath"] = argument

    print(arguments)                    
    save_arguments(arguments, arguments["savePath"])
    return arguments 


def initialize_objects(arguments):
    """成员初始化

    Args:
        arguments (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    print("Initializing objects...")

    # CUDA，不可使用，无用
    device = 'cuda' if (torch.cuda.is_available(
    ) and arguments["cuda"]) else 'cpu'
    print(f"DEVICE: {device}")

    # 环境生成器
    env_generator = GeneralEnvironmentGenerator(arguments["dim"])
    env_generator.set_obstacle_frequency(arguments["oFreq"])
    env_generator.set_fill_ratio(arguments["fillRatio"])
    env_generator.set_height_frequency(arguments["hFreq"])

    # 环境生成
    environment = GeneralEnvironment(env_generator)
    environment.set_agent_size(arguments["agentSize"])
    environment.set_field_of_view(arguments["fov"])
    environment.activate_turning(arguments["turn"])
    environment.activate_terrain(arguments["terrain"])

    # 惩罚
    # reward signal - punishment values
    GeneralEnvironment.MOVE_PUNISH = arguments["movePunish"]
    GeneralEnvironment.TERR_PUNISH = arguments["terrainPunish"]
    GeneralEnvironment.OBSTACLE_PUNISH = arguments["obstaclePunish"]

    # 奖励
    # reward signal - reward values
    GeneralEnvironment.DISC_REWARD = arguments["discoverReward"]
    GeneralEnvironment.CC_REWARD = arguments["coverageReward"]

    # max step multiplier
    GeneralEnvironment.MAX_STEP_MULTIPLIER = arguments["maxStepMultiplier"]

    # 网络结构
    state_shape = environment.get_state_shape()
    NetworkGenerator = {
        "simpleQ": SimpleDeepQNetworkGenerator,
        "simpleQ2": SimpleDeepQNetworkGenerator2,
        "network3": DeepQNetworkGenerator3
    }
    network_generator = NetworkGenerator[arguments["networkGen"]](
        (state_shape[1], state_shape[2]),
        state_shape[0],
        environment.get_nb_actions(),
        device
    )

    # RL AGENT
    AGENTS = {
        "deepQ": DeepQAgent,
        "doubleDQ": DoubleDeepQAgent
    }
    agent_class = AGENTS[arguments["rlAgent"]]
    agent_class.EPSILON_DECAY = arguments["epsilonDecay"]
    agent_class.GAMMA = arguments["gamma"]
    agent_class.TARGET_UPDATE = arguments["targetUpdate"]
    agent_class.QUEUE_LENGTH = arguments["queueLength"]
    agent_class.LEARNING_RATE = arguments["lr"]

    OPTIMIZERS = {
        "rmsProp": optim.RMSprop
    }
    agent = agent_class(
        network_generator,
        OPTIMIZERS[arguments["optim"]],
        environment.get_nb_actions()
    )

    if arguments['loadEpisode'] is not None:
        agent.load(arguments['loadPath'], arguments['loadEpisode'])
        arguments['loadings'].append(arguments['loadEpisode'])

    # 训练器
    DeepRLTrainer.NB_EPISODES = arguments["nbEpisodes"]
    DeepRLTrainer.INFO_EVERY = arguments["printEvery"]
    DeepRLTrainer.SAVE_EVERY = arguments["saveEvery"]
    DeepRLTrainer.DEVICE = device
    trainer = DeepRLTrainer(environment, agent, arguments["savePath"])

    return environment, agent, trainer


def main(argv):
    # 参数读取
    arguments = read_arguments(argv)
             
    # 初始化
    env, agent, trainer = initialize_objects(arguments)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
