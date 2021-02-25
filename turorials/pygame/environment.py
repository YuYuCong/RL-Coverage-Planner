import numpy as np

OBSTACLE_PUNISHMENT = -1
DISCOVER_REWARD = 1
MOVE_PUNISHMENT = 0.2
COVERAGE_REWARD = 10

class Environment:
    def __init__(self, obstacles, start_positions):
        self.obstacles = obstacles
        self.start_positions = start_positions
        self.visited_tiles = np.zeros_like(self.obstacles)
        self.current_pos = (0, 0)

    def get_state(self):
        return self.current_pos, self.visited_tiles, self.obstacles

    def reset(self):
        self.visited_tiles = np.zeros_like(self.obstacles)
        self.current_pos = self.start_positions[np.random.randint(len(self.start_positions))]
        return self.get_state()

    def get_reward(self):
        if self.obstacles[self.current_pos] == 1:
            return OBSTACLE_PUNISHMENT
        if self.visited_tiles[self.current_pos] == 0:
            return DISCOVER_REWARD

        if np.sum(self.visited_tiles  + self.obstacles - np.ones_like(self.obstacles)) == 0:
            return COVERAGE_REWARD

        return MOVE_PUNISHMENT

    def is_done(self):
        if self.obstacles[self.current_pos] == 1:
            return True
        if np.sum(self.visited_tiles  + self.obstacles - np.ones_like(self.obstacles)) == 0:
            return True

        return False

    def step(self, action):
        self.visited_tiles[self.current_pos] = 1

        if action == 0:
            self.current_pos = (max(0, self.current_pos[0] - 1), self.current_pos[1])
        elif action == 1:
            self.current_pos = (min(self.obstacles.shape[0] - 1, self.current_pos[0] + 1), self.current_pos[1])
        elif action == 2:
            self.current_pos = (self.current_pos[0], max(0, self.current_pos[1] - 1))
        elif action == 3:
            self.current_pos = (self.current_pos[0], min(self.obstacles.shape[1] - 1, self.current_pos[1] + 1))

        return self.get_state(), self.get_reward(), self.is_done()