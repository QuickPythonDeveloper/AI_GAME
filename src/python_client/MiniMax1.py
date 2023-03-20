from base import Action
from utils.config import GEMS
import numpy as np
import re
import datetime
import warnings


class MiniMax:
    def __init__(self, agent):
        self.agent = agent
        self.map = np.array(self.agent.grid)
        self.height = self.agent.grid_height
        self.width = self.agent.grid_width
        if 'wall_indexes' not in self.agent.__dict__:
            self.agent.wall_indexes = self.make_wall_indexes()
        if 'barbed_indexes' not in self.agent.__dict__:  # int
            self.agent.barbed_indexes = self.make_barbed_indexes()
        if 'key_indexes' not in self.agent.__dict__:  # str
            self.agent.key_indexes = self.make_key_indexes()
        if 'door_indexes' not in self.agent.__dict__:  # str
            self.agent.door_indexes = self.make_door_indexes()
        self.agent.gem_indexes = self.make_gem_indexes()
        self.gem = ['1', '2', '3', '4']
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT",
                        "DOWN_RIGHT", "DOWN_LEFT", "UP_LEFT", "UP_RIGHT"]
        self.character = self.agent.character
        self.visited_indexes_A = []
        self.visited_indexes_B = []
        self.agent_A_score = 0
        self.agent_B_score = 0
        if 'keys' not in self.agent.__dict__:
            self.agent.keys = {'r': 0,
                               'y': 0,
                               'g': 0}

    def make_barbed_indexes(self) -> list:
        return []

    def make_door_indexes(self) -> list:
        return []

    def make_key_indexes(self) -> list:
        return []

    def get_agent_index(self, character):
        agent_index = np.empty((0, 2), dtype=int)
        for row in range(self.map.shape[0]):
            if character == 'A':
                agent = np.where(self.map[row] == 'EA')
                if len(agent[0]) != 0:
                    agent_index = np.vstack((agent_index, [row, agent[0][0]]))
            if character == 'B':
                agent = np.where(self.map[row] == 'EB')
                if len(agent[0]) != 0:
                    agent_index = np.vstack((agent_index, [row, agent[0][0]]))
        return [agent_index[0][0], agent_index[0][1]]

    def make_gem_indexes(self) -> list:
        gem_indexes = []  # row, col
        for row in range(self.height):
            for col in range(self.width):
                if self.map[row][col] == "1":
                    gem_indexes.append((row, col))
                if self.map[row][col] == "2":
                    gem_indexes.append((row, col))
                if self.map[row][col] == "3":
                    gem_indexes.append((row, col))
                if self.map[row][col] == "4":
                    gem_indexes.append((row, col))
        return gem_indexes

    def make_wall_indexes(self) -> list:
        return []

    def calc_gems_scores(self, gem: str, prev_gem: str) -> int:
        return 0
