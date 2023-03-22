from typing import List, Tuple
import numpy as np
from base import Action
from model_based_policy import ModelBasedPolicy
from utils.config import GEMS


class MiniMax2(ModelBasedPolicy):
    def __init__(self, agent):
        super().__init__(agent)
        self.agent.agent_index = tuple(self.agent.agent_index)
        if not hasattr(self.agent, 'wall_indexes'):
            self.agent.wall_indexes = self.make_wall_indexes()
        if not hasattr(self.agent, 'gem_indexes'):
            self.agent.gem_indexes = self.make_gem_indexes()
        self.visited_indexes = []
        self.min_iter = 0
        self.max_iter = 0
        if not hasattr(self.agent, 'gem_groups'):
            self.gem_groups = self.group_gems()
        if not hasattr(self.agent, 'prev_gem'):
            if np.sum(self.map == '1') > 0:
                self.agent.prev_gem = None
            else:
                self.agent.prev_gem = '1'
        self.agent.prev_map = self.map

    def make_gem_indexes(self) -> np.array:
        gem_indexes = np.empty((0, 3), dtype=int)  # row, col, gem_number
        for row in range(self.map.shape[0]):
            new_arr = np.where(self.map[row] == '1')
            for col in new_arr[0]:
                gem_indexes = np.vstack((gem_indexes, [row, col, 1]))
            new_arr = np.where(self.map[row] == '2')
            for col in new_arr[0]:
                gem_indexes = np.vstack((gem_indexes, [row, col, 2]))
            new_arr = np.where(self.map[row] == '3')
            for col in new_arr[0]:
                gem_indexes = np.vstack((gem_indexes, [row, col, 3]))
            new_arr = np.where(self.map[row] == '4')
            for col in new_arr[0]:
                gem_indexes = np.vstack((gem_indexes, [row, col, 4]))

        return gem_indexes

    def group_gems(self) -> list:
        gem_groups = []
        gem_indexes = self.agent.gem_indexes
        while True:
            gem_group = np.empty((0, 3))
            searched_gems = np.empty((0, 3), dtype=int)
            if len(gem_groups) == 0:
                index = gem_indexes[0]
            else:
                flatten_groups = [x.tolist() for x in gem_groups]
                index_ls = [x for x in gem_indexes.tolist() if x not in [
                    m for y in flatten_groups for m in y]]
                if len(index_ls) == 0:
                    return gem_groups
                index = index_ls[0]
            gem_group = np.vstack((gem_group, index))
            gem_groups = self.search_gems(
                index, gem_group, searched_gems, gem_groups, gem_indexes)

    def make_wall_indexes(self) -> np.array:
        wall_indexes = []
        for row in range(self.height):
            for col in range(self.width):
                if self.map[row][col] == "W":
                    wall_indexes.append((row, col))
        return wall_indexes
