from base import Action
import numpy as np
from typing import Union
import re
import datetime
import warnings

from utils.config import GEMS


class MiniMax1:
    def __init__(self, agent):
        self.agent = agent
        self.map = np.array(self.agent.grid)
        self.height = self.agent.grid_height
        self.width = self.agent.grid_width
        if not hasattr(self.agent, 'wall_indexes'):
            self.agent.wall_indexes = self.make_wall_indexes()
        if not hasattr(self.agent, 'barbed_indexes'):
            self.agent.barbed_indexes = self.make_barbed_indexes()
        if not hasattr(self.agent, 'key_indexes'):
            self.agent.key_indexes = self.make_key_indexes()
        if not hasattr(self.agent, 'door_indexes'):
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
        if not hasattr(self.agent, 'keys'):
            self.agent.keys = {'r': 0,
                               'y': 0,
                               'g': 0}

    def make_barbed_indexes(self) -> list:
        barbed_indexes = []  # row, col
        for row in range(self.height):
            for col in range(self.width):
                if self.map[row][col] == "*":
                    barbed_indexes.append((row, col))
        return barbed_indexes

    def make_door_indexes(self) -> list:
        door_indexes = []  # row, col

        for row in range(self.height):
            for col in range(self.width):
                if self.map[row][col] == "R":
                    door_indexes.append((row, col))
                if self.map[row][col] == "G":
                    door_indexes.append((row, col))
                if self.map[row][col] == "Y":
                    door_indexes.append((row, col))
        return door_indexes

    def make_key_indexes(self) -> list:
        key_indexes = []  # row, col
        for row in range(self.height):
            for col in range(self.width):
                if self.map[row][col] == "r":
                    key_indexes.append((row, col))
                if self.map[row][col] == "g":
                    key_indexes.append((row, col))
                if self.map[row][col] == "y":
                    key_indexes.append((row, col))

        return key_indexes

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
        wall_indexes = []  # row, col

        for row in range(self.height):
            for col in range(self.width):
                if self.map[row][col] == "W":
                    wall_indexes.append((row, col))
        return wall_indexes

    @staticmethod
    def calc_gems_scores(gem: str, prev_gem: str) -> int:
        if prev_gem is None:
            if gem == GEMS['YELLOW_GEM']:
                return 50
            else:
                return 0
        elif prev_gem == GEMS['YELLOW_GEM']:
            if gem == GEMS['YELLOW_GEM']:
                return 50
            elif gem == GEMS['GREEN_GEM']:
                return 200
            elif gem == GEMS['RED_GEM']:
                return 100
            else:
                return 0
        elif prev_gem == GEMS['GREEN_GEM']:
            if gem == GEMS['YELLOW_GEM']:
                return 100
            elif gem == GEMS['GREEN_GEM']:
                return 50
            elif gem == GEMS['RED_GEM']:
                return 200
            else:
                return 100
        elif prev_gem == GEMS['RED_GEM']:
            if gem == GEMS['YELLOW_GEM']:
                return 50
            elif gem == GEMS['GREEN_GEM']:
                return 100
            elif gem == GEMS['RED_GEM']:
                return 50
            else:
                return 200
        else:
            if gem == GEMS['YELLOW_GEM']:
                return 250
            elif gem == GEMS['GREEN_GEM']:
                return 50
            elif gem == GEMS['RED_GEM']:
                return 100
            else:
                return 50

    def transition_model(self, action, state) -> Union[tuple, None]:
        # return None for impossible action and wall
        # print(state)
        (i, j) = state
        next_state = ()
        if action == 'UP':
            if i != 0:
                next_state = (i - 1, j)
            else:
                return None

        elif action == 'DOWN':
            if i != self.height - 1:
                next_state = (i + 1, j)
            else:
                return None
        elif action == 'LEFT':
            if j != 0:
                next_state = (i, j - 1)
            else:
                return None
        elif action == 'RIGHT':
            if j != self.width - 1:
                next_state = (i, j + 1)
            else:
                return None
        elif action == 'DOWN_RIGHT':
            if i != self.height - 1 and j != self.width - 1:
                next_state = (i + 1, j + 1)
            else:
                return None
        elif action == 'DOWN_LEFT':
            if i != self.height - 1 and j != 0:
                next_state = (i + 1, j - 1)
            else:
                return None
        elif action == 'UP_LEFT':
            if i != 0 and j != 0:
                next_state = (i - 1, j - 1)
            else:
                return None
        elif action == 'UP_RIGHT':
            if i != 0 and j != self.width - 1:
                next_state = (i - 1, j + 1)
            else:
                return None
        elif action == 'NOOP':
            next_state = (i, j)

        warnings.simplefilter(action='ignore', category=FutureWarning)
        if next_state not in self.agent.wall_indexes and next_state not in self.agent.door_indexes:
            return next_state

        return None

    def is_action_left(self, state, max_turn):
        if max_turn:
            for act in self.actions:
                if self.transition_model(act, state) is not None and self.transition_model(act,
                                                                                           state) not in self.visited_indexes_A:
                    return True
        else:
            for act in self.actions:
                if self.transition_model(act, state) is not None and self.transition_model(act,
                                                                                           state) not in self.visited_indexes_B:
                    return True
        return False

    def is_terminal(self, state, max_turn) -> bool:
        self.agent.gem_indexes = self.make_gem_indexes()
        if len(self.agent.gem_indexes) == 0:
            return True
        if self.is_action_left(state, max_turn):
            return False
        return True

    def is_agent_nearby_gem(self, state):
        flag = False
        action = None
        for act in self.actions:
            if self.transition_model(act, state) is not None:
                next_state = self.transition_model(act, state)
                (i, j) = next_state
                if (i, j) in self.agent.gem_indexes:
                    flag = True
                    action = act
        return flag, action

    def find_best_action(self):
        best_score = -1000
        best_action = 'NOOP'
        i = self.get_agent_index('A')[0]
        j = self.get_agent_index('A')[1]
        state_a = (i, j)
        init_state = state_a
        i = self.get_agent_index('B')[0]
        j = self.get_agent_index('B')[1]
        if self.is_terminal(state_a, True):
            return 'NOOP'
        state_b = (i, j)
        flag, action = self.is_agent_nearby_gem(state_a)
        if flag:
            return action
        max_turn = False
        self.now1 = datetime.datetime.now()
        for action in self.actions:
            self.agent_A_score = 0
            self.agent_B_score = 0

            if self.transition_model(action, init_state) is not None:
                state_a = self.transition_model(action, init_state)
                (i, j) = state_a
                self.agent.gem_indexes = self.make_gem_indexes()
                if (i, j) in self.agent.gem_indexes:
                    self.map[i][j] = f'A{self.map[i][j]}'
                    self.agent_A_score += 1000
                self.agent_A_score += -1
                self.visited_indexes_A = [init_state]
                self.visited_indexes_B = []
                print('start :', init_state)
                print('action :', action)
                score = self.minimax(state_a, state_b, max_turn)
                # print("list a :" , self.visited_indexes_A)
                # print("list b :", self.visited_indexes_B)
                self.map[i][j] = np.array(self.agent.grid)[i][j]
                self.agent.gem_indexes = self.make_gem_indexes()
                if (i, j) in self.agent.gem_indexes:
                    self.agent_A_score += -1000
                self.agent_A_score += 1
                print('score : ', score)
                print(init_state, action, state_a, state_b, max_turn)
                print('------------------------------------------------------')
                if score > best_score:
                    best_action = action
                    best_score = score
        print("act : ", best_action)
        return best_action

    def minimax(self, state_a, state_b, max_turn) -> int:
        """"
        Main function
        """
        # cutoff test
        now2 = datetime.datetime.now()
        if (now2 - self.now1).total_seconds() > 0.9:
            print("cutoff test", len(self.visited_indexes_A))
            return self.heuristic()
        if max_turn:
            if self.is_terminal(state_a, max_turn):
                return self.heuristic()
        else:
            if self.is_terminal(state_b, max_turn):
                return self.heuristic()
        if max_turn:
            best = -1000
            self.visited_indexes_A.append(state_a)
            init_state = state_a
            for act in self.actions:
                if self.transition_model(act, init_state) is not None and self.transition_model(act,
                                                                                                init_state) not in self.visited_indexes_A:
                    state_a = self.transition_model(act, init_state)
                    (i, j) = state_a
                    self.agent.gem_indexes = self.make_gem_indexes()
                    if (i, j) in self.agent.gem_indexes:
                        self.map[i][j] = f'A{self.map[i][j]}'
                        self.agent_A_score += 1000

                    self.agent_A_score += -1

                    best = max(best, self.minimax(state_a, state_b, not max_turn))
                    self.map[i][j] = np.array(self.agent.grid)[i][j]
                    self.agent.gem_indexes = self.make_gem_indexes()
                    if (i, j) in self.agent.gem_indexes:
                        self.agent_A_score += -1000

                    self.agent_A_score += 1

            return best
        else:
            best = 1000
            self.visited_indexes_B.append(state_b)
            init_state = state_b
            for act in self.actions:
                if self.transition_model(act, init_state) is not None and self.transition_model(act,
                                                                                                init_state) not in self.visited_indexes_B:
                    state_b = self.transition_model(act, init_state)
                    (i, j) = state_b
                    self.agent.gem_indexes = self.make_gem_indexes()
                    if (i, j) in self.agent.gem_indexes:
                        self.map[i][j] = f'B{self.map[i][j]}'
                        self.agent_B_score += 1000
                    self.agent_B_score += -1
                    best = min(best, self.minimax(state_a, state_b, not max_turn))
                    self.map[i][j] = np.array(self.agent.grid)[i][j]
                    self.agent.gem_indexes = self.make_gem_indexes()
                    if (i, j) in self.agent.gem_indexes:
                        self.agent_B_score += -1000
                    self.agent_B_score += 1

            return best

    @staticmethod
    def perform_action(action: str):
        if action == 'UP':
            return Action.UP

        elif action == 'DOWN':
            return Action.DOWN

        elif action == 'LEFT':
            return Action.LEFT

        elif action == 'RIGHT':
            return Action.RIGHT

        elif action == 'DOWN_RIGHT':
            return Action.DOWN_RIGHT

        elif action == 'DOWN_LEFT':
            return Action.DOWN_LEFT

        elif action == 'UP_LEFT':
            return Action.UP_LEFT

        elif action == 'UP_RIGHT':
            return Action.UP_RIGHT
        elif action == 'NOOP':
            return Action.NOOP

    def heuristic(self) -> int:
        """
        Calculates score of the terminal state
        """

        for row in range(self.height):
            for col in range(self.width):
                list_a = re.findall(f'[A][1-4]', self.map[row][col])
                self.agent_A_score += len(list_a)
                list_a_b = re.findall(f'[B][1-4]', self.map[row][col])
                self.agent_B_score += len(list_a_b)
                # if self.map[row][col] == "1":
        return self.agent_A_score - self.agent_B_score

    def main(self):
        action = self.find_best_action()
        return self.perform_action(action)
