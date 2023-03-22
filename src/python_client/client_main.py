import datetime

from base import BaseAgent, Action
from model_based_policy import ModelBasedPolicy
from reinforce_learning import ReinforceLearning
from minimax1 import MiniMax1
from minimax2 import MiniMax2


class Agent(BaseAgent):
    # actions = [Action.RIGHT] + [Action.TELEPORT] * 100
    def do_turn(self) -> Action:
        now1 = datetime.datetime.now()
        phase3 = MiniMax1(self)
        action = phase3.main()
        now2 = datetime.datetime.now()
        print(f'total_seconds: {(now2 - now1).total_seconds()}')
        return action


if __name__ == '__main__':
    data = Agent().play()
    print("FINISH : ", data)
