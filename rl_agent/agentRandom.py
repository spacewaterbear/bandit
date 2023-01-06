import random
from loguru import logger
import numpy as np

from rl_agent.agent import Agent


class RandomAgent(Agent):
    def __init__(self, eps: float, nb_action: int):
        super().__init__(eps, nb_action)
        self.nb_action = nb_action
        self.name = "Random Agent"


    def pull(self) -> int:
        """Epsilon greedy policy"""
        action = random.choice(list(range(self.nb_action)))
        return action


