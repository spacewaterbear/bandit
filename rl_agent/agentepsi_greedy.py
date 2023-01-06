import random
from loguru import logger
import numpy as np

from rl_agent.agent import Agent


class EpsiGreedyAgent(Agent):
    def __init__(self, eps: float, nb_action: int):
        super().__init__(eps, nb_action)
        self.nb_action = nb_action
        self.eps = eps
        self.V_historic = [[] for _ in range(10)]
        self.V = np.zeros(nb_action)
        self.name = "Epsi greedy Bandit Agent"


    def pull(self) -> int:
        """Epsilon greedy policy"""
        # random part
        if random.random() < self.eps:
            action = random.choice(list(range(self.nb_action)))
        # greedy part
        else:
            action = np.argmax(self.V)
        return action

    def update(self, action, reward) -> None:
        """Update value function"""
        self.V_historic[action].append(reward)
        self.V[action] = np.mean(self.V_historic[action])


