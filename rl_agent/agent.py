import numpy as np


class Agent:
    def __init__(self, eps: float, nb_action: int):
        self.nb_action = nb_action
        self.eps = eps
        self.V_historic = [[] for _ in range(10)]
        self.V = np.zeros(nb_action)

    def pull(self) -> int:
        raise NotImplementedError("Please code this in children Class")

    def update(self, action, reward):
        pass