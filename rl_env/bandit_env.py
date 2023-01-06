import random
from loguru import logger

class BanditEnv:
    def __init__(self, earning_distribution: list):
        """
        :param earning_distribution: list of earning probability for each bandit
        """
        self.earning_distribution = earning_distribution

    def step(self, action: int) -> (int, int, bool, dict):
        """
        :param action: action to perform
        :return: (observation, reward, done, information)
        """
        done = True
        information = {}
        proba_action = self.earning_distribution[action]
        if random.random()<proba_action:
            reward = 1
        else:
            reward = 0
        observation = 0
        return observation, reward, done, information



if __name__ == '__main__':
    env = BanditEnv([0.1, 0.9])
    logger.warning(env.earning_distribution)
    for _ in range(10):

        r = env.step(1)
        logger.info(r)
