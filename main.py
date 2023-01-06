import numpy as np
from matplotlib import pyplot as plt
from rl_agent.agent_random import RandomAgent
from rl_agent.agentepsi_greedy import EpsiGreedyAgent
from rl_env.bandit_env import BanditEnv
from variables import eps, earning_prob


def plot_reward(data_to_plot: dict, nb_episode: int):
    """plot cum reward among episodes"""
    plt.figure(figsize=(12, 8))
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    x = list(range(1, len(data_to_plot['data'][0]) + 1))
    for i, title in enumerate(data_to_plot['title']):
        plt.plot(
            x,
            [j for j in data_to_plot['data'][i]],
            label=title,
        )
    plt.legend()
    plt.title(f"Cumulative reward for {nb_episode} episodes")
    plt.show()


if __name__ == '__main__':
    env = BanditEnv(earning_distribution=earning_prob)
    nb_action = len(earning_prob)
    nb_episode = 10_000

    epsis = [0.01, 0.1, 0.3]
    agents_greedy = [EpsiGreedyAgent(eps=eps, nb_action=nb_action) for eps in epsis]
    agents = [RandomAgent(eps=eps, nb_action=nb_action)] + agents_greedy
    data_to_plot = {
        "title": [],
        "data": []
    }
    for agent in agents:
        rewards = []
        for _ in range(nb_episode):
            action = agent.pull()
            ob, r, done, info = env.step(action)
            agent.update(action, reward=r)
            rewards.append(r)

        cum_sum_r = np.cumsum(rewards)
        agent_name = agent.name
        title = f"{agent_name} eps: {agent.eps}"
        data_to_plot["title"].append(title)
        data_to_plot["data"].append(cum_sum_r)

    plot_reward(data_to_plot, nb_episode=nb_episode)
