import random

from tqdm import tqdm
from typing import Callable


def run(env, agent, n_episodes, training, epsilon=None):
    # For plotting metrics
    rewards_per_episode = []
    max_positions_per_episode = []

    for i in tqdm(range(0, n_episodes)):
        state = env.reset()[0]

        rewards = 0
        max_position = -99

        if epsilon is not None:
            if isinstance(epsilon, float):
                epsilon_ = epsilon
            if isinstance(epsilon, Callable):
                epsilon_ = epsilon(i)

        done = False
        while not done:
            if epsilon and random.uniform(0, 1) < epsilon_:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if training:
                agent.update_parameters(state, action, reward, next_state)

            rewards += reward

            if next_state[0] > max_position:
                max_position = next_state[0]

            state = next_state

        rewards_per_episode.append(rewards)
        max_positions_per_episode.append(max_position)

    return rewards_per_episode, max_positions_per_episode
