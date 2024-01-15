import random
import gymnasium as gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from algorithms.q_learning import QLearningAgent


def run(env, agent, n_episodes, epsilon, training):
    # For plotting metrics
    timesteps_per_episode = []
    penalties_per_episode = []

    for i in tqdm(range(0, n_episodes)):
        # Reset environment to a random state
        state = env.reset()[0]

        epochs, penalties, reward = 0, 0, 0
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                # Explore action space
                action = env.action_space.sample()
            else:
                # Exploit learned values
                action = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if training:
                agent.update_parameters(state, action, reward, next_state)

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        timesteps_per_episode.append(epochs)
        penalties_per_episode.append(penalties)

    return timesteps_per_episode, penalties_per_episode


if __name__ == "__main__":
    env = gym.make("Taxi-v3").env
    agent = QLearningAgent(env, alpha=0.1, gamma=0.6)

    # Exploration vs. exploitation prob
    train_epsilon = 0.1
    eval_epsilon = 0.05

    # Number of episodes
    train_episodes = 10000
    eval_episodes = 100

    # Train the agent
    timesteps_per_episode, penalties_per_episode = run(
        env, agent, n_episodes=train_episodes, epsilon=train_epsilon, training=True
    )

    # Plot the training process
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title("Timesteps to complete ride")
    pd.Series(timesteps_per_episode).plot(kind="line")
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title("Penalties per ride")
    pd.Series(penalties_per_episode).plot(kind="line")
    plt.show()

    # Evaluate the agent
    timesteps_per_episode, penalties_per_episode = run(
        env, agent, n_episodes=eval_episodes, epsilon=eval_epsilon, training=False
    )

    print(np.array(timesteps_per_episode).mean())
    print(np.array(penalties_per_episode).mean())
