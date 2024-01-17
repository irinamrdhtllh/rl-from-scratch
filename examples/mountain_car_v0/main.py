import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt

from .loops import run
from algorithms.sarsa.agent import SarsaAgent


if __name__ == "__main__":
    # Tuned hyperparameters
    learning_rate = 0.1
    discount_factor = 0.9

    # Exploration-exploitation prob
    epsilon_train = 0.1

    # Number of episodes
    n_episodes_train = 10000
    n_episodes_test = 1000

    # Define the environment and the agent
    env = gym.make("MountainCar-v0")
    env._max_episode_steps = 1000
    agent = SarsaAgent(env, alpha=learning_rate, gamma=discount_factor)

    # Train the agent
    rewards_per_episode, max_positions_per_episode = run(
        env,
        agent,
        n_episodes=n_episodes_train,
        training=True,
        epsilon=epsilon_train,
    )

    # Plot training progress
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title("Rewards per episode")
    pd.Series(rewards_per_episode).plot(kind="line")
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title("Max positions per episode")
    pd.Series(max_positions_per_episode).plot(kind="line")
    plt.show()

    # Evaluate the agent
    rewards_per_episode, max_positions_per_episode = run(
        env,
        agent,
        n_episodes=n_episodes_test,
        training=False,
    )

    n_completed = sum([1 if pos > 0.5 else 0 for pos in max_positions_per_episode])

    print(f"{n_completed} success out of {n_episodes_test} attemps")
