import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from methods.loops import run
from algorithms.q_learning import QLearningAgent


# Tuned hyperparameters
learning_rate = 1.0
discount_factor = 0.9

# Exploration-exploitation prob
epsilon_train = 0.1
epsilon_test = 0.05

# Number of episodes
n_episodes_train = 10000
n_episodes_test = 100

# Define the environment and the agent
env = gym.make("Taxi-v3").env
agent = QLearningAgent(env, alpha=learning_rate, gamma=discount_factor)

# Train the agent
timesteps_per_episode, penalties_per_episode = run(
    env,
    agent,
    n_episodes=n_episodes_train,
    epsilon=epsilon_train,
    training=True,
)

# Plot training progress
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
    env,
    agent,
    n_episodes=n_episodes_test,
    epsilon=epsilon_test,
    training=False,
)

print(f"Avg timesteps to complete ride: {np.array(timesteps_per_episode).mean()}")
print(f"Avg penalties per ride: {np.array(penalties_per_episode).mean()}")
