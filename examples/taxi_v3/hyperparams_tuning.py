import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns

from algorithms.q_learning import QLearningAgent
from methods.hyperparams import hyperparams_tuning


# List of hyperparameters
alphas = [0.01, 0.10, 1.00]
gammas = [0.10, 0.60, 0.90]

# Tune the hyperparameters
results = hyperparams_tuning(
    env=gym.make("Taxi-v3").env,
    agent=QLearningAgent,
    n_episodes=10000,
    n_runs=10,
    epsilon=0.1,
    alpha_list=alphas,
    gamma_list=gammas,
)

# Plot the result for all hyperparameter pairs
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.lineplot(data=results, x="episode", y="timesteps", hue="hyperparameters")
plt.show()
