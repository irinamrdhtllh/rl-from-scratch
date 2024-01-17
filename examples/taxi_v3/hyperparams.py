import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from loops import many_runs
from algorithms.q_learning.agent import QLearningAgent


def hyperparams_tuning(env, agent, n_episodes, n_runs, epsilon, alpha_list, gamma_list):
    results = pd.DataFrame()
    for alpha in alpha_list:
        for gamma in gamma_list:
            print(f"learning rate: {alpha}, discount factor: {gamma}")

            agent_ = agent(env, alpha=alpha, gamma=gamma)

            timesteps_per_episode, penalties_per_episode = many_runs(
                env, agent_, n_episodes, n_runs, epsilon, training=True
            )

            # Collect results for this pair of hyperparameters
            results_ = pd.DataFrame()
            results_["timesteps"] = timesteps_per_episode
            results_["penalties"] = penalties_per_episode
            results_["learning rate"] = alpha
            results_["discount factor"] = gamma
            results = pd.concat([results, results_])

    results = results.reset_index().rename(columns={"index": "episode"})

    # Add column with the 2 hyperparameters
    results["hyperparameters"] = [
        f"learning rate = {a}, discount factor = {g}"
        for (a, g) in zip(results["learning rate"], results["discount factor"])
    ]

    return results


if __name__ == "__main__":
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
