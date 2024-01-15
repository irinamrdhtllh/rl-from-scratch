import random
import gymnasium as gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from algorithms.q_learning import QLearningAgent


def hyperparams_tuning(env, n_episodes, n_runs, epsilon, alpha_list, gamma_list):
    results = pd.DataFrame()
    for alpha in alpha_list:
        for gamma in gamma_list:
            print(f"learning rate: {alpha}, discount factor: {gamma}")

            agent = QLearningAgent(env, alpha=alpha, gamma=gamma)

            timesteps_per_episode, penalties_per_episode = many_runs(
                env, agent, n_episodes, n_runs, epsilon, training=True
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


def many_runs(env, agent, n_episodes, n_runs, epsilon, training):
    timesteps = np.zeros(shape=(n_runs, n_episodes))
    penalties = np.zeros(shape=(n_runs, n_episodes))

    for i in range(0, n_runs):
        agent.reset()

        timesteps[i, :], penalties[i, :] = run(
            env, agent, n_episodes, epsilon, training=True
        )

    timesteps_per_episode = np.mean(timesteps, axis=0).tolist()
    penalties_per_episode = np.mean(penalties, axis=0).tolist()

    return timesteps_per_episode, penalties_per_episode


if __name__ == "__main__":
    env = gym.make("Taxi-v3").env

    # hyperparameters
    alphas = [0.01, 0.1, 1]
    gammas = [0.1, 0.6, 0.9]

    results = hyperparams_tuning(
        env,
        n_episodes=10000,
        n_runs=10,
        epsilon=0.1,
        alpha_list=alphas,
        gamma_list=gammas,
    )

    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    sns.lineplot(data=results, x="episode", y="timesteps", hue="hyperparameters")
    plt.show()
