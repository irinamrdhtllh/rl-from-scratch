import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns


from algorithms.q_learning import QLearningAgent
from methods import hyperparams_tuning


if __name__ == "__main__":
    env = gym.make("Taxi-v3").env
    agent = QLearningAgent

    # hyperparameters
    alphas = [0.01, 0.1, 1]
    gammas = [0.1, 0.6, 0.9]

    results = hyperparams_tuning(
        env,
        agent,
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
