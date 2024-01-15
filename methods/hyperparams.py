import pandas as pd

from methods.loops import many_runs


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
