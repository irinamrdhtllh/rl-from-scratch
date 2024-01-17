import random
import numpy as np

from tqdm import tqdm


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
            env, agent, n_episodes, epsilon, training
        )

    timesteps_per_episode = np.mean(timesteps, axis=0).tolist()
    penalties_per_episode = np.mean(penalties, axis=0).tolist()

    return timesteps_per_episode, penalties_per_episode
