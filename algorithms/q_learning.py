import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class QLearningAgent:
    def __init__(self, env, alpha, gamma):
        self.env = env
        # Table with q-values (n_states * n_actions)
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        # hyper-parameters
        self.learning_rate = alpha
        self.discount_factor = gamma

    def get_action(self, state):
        return np.argmax(self.q_table[state])

    def update_parameters(self, state, action, reward, next_state):
        # Q-learning formula
        old_value = self.q_table[state, action]
        next_max_value = np.max(self.q_table[next_state])
        new_value = old_value + self.learning_rate * (
            reward + self.discount_factor * next_max_value - old_value
        )

        # Update the q-table
        self.q_table[state, action] = new_value
