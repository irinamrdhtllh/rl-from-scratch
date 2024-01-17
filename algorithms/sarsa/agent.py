import numpy as np

from algorithms.base_agent import BaseAgent


class SarsaAgent(BaseAgent):
    def __init__(self, env, alpha, gamma):
        self.env = env
        self.q_table = self._init_q_table()
        self.learning_rate = alpha
        self.discount_factor = gamma

    def _init_q_table(self):
        # Discretize state space from a continuous to discrete
        high = self.env.observation_space.high
        low = self.env.observation_space.low
        n_states = (high - low) * np.array([10, 100])
        n_states = np.round(n_states, 0).astype(int) + 1

        n_actions = self.env.action_space.n

        return np.zeros([n_states[0], n_states[1], n_actions])

    def _discretize_state(self, state):
        min_states = self.env.observation_space.low
        discrete_state = (state - min_states) * np.array([10, 100])
        discrete_state = np.round(discrete_state, 0).astype(int)

        return discrete_state

    def get_action(self, state):
        state_ = self._discretize_state(state)
        return np.argmax(self.q_table[state_[0], state_[1]])

    def update_parameters(self, state, action, reward, next_state):
        state_ = self._discretize_state(state)
        next_state_ = self._discretize_state(next_state)
        action_ = self.get_action(next_state)

        # SARSA update formula
        delta = self.learning_rate * (
            reward
            + self.discount_factor
            * self.q_table[next_state_[0], next_state_[1], action_]
            - self.q_table[state_[0], state_[1], action]
        )
        self.q_table[state_[0], state_[1], action] += delta
