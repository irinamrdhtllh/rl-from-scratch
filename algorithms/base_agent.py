import pickle
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def update_parameters(self, state, action, reward, next_state):
        pass

    def save_to_disk(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_from_disk(cls, path):
        with open(path, "rb") as f:
            dump = pickle.load(f)

        return dump
