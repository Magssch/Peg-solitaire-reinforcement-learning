from abc import ABC, abstractmethod


class Critic(ABC):

    def __init__(
        self,
        learning_rate: float,
        discount_factor: float,
        trace_decay: float,
    ):
        self._learning_rate = learning_rate  # alpha
        self._discount_factor = discount_factor  # gamma
        self._trace_decay = trace_decay  # lambda

    def td_error(self, current_state, successor_state, reward) -> float:
        """Temporal difference (TD) error. Same as lowercase delta"""
        return reward + self._discount_factor * (self._get_value(successor_state) if reward != 1 else 0) - self._get_value(current_state)

    @abstractmethod
    def _get_value(self, state) -> float:
        raise NotImplementedError

    @abstractmethod
    def update(self, current_state, successor_state, reward) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset_eligibilities(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def replace_eligibilities(self, state) -> None:
        raise NotImplementedError

    def plot_training_data(self):
        raise NotImplementedError
