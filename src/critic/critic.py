from abc import ABC, abstractmethod
from typing import Tuple


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

    def td_error(self, reward: float, successor_state: Tuple[int], current_state: Tuple[int]) -> float:
        """Temporal difference (TD) error. Same as lowercase delta"""
        terminalStateFactor = 1 - int(reward != 0)  # Ensures terminal state is always 0
        return reward + self._discount_factor * self._get_value(successor_state) * terminalStateFactor - self._get_value(current_state)

    @abstractmethod
    def _get_value(self, state: Tuple[int]) -> float:
        raise NotImplementedError

    @abstractmethod
    def update(self, reward: float, successor_state: Tuple[int], current_state: Tuple[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset_eligibilities(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def replace_eligibilities(self, state: Tuple[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def plot_training_data(self) -> None:
        raise NotImplementedError
