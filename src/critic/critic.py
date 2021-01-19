from abc import ABC, abstractmethod


class Critic(ABC):

    def __init__(
        self,
        learning_rate: float,
        trace_decay: float,
        discount_factor: float,
    ):
        self.learning_rate = learning_rate  # alpha
        self.trace_decay = trace_decay  # lambda
        self.discount_factor = discount_factor  # gamma

    def td_error(self, current_state, successor_state, reward) -> float:
        return reward + self.discount_factor * self.get_value(successor_state) - self.get_value(current_state)

    @abstractmethod
    def get_value(self, state) -> float:
        pass

    @abstractmethod
    def update_value(self, state, td_error) -> None:
        pass

    @abstractmethod
    def reset_eligibilities(self) -> None:
        pass

    @abstractmethod
    def replace_eligibilities(self, state) -> None:
        pass

    @abstractmethod
    def update_eligibilities(self, state) -> None:
        pass
