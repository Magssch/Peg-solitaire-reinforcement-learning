import random
from collections import defaultdict

from .critic import Critic


class TableCritic(Critic):

    def __init__(
        self,
        learning_rate: float,
        discount_factor: float,
        trace_decay: float,
    ):
        super().__init__(
            learning_rate,  # alpha
            discount_factor,  # gamma
            trace_decay,  # lambda
        )
        self.__values = defaultdict(lambda: random.random() * 0.2)  # V(s)
        self.reset_eligibilities()

    def get_value(self, state) -> float:
        return self.__values[state]

    def update_value(self, current_state, successor_state, reward) -> None:
        td_error = self.td_error(current_state, successor_state, reward)
        self.__values[current_state] += self._learning_rate * td_error * self.__eligibilities[current_state]

    def reset_eligibilities(self) -> None:
        self.__eligibilities = defaultdict(float)

    def replace_eligibilities(self, state) -> None:
        self.__eligibilities[state] = 1

    def update_eligibilities(self, state) -> None:
        self.__eligibilities[state] *= self._discount_factor * self._trace_decay
