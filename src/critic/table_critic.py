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

    def update_values(self, current_state, successor_state, reward) -> None:
        td_error = self.td_error(current_state, successor_state, reward)
        for state, eligibility in self.__eligibilities.items():
            self.__values[state] += self._learning_rate * td_error * eligibility

    def reset_eligibilities(self) -> None:
        self.__eligibilities = defaultdict(float)

    def replace_eligibilities(self, state) -> None:
        self.__eligibilities[state] = 1

    def update_eligibilities(self) -> None:
        for state in self.__eligibilities:
            self.__eligibilities[state] *= self._discount_factor * self._trace_decay
