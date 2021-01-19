from .critic import Critic
import random
from collections import defaultdict


class TableCritic(Critic):

    def __init__(
        self,
        learning_rate: float,
        trace_decay: float,
        discount_factor: float,
    ):
        super().__init__(
            learning_rate,
            trace_decay,
            discount_factor,
        )
        self.__values = defaultdict(lambda: random.random() * 0.2)

    def get_value(self, state) -> float:
        return self.__values[state]

    def update_value(self, state, td_error) -> None:
        self.__values[state] += self.learning_rate * td_error * self.eligibilities[state]

    def reset_eligibilities(self) -> None:
        self.eligibilities = defaultdict(float)

    def replace_eligibilities(self, state) -> None:
        self.eligibilities[state] = 1

    def update_eligibilities(self, state) -> None:
        self.eligibilities[state] *= self.discount_factor * self.trace_decay
