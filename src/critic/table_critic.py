import random
from collections import defaultdict

from .critic import Critic


class TableCritic(Critic):
    """
    Table based Critic

    ...

    Attributes
    ----------

    Methods
    -------
    update(current_state, successor_state, reward):
        Updates value function, then eligibilities for each state in the episode.
    reset_eligibilities():
        Sets all eligibilities to 0.0
    replace_eligibilities(state, action):
        Replaces trace e(state) with 1.0
    """

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
        self.__values = defaultdict(lambda: (random.random() - 0.5) * 0.02)  # V(s)
        self.reset_eligibilities()

    def _get_value(self, state) -> float:
        """Value function V(s)"""
        return self.__values[state]

    def update(self, current_state, successor_state, reward) -> None:
        """Updates value function, then eligibilities for each state in the episode."""
        self.__update_values(current_state, successor_state, reward)
        self.__update_eligibilities()

    def __update_values(self, current_state, successor_state, reward) -> None:
        td_error = self.td_error(current_state, successor_state, reward)
        for state, eligibility in self.__eligibilities.items():
            self.__values[state] += self._learning_rate * td_error * eligibility

    def __update_eligibilities(self) -> None:
        for state in self.__eligibilities:
            self.__eligibilities[state] *= self._discount_factor * self._trace_decay

    def reset_eligibilities(self) -> None:
        """Sets all eligibilities to 0.0"""
        self.__eligibilities = defaultdict(float)

    def replace_eligibilities(self, state) -> None:
        """Replaces trace e(state) with 1.0"""
        self.__eligibilities[state] = 1
