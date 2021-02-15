import random
from collections import defaultdict
from typing import Tuple

from visualize import Visualize

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
        self.__value_history = []
        self.reset_eligibilities()

    def _get_value(self, state) -> float:
        """Value function V(s)"""
        return self.__values[state]

    def update(self, current_state, successor_state, reward) -> None:
        """Updates value function, then eligibilities for each state in the episode."""
        self.__update_values(current_state, successor_state, reward)
        self.__update_eligibilities()

    def __update_values(self, reward: float, successor_state: Tuple[int], current_state: Tuple[int]) -> None:
        td_error = self.td_error(reward, successor_state, current_state)
        for state, eligibility in self.__eligibilities.items():
            self.__values[state] += self._learning_rate * td_error * eligibility

    def __update_eligibilities(self) -> None:
        for state in self.__eligibilities:
            self.__eligibilities[state] *= self._discount_factor * self._trace_decay

    def reset_eligibilities(self) -> None:
        """Sets all eligibilities to 0.0"""
        self.__eligibilities = defaultdict(float)

        # Used for plotting:
        if len(self.__values) != 0:
            self.__value_history.append(self.__values[max(self.__values, key=lambda state: abs(self._get_value(state)))])

    def replace_eligibilities(self, state: Tuple[int]) -> None:
        """Replaces trace e(state) with 1.0"""
        self.__eligibilities[state] = 1

    def plot_training_data(self) -> None:
        Visualize.plot_value_history(self.__value_history)
