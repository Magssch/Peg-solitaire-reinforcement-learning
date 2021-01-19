import random
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam

from environment import Environment


class CriticFactory:

    @classmethod
    def get_critic(cls, env: Environment):
        if env.critic_nn_dimensions is None:
            return TableCritic(
                env.critic_learning_rate,
                env.critic_trace_decay,
                env.critic_discount_factor,
                None,
            )
        else:
            return NNCritic(
                env.critic_learning_rate,
                env.critic_trace_decay,
                env.critic_discount_factor,
                env.critic_nn_dimensions,
            )


class Critic(ABC):

    def __init__(
        self,
        learning_rate: float,
        trace_decay: float,
        discount_factor: float,
    ):
        self.__learning_rate = learning_rate  # alpha
        self.__trace_decay = trace_decay  # lambda
        self.__discount_factor = discount_factor  # gamma

    def td_error(self, current_state, successor_state, reward) -> float:
        return reward + self.__discount_factor * self.get_value(successor_state) - self.get_value(current_state)

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


class TableCritic(Critic):

    def __init__(
        self,
        learning_rate: float,
        trace_decay: float,
        discount_factor: float,
        nn_dimensions: tuple = None
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
        self.__values[state] += self.__learning_rate * td_error * self.eligibilities[state]

    def reset_eligibilities(self) -> None:
        self.eligibilities = defaultdict(float)

    def replace_eligibilities(self, state) -> None:
        self.eligibilities[state] = 1

    def update_eligibilities(self, state) -> None:
        self.eligibilities[state] *= self.__discount_factor * self.__trace_decay


class NNCritic(Critic):

    def __init__(
        self,
        learning_rate: float,
        trace_decay: float,
        discount_factor: float,
        nn_dimensions: tuple
    ):
        super().__init__(
            learning_rate,
            trace_decay,
            discount_factor,
        )
        self.__nn_dimensions = nn_dimensions
        assert nn_dimensions is not None
        self.values = self.__build_critic_network(nn_dimensions)

    def __build_critic_network(self, nn_dimensions: tuple) -> Sequential:
        input_dim, *hidden_dims, output_dim = nn_dimensions

        assert output_dim == 1

        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        for dimension in hidden_dims:
            model.add(Dense(dimension, activation='relu'))

        model.add(Dense(units=1, activation='linear'))

        model.compile(
            optimizer=Adam(learning_rate=self.__learning_rate),
            loss='mean_squared_error'
        )
        model.summary()
        return model

    def get_value(self, state) -> float:
        return np.squeeze(self.values(state))

    def td_error(self, current_state, successor_state, reward) -> float:
        return reward + self.__discount_factor * self.get_value(successor_state) - self.get_value(current_state)

    def update_value(self, state, td_error) -> None:
        pass

    def reset_eligibilities(self) -> None:
        input_dim, *hidden_dims, _ = self.__nn_dimensions
        self.eligibilities = [np.zeros(input_dim)]
        for dimension in hidden_dims:
            self.eligibilities.append(np.zeros(dimension))

    def replace_eligibilities(self, state) -> None:
        pass

    def update_eligibilities(self, state) -> None:
        pass


if __name__ == "__main__":
    env = Environment()
    critic = CriticFactory.get_critic(env)
    critic.reset_eligibilities()
