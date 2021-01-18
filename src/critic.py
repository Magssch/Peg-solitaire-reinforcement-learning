from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam


class Critic:

    def __init__(
        self,
        learning_rate: float,
        trace_decay: float,
        discount_factor: float,
        nn_dimensions: tuple = None,
    ):
        self.learning_rate = learning_rate  # alpha
        self.trace_decay = trace_decay  # lambda
        self.discount_factor = discount_factor  # gamma

        self.values = {}  # V
        self.reset_eligibilities()
        self.nn_dimensions = nn_dimensions

        if nn_dimensions is not None:
            self.model = self._build_critic_network(nn_dimensions)

    def _build_critic_network(self, nn_dimensions: tuple) -> Sequential:
        input_dim, *hidden_dims, output_dim = nn_dimensions

        assert output_dim == 1

        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        for dimension in hidden_dims:
            model.add(Dense(dimension, activation='relu'))

        model.add(Dense(units=1, activation='linear'))

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error'
        )
        model.summary()
        return model

    def get_values(self, state) -> float:
        if self.nn_dimensions is not None:
            return np.squeeze(self.model(state))
        return self.values[state]

    def td_error(self, current_state, successor_state, reward) -> float:
        return reward + self.discount_factor * self.get_values(successor_state) - self.get_values(current_state)

    def update_value(self, state, td_error) -> None:
        self.values[state] += self.learning_rate * td_error * self.eligibilities[state]

    def reset_eligibilities(self) -> None:
        if self.nn_dimensions is not None:
            input_dim, *hidden_dims, _ = self.nn_dimensions
            self.eligibilities = [np.zeros(input_dim)]
            for dimension in hidden_dims:
                self.eligibilities.append(np.zeros(dimension))
        self.eligibilities = defaultdict(float)

    def replace_trace(self, state) -> None:
        self.eligibilities[state] = 1

    def update_trace(self, state) -> None:
        self.eligibilities[state] *= self.discount_factor * self.trace_decay  # + nabla V(s) for NNs


if __name__ == "__main__":
    critic = Critic(0.01, 0.9, 0.01)
    critic.reset_eligibilities()
