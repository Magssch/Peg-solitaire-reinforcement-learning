import numpy as np
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam

from .critic import Critic


class NNCritic(Critic):

    def __init__(
        self,
        learning_rate: float,
        discount_factor: float,
        trace_decay: float,
        nn_dimensions: tuple
    ):
        super().__init__(
            learning_rate,  # alpha
            discount_factor,  # gamma
            trace_decay,  # lambda
        )
        assert nn_dimensions is not None
        self.__nn_dimensions = nn_dimensions
        self.__values = self.__build_critic_network(nn_dimensions)  # V(s)

    def __build_critic_network(self, nn_dimensions: tuple) -> Sequential:
        input_dim, *hidden_dims, output_dim = nn_dimensions

        assert output_dim == 1

        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        for dimension in hidden_dims:
            model.add(Dense(dimension, activation='relu'))

        model.add(Dense(units=1, activation='linear'))

        model.compile(
            optimizer=Adam(learning_rate=self._learning_rate),
            loss='mean_squared_error'
        )
        model.summary()
        return model

    def get_value(self, state) -> float:
        return np.squeeze(self.__values(state))

    def td_error(self, current_state, successor_state, reward) -> float:
        return reward + self._discount_factor * self.get_value(successor_state) - self.get_value(current_state)

    def update_value(self, state, td_error) -> None:
        pass

    def reset_eligibilities(self) -> None:
        input_dim, *hidden_dims, _ = self.__nn_dimensions
        self.eligibilities = [np.zeros(input_dim)]
        for dimension in hidden_dims:
            self.eligibilities.append(np.zeros(dimension))
        print(self.eligibilities)
        for params in self.__values.trainable_weights:
            print(params)

    def replace_eligibilities(self, state) -> None:
        pass

    def update_eligibilities(self, state) -> None:
        pass
