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
        ANN_dimensions: tuple,
    ):
        self.learning_rate = learning_rate  # alpha
        self.trace_decay = trace_decay  # lambda
        self.discount_factor = discount_factor  # gamma

        self.values = {}  # V
        self.eligibilities = {}
        self.dimensions = ANN_dimensions

        if bool(ANN_dimensions):
            self.__build_critic_network(ANN_dimensions)

    def __build_critic_network(self, dimensions: tuple) -> None:
        input_dim, *hidden_dims, output_dim = dimensions

        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        for dimension in hidden_dims:
            model.add(Dense(dimension, activation='relu'))

        model.add(Dense(output_dim, activation='linear'))

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error'
        )
        self.value = model

    def td_error(self, current_state, successor_state, reward) -> float:
        return reward + self.discount_factor * self.values[successor_state] - self.values[current_state]

    def update_value(self, state, td_error) -> None:
        self.values[state] += self.learning_rate * td_error * self.eligibilities[state]

    def reset_traces(self) -> None:
        self.eligibilities = {}

    def replace_trace(self, state) -> None:
        self.eligibilities[state] = 1

    def update_trace(self, state) -> None:
        self.eligibilities[state] *= self.discount_factor * self.trace_decay


if __name__ == "__main__":
    critic = Critic(0.01, 0.9, 0.01, (15, 20, 30, 5, 1))
