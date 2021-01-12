

class Critic:

    def __init__(
        self,
        learning_rate: float,
        trace_decay: float,
        discount_factor: float,
        value_function: str,
        ANN_dimentions: tuple,
    ):
        self.learning_rate = learning_rate  # alpha
        self.trace_decay = trace_decay  # lambda
        self.discount_factor = discount_factor  # gamma

        self.values = {}  # V
        self.eligibilities = {}
        self.value_function = value_function
        self.dimensions = ANN_dimentions

    def td_error(self, current_state, successor_state, reward):
        return reward + self.discount_factor * self.values[successor_state] - self.values[current_state]

    def update_value(self, state, td_error):
        self.values[state] += self.learning_rate * td_error * self.eligibilities[state]

    def reset_traces(self):
        self.eligibilities = {}

    def replace_trace(self, state):
        self.eligibilities[state] = 1

    def update_trace(self, state):
        self.eligibilities[state] *= self.discount_factor * self.trace_decay
