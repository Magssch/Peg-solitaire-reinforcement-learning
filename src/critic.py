
class Critic:

    def __init__(
        self,
        learning_rate,
        trace_decay,
        discount_factor,
        simple,
        dimensions,
    ):
        self.learning_rate = learning_rate  # alpha
        self.trace_decay = trace_decay  # lambda
        self.discount_factor = discount_factor  # gamma

        self.values = {}  # V
        self.eligibilities = {}
        self.simple = simple
        self.dimensions = dimensions

    def td_error(self, current_state, successor_state):
        r = 0  # TODO: add reward
        return r + self.gamma * self.values[successor_state] - self.values[current_state]

    def update_value(self, state, delta):
        self.values[state] += self.learning_rate * delta * self.eligibilities[state]

    def update_eligibility(self, state):
        self.eligibilities[state] *= self.discount_factor * self.trace_decay
