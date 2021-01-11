
class Actor:

    def __init__(
        self,
        learning_rate,
        trace_decay,
        discount_factor,
        initial_epsilon,
    ):
        self.learning_rate = learning_rate  # alpha
        self.trace_decay = trace_decay  # lambda
        self.discount_factor = discount_factor  # gamma

        self.epsilon = initial_epsilon
        self.policy = {}  # Pi

    def update_value(self, state, action, delta):
        self.policy[state][action] += self.learning_rate * delta * self.eligibilities[state][action]

    def update_eligibility(self, state, action):
        self.eligibilities[state][action] *= self.discount_factor * self.trace_decay
