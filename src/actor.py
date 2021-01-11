
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
        self.policies = {}

    def get_utility(self, s, a):
        return self.policies[s][a]
