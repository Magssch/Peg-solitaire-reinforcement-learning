
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

        self.simple = simple
        self.dimensions = dimensions
