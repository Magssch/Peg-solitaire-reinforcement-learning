from collections import defaultdict

class Actor:

    def __init__(
        self,
        learning_rate: float,
        discount_factor: float,
        trace_decay: float,
    ):
        self.__learning_rate = learning_rate  # alpha
        self.__discount_factor = discount_factor  # gamma
        self.__trace_decay = trace_decay  # lambda

        self.__policy = defaultdict(lambda: defaultdict(float))  # Pi(s, a)
        self.reset_eligibilities()

    def __boltzmann_scale(self, state, action):
        pass  # TODO: implement

    def update_policy(self, state, action, td_error) -> None:
        self.__policy[state][action] += self.__learning_rate * td_error * self.__eligibilities[state][action]

    def reset_eligibilities(self) -> None:
        self.__eligibilities = defaultdict(lambda: defaultdict(float))

    def replace_eligibilities(self, state, action) -> None:
        self.__eligibilities[state][action] = 1

    def update_eligibilities(self, state, action) -> None:
        self.__eligibilities[state][action] *= self.__discount_factor * self.__trace_decay

    def choose_action(self, state):
        return (3, 1)
