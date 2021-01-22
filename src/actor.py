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

    def choose_action(self, state):
        return (3, 1)

    def update_policy(self, td_error) -> None:
        for state in self.__eligibilities:
            for action, eligibility in self.__eligibilities[state].items():
                self.__policy[state][action] += self.__learning_rate * td_error * eligibility

    def reset_eligibilities(self) -> None:
        self.__eligibilities = defaultdict(lambda: defaultdict(float))

    def replace_eligibilities(self, state, action) -> None:
        self.__eligibilities[state][action] = 1

    def update_eligibilities(self) -> None:
        for state in self.__eligibilities:
            for action in self.__eligibilities[state]:
                self.__eligibilities[state][action] *= self.__discount_factor * self.__trace_decay
