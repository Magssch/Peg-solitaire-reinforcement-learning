import random
from collections import defaultdict


class Actor:

    def __init__(
        self,
        learning_rate: float,
        discount_factor: float,
        trace_decay: float,
        epsilon: float,
        epsilon_decay: float,
    ):
        self.__learning_rate = learning_rate  # alpha
        self.__discount_factor = discount_factor  # gamma
        self.__trace_decay = trace_decay  # lambda

        self.__epsilon = epsilon
        self.__epsilon_decay = epsilon

        self.__policy = defaultdict(lambda: defaultdict(float))  # Pi(s, a)
        self.reset_eligibilities()

    def choose_action(self, state, possible_actions):
        """Epsilon-greedy action selection"""
        def choose_uniform(possible_actions):
            return random.choice(possible_actions)

        def choose_greedy(state, possible_actions):
            return max(possible_actions, key=lambda action: self.__policy[state][action])

        if random.random() < self.__epsilon:
            return choose_uniform(possible_actions)
        return choose_greedy(state, possible_actions)

    def update(self, td_error):
        self.__update_policy(td_error)
        self.__update_eligibilities()

    def __update_policy(self, td_error) -> None:
        self.__epsilon *= self.__epsilon_decay
        for state in self.__eligibilities:
            for action, eligibility in self.__eligibilities[state].items():
                self.__policy[state][action] += self.__learning_rate * td_error * eligibility

    def __update_eligibilities(self) -> None:
        for state in self.__eligibilities:
            for action in self.__eligibilities[state]:
                self.__eligibilities[state][action] *= self.__discount_factor * self.__trace_decay

    def reset_eligibilities(self) -> None:
        self.__eligibilities = defaultdict(lambda: defaultdict(float))

    def replace_eligibilities(self, state, action) -> None:
        self.__eligibilities[state][action] = 1
