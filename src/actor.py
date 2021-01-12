import random

import numpy as np


class Actor:

    def __init__(
        self,
        learning_rate: float,
        trace_decay: float,
        discount_factor: float,
        epsilon: float,
    ):
        self.learning_rate = learning_rate  # alpha
        self.trace_decay = trace_decay  # lambda
        self.discount_factor = discount_factor  # gamma

        self.epsilon = epsilon
        self.policy = {}  # Pi

    def update_policy(self, state, action, delta):
        self.policy[state][action] += self.learning_rate * delta * self.eligibilities[state][action]

    def reset_traces(self):
        self.eligibilities = {}

    def replace_trace(self, state, action):
        self.eligibilities[state][action] = 1

    def update_trace(self, state, action):
        self.eligibilities[state][action] *= self.discount_factor * self.trace_decay

    def boltzmann_scale(self, state, action):
        p = np.e ** self.policy[state][action]
        q = sum([np.e ** self.policy[state][action_i] for action_i in []])  # TODO: add get_actions(state) here
        return p / q

    def choose_greedy(self, state):
        actions = []  # TODO: add get_actions(state) here
        utilities = {action: self.policy[state][action] for action in actions}
        return max(utilities, key=utilities.get)

    def choose_random(self, state):
        actions = []  # TODO: add get_actions(state) here
        return random.choice(actions)

    def choose_mixed(self, state):
        if random.random() < self.epsilon:
            return self.choose_random(state)
        return self.choose_greedy(state)
