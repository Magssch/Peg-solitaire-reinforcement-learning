import random

import numpy as np
from keras import backend as K
from keras.layers import Dense, Input
from keras.optimizers import Adam
from tensorflow.python.keras.models import Model


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

    def __build_actor_network(
        self,
        input_dim: int,
        dense_1_dim: int,
        dense_2_dim: int,
        n_actions: int,
    ) -> Model:
        input = Input(shape=(input_dim,))
        td_error = Input(shape=[1])
        dense_1 = Dense(dense_1_dim, activation='relu')(input)
        dense_2 = Dense(dense_2_dim, activation='relu')(dense_1)
        probabilities = Dense(n_actions, activation='softmax')(dense_2)

        model = Model(input=[input, td_error], output=[probabilities])
        model.compile(optimizer=Adam(
            learning_rate=self.learning_rate),
            loss='mean_squared_error'
        )
        return model

    def update_policy(self, state, action, td_error) -> None:
        self.policy[state][action] += self.learning_rate * td_error * self.eligibilities[state][action]

    def reset_traces(self) -> None:
        self.eligibilities = {}

    def replace_trace(self, state, action) -> None:
        self.eligibilities[state][action] = 1

    def update_trace(self, state, action) -> None:
        self.eligibilities[state][action] *= self.discount_factor * self.trace_decay

    def boltzmann_scale(self, state, action):
        p = np.e ** self.policy[state][action]
        q = sum([np.e ** self.policy[state][action_i] for action_i in []])  # TODO: add get_actions(state) here
        return p / q

    def choose_greedy(self, state):
        actions = []  # TODO: add get_actions(state) here
        preferences = [self.policy[state][action] for action in actions]
        return actions[np.argmax(preferences)]

    def choose_random(self, state):
        actions = []  # TODO: add get_actions(state) here
        return random.choice(actions)

    def choose_mixed(self, state):
        if random.random() < self.epsilon:
            return self.choose_random(state)
        return self.choose_greedy(state)

    def choose_action(self, state):
        actions = []  # TODO: add get_actions(state) here
        probabilities = self.boltzmann_scale(state, actions)
        return np.random.choice(actions, p=probabilities)
