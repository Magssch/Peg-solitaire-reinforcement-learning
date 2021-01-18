import random
from collections import defaultdict

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam


class Actor:

    def __init__(
        self,
        learning_rate: float,
        trace_decay: float,
        discount_factor: float,
        epsilon: float,
        nn_dimensions: tuple = None,
    ):
        self.learning_rate = learning_rate  # alpha
        self.trace_decay = trace_decay  # lambda
        self.discount_factor = discount_factor  # gamma

        self.epsilon = epsilon
        self.policy = {}  # Pi
        self.reset_eligibilities()
        self.nn_dimensions = nn_dimensions

        if nn_dimensions is not None:
            self._build_actor_network(nn_dimensions)

    def _build_actor_network(self, nn_dimensions: tuple) -> Sequential:
        input_dim, *hidden_dims, output_dim = nn_dimensions

        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        for dimension in hidden_dims:
            model.add(Dense(dimension, activation='relu'))

        model.add(Dense(output_dim, activation='softmax'))

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy'
        )
        model.summary()
        return model

    def update_policy(self, state, action, td_error) -> None:
        self.policy[state][action] += self.learning_rate * td_error * self.eligibilities[state][action]

    def reset_eligibilities(self) -> None:
        if self.nn_dimensions is not None:
            input_dim, *hidden_dims, _ = self.nn_dimensions
            self.eligibilities = [np.zeros(input_dim)]
            for dimension in hidden_dims:
                self.eligibilities.append(np.zeros(dimension))
        self.eligibilities = defaultdict(lambda: defaultdict(float))

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
        # probabilities = np.squeeze(self.pi(state))
        return np.random.choice(actions, p=probabilities)

    def update_pi(self, state, action, td_error):
        with tf.GradientTape() as tape:
            probabilities = np.squeeze(self.pi(state))
            log_probability = tf.math.log(probabilities[action])
            # td_error = val - pred
            # 1/2*delta *
            loss = -tf.squeeze(log_probability) * td_error
        gradient = tape.gradient(loss, self.pi.trainable_variables)
        self.pi.optimizer.apply_gradients(zip(gradient, self.pi.trainable_variables))


if __name__ == "__main__":
    actor = Actor(0.01, 0.9, 0.01, 0.5)
