import random

import numpy as np
import tensorflow as tf
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
        self.__build_actor_network()

    def __build_actor_network(
        self,
        input_dim: int = 25,
        hidden_dim: int = 512,
        n_actions: int = 6,
    ) -> None:
        input = Input(shape=(input_dim,))
        dense = Dense(hidden_dim, activation='relu')(input)
        probabilities = Dense(n_actions, activation='softmax')(dense)

        model = Model(inputs=[input], outputs=[probabilities])
        model.compile(optimizer=Adam(
            learning_rate=self.learning_rate),
            loss='huber_loss'
        )
        self.pi = model

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
        # probabilities = np.squeeze(self.pi(state))
        return np.random.choice(actions, p=probabilities)

    def update_pi(self, state, action, td_error):
        with tf.GradientTape() as tape:
            probabilities = np.squeeze(self.pi(state))
            log_probability = tf.math.log(probabilities[action])
            loss = -tf.squeeze(log_probability) * td_error
        gradient = tape.gradient(loss, self.pi.trainable_variables)
        self.pi.optimizer.apply_gradients(zip(gradient, self.pi.trainable_variables))


if __name__ == "__main__":
    actor = Actor(0.01, 0.9, 0.01, 0.5)
