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
        discount_factor: float,
        trace_decay: float,
        nn_dimensions: tuple = None,
    ):
        self.__learning_rate = learning_rate  # alpha
        self.__discount_factor = discount_factor  # gamma
        self.__trace_decay = trace_decay  # lambda

        self.__nn_dimensions = nn_dimensions

        self.__policy = defaultdict(lambda: defaultdict(float))  # Pi(s, a)
        self.reset_eligibilities()

        if nn_dimensions is not None:
            self.model = self.__build_actor_network(nn_dimensions)

    def __build_actor_network(self, nn_dimensions: tuple) -> Sequential:
        input_dim, *hidden_dims, output_dim = nn_dimensions

        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        for dimension in hidden_dims:
            model.add(Dense(dimension, activation='relu'))

        model.add(Dense(output_dim, activation='softmax'))

        model.compile(
            optimizer=Adam(learning_rate=self.__learning_rate),
            loss='categorical_crossentropy'
        )
        model.summary()
        return model

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
        actions = []  # TODO: add get_actions(state) here
        probabilities = self.__boltzmann_scale(state, actions)
        # probabilities = np.squeeze(self.pi(state))
        return np.random.choice(actions, p=probabilities)

    # def update_policy(self, state, action, td_error):
    #     with tf.GradientTape(persistent=True) as tape:
    #         x0 = tf.Variable(0.2, name='x')
    #         print(self.model.trainable_variables)

    #     gradient = tape.gradient(x0, self.model.trainable_variables)
    #     print(gradient)
    #     self.model.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
