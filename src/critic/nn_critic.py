import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam

from .critic import Critic


class NNCritic(Critic):

    def __init__(
        self,
        learning_rate: float,
        discount_factor: float,
        trace_decay: float,
        nn_dimensions: tuple
    ):
        super().__init__(
            learning_rate,  # alpha
            discount_factor,  # gamma
            trace_decay,  # lambda
        )
        assert nn_dimensions is not None
        self.__nn_dimensions = nn_dimensions
        self.__values = self.__build_critic_network(nn_dimensions)  # V(s)
        self.reset_eligibilities()

    def __build_critic_network(self, nn_dimensions: tuple) -> Sequential:
        input_dim, *hidden_dims, output_dim = nn_dimensions

        assert output_dim == 1

        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        for dimension in hidden_dims:
            model.add(Dense(dimension, activation='relu'))

        model.add(Dense(units=1, activation='linear'))

        model.compile(
            optimizer=Adam(learning_rate=self._learning_rate),
            loss='mean_squared_error'
        )
        model.summary()
        return model

    def get_value(self, state) -> float:
        return np.squeeze(self.__values(np.array([state, ])))

    def update_value(self, current_state, successor_state, reward) -> None:
        successor_state = tf.convert_to_tensor([successor_state], dtype=tf.float32)
        current_state = tf.convert_to_tensor([current_state], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            target = tf.add(reward, tf.multiply(self._discount_factor, self.__values(successor_state)))
            prediction = tf.convert_to_tensor([self.__values(current_state)])
            loss = self.__values.compiled_loss(target, prediction)
            td_error = tf.subtract(target, prediction)

        gradients = tape.gradient(loss, self.__values.trainable_weights)
        gradients = self.__modify_gradients(gradients, td_error)
        self.__values.optimizer.apply_gradients(zip(gradients, self.__values.trainable_weights))

    def __modify_gradients(self, gradients, td_error):
        for gradient, eligibility in zip(gradients, self.__eligibilities):
            gradient = tf.multiply(td_error, tf.add(eligibility, gradient))
        return gradients

    def reset_eligibilities(self) -> None:
        self.__eligibilities = []
        for weights in self.__values.trainable_weights:
            self.__eligibilities.append(tf.ones(weights.shape))

    # Not used by NNCritic
    def replace_eligibilities(self, _) -> None:
        pass

    def update_eligibilities(self, state) -> None:
        self.__eligibilities = [self._discount_factor * self._trace_decay * e for e in self.__eligibilities]
