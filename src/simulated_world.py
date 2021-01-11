# This is the specialized domain class for Peg Solitaire,
# it should do the following:
#
# - Understands game states and the operators that convert one game state to
# another.
# - Produces initial game states.
# - Generates child states from parent states using the legal operators of the
#  domain.
# - Recognizes final (winning, losing and neutral) states
from enum import Enum

import numpy as np


class Shape(Enum):
    Diamond = 1
    Triangle = 2


class SimulatedWorld:

    def __init__(self, shape, size):
        self.shape = shape
        self.size = size
        self.board = np.zeros((size, size))
        if shape == Shape.Triangle:
            self.board = np.triu(np.full((size, size), None), 1)

    def produce_initial_state(self):
        pass

    def generate_child_states(self):
        pass

    def is_final_state(self):
        return (self.board == 1).sum() == 1


if __name__ == "__main__":
    world = SimulatedWorld(Shape.Triangle, 4)
    world.board[1, 0] = 1
    print(world.board)
    print(world.is_final_state())
