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

import networkx as nx
import numpy as np

from configparser import ConfigParser
config = ConfigParser()

class Shape(Enum):
    Diamond = 1
    Triangle = 2


class SimulatedWorld:

    def __init__(self, shape: Shape, size: int):
        self.__shape = shape
        self.__size = size
        self.__board = np.zeros((size, size))
        self.__adjacent_cells = (
            (0, -1),
            (1, -1),
            (1, 0),
            (0, 1),
            (-1, 1),
            (0, -1),
        )
        if shape == Shape.Triangle:
            self.__board = np.triu(np.full((size, size), None), 1)

    def produce_initial_state(self) -> None:
        pass

    def generate_child_states(self) -> None:
        pass

    def is_final_state(self) -> bool:
        pass
        #return (self.__board == 1).sum() == 1

    def draw_board(self) -> None:
        pass

    def __is_legal_move(self) -> bool:
        pass

    def __is_adjacent(self, x: int, y: int) -> bool:
        return (x, y) in self.__adjacent_cells

