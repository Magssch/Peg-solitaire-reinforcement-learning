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
from typing import Tuple

import networkx as nx
import numpy as np


class Shape(Enum):
    Diamond = 1
    Triangle = 2


class SimulatedWorld:

    def __init__(self, board_type: Shape, size: int):
        self.__board_type = board_type
        self.__size = size
        self.__board = tuple([1 for _ in range(size**2)])
        self.__adjacent_cells = (
            (0, -1),
            (1, -1),
            (1, 0),
            (0, 1),
            (-1, 1),
            (0, -1),
        )
        if self.__board_type == Shape.Triangle:
            self.__board = np.triu(np.full((size, size), None), 1)

    def reset(self) -> tuple:
        return self.__board
        # reset to

    def step(self, state) -> tuple:
        return self.__board, 0, False
        # Go to next state

    def draw_board(self) -> None:
        pass

    def is_final_state(self) -> bool:
        return (self.__board == 1).sum() == 1

    def __is_legal_move(self, coordinates: Tuple[int, int], move: Tuple[int, int]) -> bool:
        return self.__move_is_inside_board(coordinates, move) and \
                self.__cell_contains_peg((coordinates[0] + move[0], coordinates[1] + move[1])) and \
                self.__move_is_inside_board(coordinates, (move[0] * 2, move[1] * 2)) and \
                not self.__cell_contains_peg((coordinates[0] + (move[0] * 2), coordinates[1] + (move[1] * 2)))

    def __cell_contains_peg(self, cell_position: Tuple[int, int]) -> bool:
        return self.__board[self.__get_index_by_coordinates(cell_position)] == 1

    def __move_is_inside_board(self, cell_position: Tuple[int, int], move: Tuple[int, int]) -> bool:
        return (cell_position[0] + move[0] > 0 and cell_position[0] + move[0] < self.__size) \
                and (cell_position[1] + move[1] > 0 and cell_position[1] + move[1] < self.__size)

    def __get_coordinates_for_position(self, position: int) -> Tuple[int, int]:
        """
        Converts numerical 1D-index to a tuple of 2D-coordinates
        """
        return position // self.__size, position % self.__size

    def __get_index_by_coordinates(self, cell_position: Tuple[int, int]) -> int:
        """
        Converts 2D-coordinates to an 1D-index position
        """
        row = cell_position[0] * self.__size
        return row + cell_position[1]

    def __get_legal_moves_for_position(self, position: int) -> Tuple[int, int]:
        legal_moves = []
        coordinates = self.__get_coordinates_for_position(position)

        for move in self.__adjacent_cells:
            if self.__is_legal_move(coordinates, move):
                legal_moves.append(self.__get_index_by_coordinates((coordinates[0] + (move[0]*2), coordinates[1]+(move[1]*2))))

        return tuple(legal_moves)

    def get_all_legal_actions(self) -> Tuple[Tuple[int, int]]:
        legal_moves = []
        i = 0
        for position in self.__board:
            legal_moves[i] = (position, self.__get_legal_moves_for_position(position))
        return tuple(legal_moves)
