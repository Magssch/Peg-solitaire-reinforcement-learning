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


class Shape(Enum):
    Diamond = 1
    Triangle = 2

# move = (coordinates for)
# action = (starting_index_position, ending_index_position)
# move = ()


class SimulatedWorld:

    def __init__(self, board_type: Shape, size: int):
        self.__board_type = board_type
        self.__size = size
        self.__board = None
        self.__set_initial_state()
        self.__edges = (
            (0, -1),
            (1, -1),
            (1, 0),
            (0, 1),
            (-1, 1),
            (-1, 0),
        )

    def __set_initial_state(self) -> None:
        self.__board = np.zeros(self.__size**2)
        if self.__board_type == Shape.Triangle:
            self.__board = np.triu(np.full((self.__size, self.__size), None), 1)

    def reset(self) -> None:
        self.__set_initial_state()

    def step(self, action: tuple(int, int)) -> tuple(int):
        start_index, direction_index = action
        start_coordinates = self.__coordinates_from_index(start_index)
        direction_vector = self.__board(direction_index)
        if self.__is_legal_move(start_coordinates, direction_vector):
            self.__board[start_index] = 0
            removed_peg_coordinates = self.__get_next_node(
                start_coordinates, direction_vector)
            self.__board[self.__index_from_coordinates(
                removed_peg_coordinates)] = 0
            landing_cell_coordinates = self.__get_next_node(
                removed_peg_coordinates, direction_vector)
            self.__board[self.__index_from_coordinates(
                landing_cell_coordinates)] = 1

    def draw_board(self) -> None:
        pass

    def is_final_state(self) -> bool:
        return (self.__board == 1).sum() == 1

    def __is_legal_move(self, coordinates: tuple(int, int), move: tuple(int, int)) -> bool:
        return self.__move_is_inside_board(coordinates, move) and \
            self.__cell_contains_peg((coordinates[0] + move[0], coordinates[1] + move[1])) and \
            self.__move_is_inside_board(coordinates, (move[0] * 2, move[1] * 2)) and \
            not self.__cell_contains_peg(
                (coordinates[0] + (move[0] * 2), coordinates[1] + (move[1] * 2)))

    def __cell_contains_peg(self, cell_position: tuple(int, int)) -> bool:
        return self.__board[self.__index_from_coordinates(cell_position)] == 1

    def __move_is_inside_board(self, cell_position: tuple(int, int), move: tuple(int, int)) -> bool:
        return (cell_position[0] + move[0] > 0 and cell_position[0] + move[0] < self.__size) \
            and (cell_position[1] + move[1] > 0 and cell_position[1] + move[1] < self.__size)

    def __coordinates_from_index(self, position: int) -> tuple(int, int):
        """
        Converts numerical 1D-index to a tuple of 2D-coordinates
        """
        return position // self.__size, position % self.__size

    def __index_from_coordinates(self, cell_position: tuple(int, int)) -> int:
        """
        Converts 2D-coordinates to an 1D-index position
        """
        row = cell_position[0] * self.__size
        return row, row + cell_position[1]

    def __get_next_node(self, start_coordinates: tuple(int, int), direction_vector: tuple(int, int)) -> int:
        return start_coordinates[0] + direction_vector[0], start_coordinates[1] + direction_vector[1]

    def __direction_index_from_vector(self, direction_vector: tuple(int, int)) -> int:
        """
        Converts 2D-direction vector to an index
        """
        return self.__edges.index(direction_vector)

    def __get_legal_moves_for_position(self, position: int) -> tuple(int):
        legal_moves = []
        coordinates = self.__coordinates_from_index(position)

        for move in self.__edges:
            if self.__is_legal_move(coordinates, move):
                legal_moves.append(self.__index_from_coordinates(move))

        return tuple(legal_moves)

    def get_all_legal_actions(self) -> tuple(tuple(int, int)):
        legal_moves = []
        i = 0
        for position in self.__board:
            legal_moves[i] = (
                position, self.__get_legal_moves_for_position(position))
        return tuple(legal_moves)
