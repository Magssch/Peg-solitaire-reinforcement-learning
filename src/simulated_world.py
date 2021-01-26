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

# move = (coordinates for)
# action = (starting_index_position, ending_index_position)
# move = ()


class SimulatedWorld:

    def __init__(self, board_type: Shape, size: int, holes: list[int]):
        self.__board_type = board_type
        self.__size = size
        self.__board = None
        self.__set_initial_state()
        if board_type == Shape.Diamond:
            self.__edges = (
                (0, -1),
                (1, -1),
                (1, 0),
                (0, 1),
                (-1, 1),
                (-1, 0),
            )
        else:
            self.__edges = (
                (0, -1),
                (-1, -1),
                (1, 0),
                (0, 1),
                (-1, 0),
                (1, 1),
            )

    def __set_initial_state(self) -> None:
        self.__board = np.ones((self.__size, self.__size))
        if self.__board_type == Shape.Triangle:
            self.__board = np.triu(np.full((self.__size, self.__size), None), 1)

    def reset(self) -> None:
        self.__set_initial_state()

    def step(self, action: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[int]:
        start_coordinates, direction_vector = action
        if self.__is_legal_move(start_coordinates, direction_vector):
            self.__board[start_coordinates] = 0
            removed_peg_coordinates = self.__get_next_node(
                start_coordinates, direction_vector)
            self.__board[removed_peg_coordinates] = 0
            landing_cell_coordinates = self.__get_next_node(
                removed_peg_coordinates, direction_vector)
            self.__board[landing_cell_coordinates] = 1
        return self.__board, 0, False

    def draw_board(self) -> None:
        pass

    def pegs_remaining(self) -> int:
        return (self.__board == 1).sum()

    def is_final_state(self) -> bool:
        return self.pegs_remaining() == 1

    def __is_legal_move(self, coordinates: Tuple[int, int], move: Tuple[int, int]) -> bool:
        return self.__move_is_inside_board(coordinates, move) and \
            self.__cell_contains_peg((coordinates[0] + move[0], coordinates[1] + move[1])) and \
            self.__move_is_inside_board(coordinates, (move[0] * 2, move[1] * 2)) and \
            not self.__cell_contains_peg(
                (coordinates[0] + (move[0] * 2), coordinates[1] + (move[1] * 2)))

    def __cell_contains_peg(self, coordinates: Tuple[int, int]) -> bool:
        return self.__board[coordinates] == 1

    def __move_is_inside_board(self, coordinates: Tuple[int, int], move: Tuple[int, int]) -> bool:
        adjacent_node = SimulatedWorld.get_coordinates_for_adjacent_cell(coordinates, move)
        return (adjacent_node[0] > 0 and adjacent_node[0] < self.__size and not self.__board(adjacent_node) is None) \
            and (adjacent_node[1] > 0 and adjacent_node[1] < self.__size and not self.__board(adjacent_node) is None)

    def grid_to_vector(self):
        return [cell for vector in self.__board for cell in vector]

    @staticmethod
    def get_coordinates_for_adjacent_cell(start_coordinates: Tuple[int, int], direction_vector: Tuple[int, int]) -> Tuple[int, int]:
        return start_coordinates[0] + direction_vector[0], start_coordinates[1] + direction_vector[1]

    def __get_legal_moves_for_position(self, coordinates: Tuple[int, int]) -> Tuple[int]:
        legal_moves = []
        for move in self.__edges:
            if self.__is_legal_move(coordinates, move):
                legal_moves.append(move)
        return tuple(legal_moves)

    def get_all_legal_actions(self) -> Tuple[Tuple[int, int]]:
        legal_moves = []
        for i in self.__board:
            for j in self.__board:
                legal_moves.append(((i, j), self.__get_legal_moves_for_position((i, j))))
        return tuple(legal_moves)
