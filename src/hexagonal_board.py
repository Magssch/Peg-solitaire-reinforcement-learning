from abc import ABC
from typing import Tuple

import numpy as np

from data_classes import Action, Shape
from visualize import Visualize


class HexagonalBoard(ABC):

    def __init__(self, board_type: Shape, size: int, holes: Tuple[int]):
        self.__board_type = board_type
        self.__size = size
        self.__holes = holes
        self.__board = None
        self._edges = []
        self.__set_initial_state()

    def get_board(self):
        return self.__board

    def __set_initial_state(self) -> None:
        self.__board = np.ones((self.__size, self.__size), dtype=np.int8)
        if self.__board_type == Shape.Triangle:
            self.__board = np.triu(
                np.full((self.__size, self.__size), 0, dtype=np.int8), 1)

    def reset_game(self) -> None:
        self.__set_initial_state()

    def draw_board(self, action: Action, delay: float) -> None:
        Visualize.draw_board(self.__board_type, self.__board,
                             action.positions, delay)

    def make_move(self, action: Action) -> None:
        if self.__is_legal_action(action):
            self.__board[action.start_coordinates] = 2
            self.__board[action.adjacent_coordinates] = 2
            self.__board[action.landing_coordinates] = 1

    def pegs_remaining(self) -> int:
        return (self.__board == 1).sum()

    def game_over(self) -> bool:
        return len(self.get_all_legal_actions()) < 1

    def __is_legal_action(self, action: Action) -> bool:
        return self.__action_is_inside_board(action) and \
            self.__cell_contains_peg(action.adjacent_coordinates) and \
            self.__action_is_inside_board(Action(action.start_coordinates, (action.direction_vector[0] * 2, action.direction_vector[1] * 2))) and \
            not self.__cell_contains_peg(
                action.landing_coordinates)

    def __cell_contains_peg(self, coordinates: Tuple[int, int]) -> bool:
        return self.__board[coordinates] == 1

    def __action_is_inside_board(self, action: Action) -> bool:
        adjacent_node = action.adjacent_coordinates
        return (adjacent_node[0] > 0 and adjacent_node[0] < self.__size and not self.__board[adjacent_node] is None) \
            and (adjacent_node[1] > 0 and adjacent_node[1] < self.__size and not self.__board[adjacent_node] is None)

    def __get_legal_actions_for_coordinates(self, coordinates: Tuple[int, int]) -> Tuple[Action]:
        legal_actions = []
        for direction_vector in self._edges:
            action = Action(coordinates, direction_vector)
            if self.__is_legal_action(action):
                legal_actions.append(action)
        return tuple(legal_actions)

    def get_all_legal_actions(self) -> Tuple[Action]:
        legal_moves = []
        for i in range(self.__board.size):
            for j in range(self.__board.size):
                legal_moves.append(
                    self.__get_legal_actions_for_coordinates((i, j)))
        return tuple(legal_moves)


class Diamond(HexagonalBoard):
    def __init__(self, board_type: Shape, size: int, holes: Tuple[int]):
        super().__init__(
            board_type,
            size,
            holes,
        )
        self._edges = (
            (0, -1),
            (1, -1),
            (1, 0),
            (0, 1),
            (-1, 1),
            (-1, 0),
        ),


class Triangle(HexagonalBoard):
    def __init__(self, board_type: Shape, size: int, holes: Tuple[int]):
        super().__init__(
            board_type,
            size,
            holes,
        )
        self._edges = (
            (0, -1),
            (-1, -1),
            (1, 0),
            (0, 1),
            (-1, 0),
            (1, 1),
        ),
