from abc import ABC
from typing import Set, Tuple

import numpy as np

from data_classes import Action, Shape
from visualize import Visualize


class HexagonalBoard(ABC):

    def __init__(self, board_type: Shape, size: int, holes: Tuple[Tuple[int, int]]):
        self.__board_type = board_type
        self.__size: int = size
        self.__holes = holes
        self.__board = None
        self._edges: Set[Tuple[int, int]] = set()
        self.__set_initial_state()

    def get_board(self):
        return self.__board

    def __set_initial_state(self) -> None:
        self.__board = np.ones((self.__size, self.__size), dtype=np.int8)
        if self.__board_type == Shape.Triangle:
            self.__board = np.triu(
                np.full((self.__size, self.__size), 0, dtype=np.int8), 1)

        for hole in self.__holes:
            if self.__board[hole]:
                self.__board[hole] = 2

    def reset_game(self) -> None:
        self.__set_initial_state()

    def __draw_board(self, action: Action) -> None:
        Visualize.draw_board(self.__board_type, self.__board,
                             action.positions)

    def make_move(self, action: Action) -> None:
        if self.__is_legal_action(action):
            self.__board[action.start_coordinates] = 2
            self.__board[action.adjacent_coordinates] = 2
            self.__board[action.landing_coordinates] = 1
            self.__draw_board(action)

    def pegs_remaining(self) -> int:
        return (self.__board == 1).sum()

    def game_over(self) -> bool:
        return len(self.get_all_legal_actions()) < 1

    def __is_legal_action(self, action: Action) -> bool:
        return self.__action_is_inside_board(action) \
            and self.__cell_contains_peg(action.adjacent_coordinates) \
            and not self.__cell_contains_peg(
            action.landing_coordinates)

    def __cell_contains_peg(self, coordinates: Tuple[int, int]) -> bool:
        return self.__board[coordinates] == 1

    def __action_is_inside_board(self, action: Action) -> bool:
        return (action.adjacent_coordinates[0] >= 0 and action.adjacent_coordinates[0] < self.__size) \
            and (action.adjacent_coordinates[1] >= 0 and action.adjacent_coordinates[1] < self.__size) \
            and (action.landing_coordinates[0] >= 0 and action.landing_coordinates[0] < self.__size) \
            and (action.landing_coordinates[1] >= 0 and action.landing_coordinates[1] < self.__size) \
            and self.__board[action.adjacent_coordinates] != 0 and self.__board[action.landing_coordinates] != 0

    def __get_legal_actions_for_coordinates(self, coordinates: Tuple[int, int]) -> Tuple[Action]:
        legal_actions = []
        for direction_vector in self._edges:
            action = Action(coordinates, direction_vector)
            if self.__is_legal_action(action):
                legal_actions.append(action)
        return tuple(legal_actions)

    def get_all_legal_actions(self) -> Tuple[Action]:
        legal_moves = []
        for i in range(self.__board.shape[0]):
            for j in range(self.__board.shape[0]):
                legal_actions_for_position = self.__get_legal_actions_for_coordinates(
                    (i, j))
                if len(legal_actions_for_position) > 0:
                    legal_moves.append(legal_actions_for_position)
        return tuple(legal_moves)


class Diamond(HexagonalBoard):
    def __init__(self, board_type: Shape, size: int, holes: Tuple[Tuple[int, int]]):
        super().__init__(
            board_type,
            size,
            holes,
        )
        self._edges = set([
            (0, -1),
            (1, -1),
            (1, 0),
            (0, 1),
            (-1, 1),
            (-1, 0),
        ])


class Triangle(HexagonalBoard):
    def __init__(self, board_type: Shape, size: int, holes: Tuple[Tuple[int, int]]):
        super().__init__(
            board_type,
            size,
            holes,
        )
        self._edges = set([
            (0, -1),
            (-1, -1),
            (1, 0),
            (0, 1),
            (-1, 0),
            (1, 1),
        ])
