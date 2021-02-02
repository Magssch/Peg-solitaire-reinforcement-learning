from abc import ABC
from enum import Enum
from typing import Tuple
import numpy as np


class Action:

    def __init__(self, start_coordinates: Tuple[int, int], direction_vector: Tuple[int, int]):
        self.start_coordinates = start_coordinates
        self.direction_vector = direction_vector

    def get_coordinates_for_adjacent_cell(self) -> Tuple[int, int]:
        return self.start_coordinates[0] + self.direction_vector[0], self.start_coordinates[1] + self.direction_vector[1]

    def get_coordinates_for_landing_cell(self) -> Tuple[int, int]:
        return self.start_coordinates[0] + (self.direction_vector[0] * 2), self.start_coordinates[1] + (self.direction_vector[1] * 2)

    def __hash__(self):
        return hash(self.start_coordinates) ^ hash(self.direction_vector)


class Shape(Enum):
    Diamond = 1
    Triangle = 2


class HexagonalBoard(ABC):

    def __init__(self, board_type: Shape, size: int, holes: Tuple[int]):
        self.__board_type = board_type
        self.__size = size
        self.__holes = holes
        self.__board = None
        self.__set_initial_state()

    def __set_initial_state(self) -> None:
        self.__board = np.ones((self.__size, self.__size))
        if self.__board_type == Shape.Triangle:
            self.__board = np.triu(np.full((self.__size, self.__size), None), 1)

    def reset_game(self) -> None:
        self.__set_initial_state()

    def make_move(self, action: Action) -> None:
        if self.__is_legal_action(action):
            self.__board[action.start_coordinates] = 0
            self.__board[action.get_coordinates_for_adjacent_cell()] = 0
            self.__board[action.get_coordinates_for_landing_cell()] = 1

    def draw_board(self) -> None:
        pass

    def pegs_remaining(self) -> int:
        return (self.__board == 1).sum()

    def game_over(self) -> bool:
        return len(self.__get_all_legal_actions()) < 1

    def __is_legal_action(self, action: Action) -> bool:
        return self.__action_is_inside_board(action) and \
            self.__cell_contains_peg((action.start_coordinates[0] + action.direction_vector[0], action.start_coordinates[1] + action.direction_vector[1])) and \
            self.__action_is_inside_board(Action(action.start_coordinates, (action.direction_vector[0] * 2, action.direction_vector[1] * 2))) and \
            not self.__cell_contains_peg(
                (action.start_coordinates[0] + (action.direction_vector[0] * 2), action.start_coordinates[1] + (action.direction_vector[1] * 2)))

    def __cell_contains_peg(self, coordinates: Tuple[int, int]) -> bool:
        return self.__board[coordinates] == 1

    def __action_is_inside_board(self, action: Action) -> bool:
        adjacent_node = action.get_coordinates_for_adjacent_cell()
        return (adjacent_node[0] > 0 and adjacent_node[0] < self.__size and not self.__board(adjacent_node) is None) \
            and (adjacent_node[1] > 0 and adjacent_node[1] < self.__size and not self.__board(adjacent_node) is None)

    def __get_legal_actions_for_coordinates(self, coordinates: Tuple[int, int]) -> Action:
        legal_actions = []
        for direction_vector in self.__edges:
            if self.__is_legal_action(coordinates, direction_vector):
                legal_actions.append(Action(coordinates, direction_vector))
        return tuple(legal_actions)

    def get_all_legal_actions(self) -> Tuple[Action]:
        legal_moves = []
        for i in self.__board:
            for j in self.__board:
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
        self.edges = (
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
        self.edges = (
            (0, -1),
            (-1, -1),
            (1, 0),
            (0, 1),
            (-1, 0),
            (1, 1),
        ),
