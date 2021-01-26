from abc import ABC
import enum
from typing import Tuple
import numpy as np


class Action:

    def __init__(self, start_coordinates: Tuple[int, int], direction_vector: Tuple[int, int]):
        self.start_coordinates = start_coordinates
        self.direction_vector = direction_vector

    def __hash__(self):
        return hash(self.start_coordinates) ^ hash(self.direction_vector)


class Shape(enum):
    Diamond = 1
    Triangle = 2


class HexagonalBoard(ABC):

    def __init__(self, board_type: Shape, size: int, edges: Tuple[Tuple[int, int]], holes: list[int]):
        self.__board_type = board_type
        self.__size = size
        self.__board = None
        self.__set_initial_state()

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
        return self.__grid_to_vector(), self.__calculate_reward(), self.__is_final_state(), self.__get_all_legal_actions()

    def __draw_board(self) -> None:
        pass

    def __pegs_remaining(self) -> int:
        return (self.__board == 1).sum()

    def __game_over(self) -> bool:
        return len(self.__get_all_legal_actions()) < 1

    def __is_final_state(self) -> bool:
        return self.pegs_remaining() == 1 or self.__game_over()

    def __calculate_reward(self) -> int:
        if self.__game_over():
            return -1
        elif self.pegs_remaining() == 1:
            return 1
        else:
            return 0

    def __is_legal_move(self, coordinates: Tuple[int, int], move: Tuple[int, int]) -> bool:
        return self.__move_is_inside_board(coordinates, move) and \
            self.__cell_contains_peg((coordinates[0] + move[0], coordinates[1] + move[1])) and \
            self.__move_is_inside_board(coordinates, (move[0] * 2, move[1] * 2)) and \
            not self.__cell_contains_peg(
                (coordinates[0] + (move[0] * 2), coordinates[1] + (move[1] * 2)))

    def __cell_contains_peg(self, coordinates: Tuple[int, int]) -> bool:
        return self.__board[coordinates] == 1

    def __move_is_inside_board(self, coordinates: Tuple[int, int], move: Tuple[int, int]) -> bool:
        adjacent_node = HexagonalBoard.get_coordinates_for_adjacent_cell(coordinates, move)
        return (adjacent_node[0] > 0 and adjacent_node[0] < self.__size and not self.__board(adjacent_node) is None) \
            and (adjacent_node[1] > 0 and adjacent_node[1] < self.__size and not self.__board(adjacent_node) is None)

    def __grid_to_vector(self):
        return tuple([cell for vector in self.__board for cell in vector])

    @staticmethod
    def get_coordinates_for_adjacent_cell(start_coordinates: Tuple[int, int], direction_vector: Tuple[int, int]) -> Tuple[int, int]:
        return start_coordinates[0] + direction_vector[0], start_coordinates[1] + direction_vector[1]

    def __get_legal_moves_for_position(self, coordinates: Tuple[int, int]) -> Tuple[int, int]:
        legal_moves = []
        for move in self.__edges:
            if self.__is_legal_move(coordinates, move):
                legal_moves.append(move)
        return tuple(legal_moves)

    def __get_all_legal_actions(self) -> Tuple(Action):
        legal_moves = []
        for i in self.__board:
            for j in self.__board:
                legal_moves.append(Action((i, j), self.__get_legal_moves_for_position((i, j))))
        return tuple(legal_moves)


class Diamond(HexagonalBoard):
    def __init__(self, board_type: Shape, size: int, holes: list[int]):
        super().__init__(
            self,
            board_type,
            size,
            holes,
            edges=(
                (0, -1),
                (1, -1),
                (1, 0),
                (0, 1),
                (-1, 1),
                (-1, 0),
            ),
        )


class Triangle(HexagonalBoard):
    def __init__(self, board_type: Shape, size: int, holes: list[int]):
        super().__init__(
            self,
            board_type,
            size,
            holes,
            edges=(
                (0, -1),
                (-1, -1),
                (1, 0),
                (0, 1),
                (-1, 0),
                (1, 1),
            ),
        )
