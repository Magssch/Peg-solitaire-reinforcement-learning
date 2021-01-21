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


class SimulatedWorld:

    def __init__(self, board_type: Shape, size: int):
        self.__board_type = board_type
        self.__size = size
        self.__board = np.zeros(size**2)
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

    def reset(self) -> None:
        pass
        # reset to 

    def step(self) -> tuple:
        pass
        # Go to next state

    def draw_board(self) -> None:
        # return (self.__board, #poeng, self.is_final_state())
        pass
    
    def is_final_state(self) -> bool:
        pass
        # return (self.__board == 1).sum() == 1

    def __is_legal_move(self) -> bool:
        pass

    def __is_adjacent(self, x: int, y: int) -> bool:
        return (x, y) in self.__adjacent_cells

    def __position_contains_peg(self, position: int) -> bool:
        return self.__board[position] == 1

    def __move_is_inside_board(self, cell_position: tuple(int, int), move: tuple(int, int)) -> bool:
        return (cell_position[0] + move[0] > 0 and cell_position[0] + move[0] < self.__size) and (cell_position[1] + move[1] > 0 and cell_position[1] + move[1] < self.__size)

    def __get_coordinates_for_position(self, position: int) -> tuple(int, int):
        """
        Converts numerical 1D-index to a tuple of 2D-coordinates
        """
        return position // self.__size, position % self.__size

    def __get_legal_actions_for_position(self, position: int) -> tuple(int, int):
        legal_moves = []
        coordinates = self.__get_coordinates_for_position(position)

        for jumping_move in self.__adjacent_cells:
            if self.__move_is_inside_board(coordinates, jumping_move) and self.__position_contains_peg(position):
                # TODO: CHECK IF THERE IS A PEG TO JUMP OVER
                # Generate cell position tuples for cells beyond pegs
                landing_moves = [tuple((a*2, b*2) for a, b in adjacent_cell) for adjacent_cell in self.__adjacent_cells]
                for landing_move in landing_moves:
                    if self.__move_is_inside_board(coordinates, landing_move) and not self.__position_contains_peg(position):
                        legal_moves.append(landing_move)

        return tuple(legal_moves)
                    

    def get_all_legal_actions(self) -> tuple(tuple):
        for cell in self.__board:
            pass
