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

    def __move_is_inside_board(self, cell_position: int, move: int) -> bool:
        return cell_position + move > 0 and cell_position + move < self.__size

    def __get_coordinates_for_position(self, position: int) -> tuple(int, int):
        """
        Converts numerical 1D-index to a tuple of 2D-coordinates
        """
        return position // self.__size, position % self.__size

    def __get_legal_actions_for_position(self, position: int) -> tuple(int, int):
        legal_moves = []
        row, col = self.__get_coordinates_for_position(position)

        for adjacent_row, adjacent_col in self.__adjacent_cells:
            if self.__move_is_inside_board(row, adjacent_row) and self.__move_is_inside_board(col, adjacent_col):
                # TODO: CHECK IF THERE IS A PEG TO JUMP OVER
                # Generate cell position tuples for cells beyond pegs
                landing_cells = [tuple((a*2, b*2) for a, b in adjacent_cell) for adjacent_cell in self.__adjacent_cells]
                for landing_row, landing_col in landing_cells:
                    if self.__move_is_inside_board(row, adjacent_row) and self.__move_is_inside_board(col, adjacent_col):
                        legal_moves.append((landing_row, landing_col))

        return tuple(legal_moves)

                    

    def get_all_legal_actions(self) -> tuple(tuple):
        for cell in self.__board:
            pass
