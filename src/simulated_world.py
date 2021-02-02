from typing import Tuple

from data_classes import Action, Shape
from hexagonal_board import Diamond, Triangle
from parameters import Parameters
from visualize import Visualize


class SimulatedWorld:

    def __init__(self):
        self.__board_type = Parameters.board_type
        if Parameters.board_type == Shape.Diamond:
            self.__game_board = Diamond(
                Parameters.board_type, Parameters.size, Parameters.holes)
        else:
            self.__game_board = Triangle(
                Parameters.board_type, Parameters.size, Parameters.holes)
        self.__frame_delay = Parameters.frame_delay
        self.__peg_history = []

    def __is_final_state(self) -> bool:
        self.__peg_history.append(self.__game_board.pegs_remaining())
        return self.__game_board.pegs_remaining() == 1 or self.__game_board.game_over()

    def __calculate_reward(self) -> int:
        if self.__game_board.game_over():
            return -1
        elif self.__game_board.pegs_remaining() == 1:
            return 1
        else:
            return 0

    def __grid_to_vector(self):
        return tuple(self.__game_board.get_board().flatten())

    def step(self, action: Action) -> Tuple[Tuple[int], int, bool, Tuple[Action]]:
        self.__game_board.make_move(action)
        return self.__grid_to_vector(), self.__calculate_reward(), self.__is_final_state(), self.__game_board.get_all_legal_actions()

    def reset(self) -> Tuple[Tuple[int], Tuple[Action]]:
        self.__game_board.reset_game()
        return self.__grid_to_vector(), self.__game_board.get_all_legal_actions()

    def exit(self) -> None:
        Visualize.plot_training_data(self.__peg_history)
