from typing import Tuple, Union

import parameters
from data_classes import Action, Shape
from hexagonal_board import Diamond, Triangle
from visualize import Visualize


class SimulatedWorld:

    def __init__(self):
        self.__board_type = parameters.BOARD_TYPE
        if parameters.BOARD_TYPE == Shape.Diamond:
            self.__game_board = Diamond(parameters.BOARD_TYPE, parameters.SIZE, parameters.HOLES)
            Visualize.initialize_board(self.__game_board.get_board(), self.__game_board._edges, self.__board_type)
        else:
            self.__game_board = Triangle(parameters.BOARD_TYPE, parameters.SIZE, parameters.HOLES)
            Visualize.initialize_board(self.__game_board.get_board(), self.__game_board._edges, self.__board_type)
        self.__peg_history = []
        self.__memoized_legal_actions = {}
        print('Initial board:')
        print(self.__game_board)

    def __is_final_state(self) -> bool:
        return self.__game_board.game_over()

    def __calculate_reward(self) -> int:
        if self.__game_board.pegs_remaining() == 1:
            return 1
        elif self.__game_board.game_over():
            return -1
        else:
            return 0

    def __grid_to_vector(self) -> Tuple[bool]:
        return tuple(filter(lambda cell: bool(cell), self.__game_board.get_board().flatten()))

    def step(self, action: Union[Action, None], visualize: bool) -> Tuple[Tuple[int], int, bool, Tuple[Action]]:
        assert action is not None, 'No actions found. Cannot play game.'
        self.__game_board.make_move(action, visualize)
        grid_to_vector = self.__grid_to_vector()
        return grid_to_vector, self.__calculate_reward(), self.__is_final_state(), self.__memoize_legal_actions(grid_to_vector)

    def reset(self) -> Tuple[Tuple[int], Tuple[Action]]:
        self.__peg_history.append(self.__game_board.pegs_remaining())  # Used for plotting
        self.__game_board.reset_game()
        return self.__grid_to_vector(), self.__game_board.get_all_legal_actions()

    def plot_training_data(self) -> None:
        self.__peg_history.append(self.__game_board.pegs_remaining())
        Visualize.plot_training_data(self.__peg_history[1:])

    def __memoize_legal_actions(self, grid_to_vector: Tuple[bool]) -> Tuple[Action]:
        if grid_to_vector not in self.__memoized_legal_actions:
            all_legal_actions = self.__game_board.get_all_legal_actions()
            self.__memoized_legal_actions[grid_to_vector] = all_legal_actions
            return all_legal_actions
        else:
            return self.__memoized_legal_actions[grid_to_vector]
