# This is the specialized domain class for Peg Solitaire,
# it should do the following:
#
# - Understands game states and the operators that convert one game state to
# another.
# - Produces initial game states.
# - Generates child states from parent states using the legal operators of the
#  domain.
# - Recognizes final (winning, losing and neutral) states

from typing import Tuple

from hexagonal_board import Action, Diamond, Shape, Triangle


class SimulatedWorld:

    def __init__(self, board_type: Shape, size: int, holes: Tuple[int]):
        if board_type == Shape.Diamond:
            self.__game_board = Diamond(board_type, size, holes)
        else:
            self.__game_board = Triangle(board_type, size, holes)

    def step(self, action: Action) -> Tuple[int]:
        self.__game_board.make_move(action)
        return self.__grid_to_vector(), self.__calculate_reward(), self.__is_final_state(), self.__game_board.get_all_legal_actions()

    def reset(self) -> None:
        self.__game_board.reset_game()

    def __is_final_state(self) -> bool:
        return self.__game_board.pegs_remaining() == 1 or self.__game_board.game_over()

    def __calculate_reward(self) -> int:
        if self.__game_board.game_over():
            return -1
        elif self.__game_board.pegs_remaining() == 1:
            return 1
        else:
            return 0

    def __grid_to_vector(self):
        return tuple([cell for vector in self.__board for cell in vector])
