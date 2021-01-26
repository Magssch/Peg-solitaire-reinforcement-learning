# This is the specialized domain class for Peg Solitaire,
# it should do the following:
#
# - Understands game states and the operators that convert one game state to
# another.
# - Produces initial game states.
# - Generates child states from parent states using the legal operators of the
#  domain.
# - Recognizes final (winning, losing and neutral) states

from hexagonal_board import Shape, Diamond, Triangle


class SimulatedWorld:

    def __init__(self, board_type: Shape, size: int, holes: list[int]):
        if board_type == Shape.Diamond:
            self.__game_board = Diamond(board_type, size, holes)
        else:
            self.__game_board = Triangle(board_type, size, holes)
