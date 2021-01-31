from abc import ABC
import enum
from typing import Tuple
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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

    def __init__(self, board_type: Shape, size: int, holes: Tuple[int]):
        self.__board_type = board_type
        self.__size = size
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
            removed_peg_coordinates = HexagonalBoard.get_coordinates_for_adjacent_cell(action)
            self.__board[removed_peg_coordinates] = 0
            landing_cell_coordinates = HexagonalBoard.get_coordinates_for_adjacent_cell(Action(removed_peg_coordinates, action.direction_vector))
            self.__board[landing_cell_coordinates] = 1

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
        adjacent_node = HexagonalBoard.get_coordinates_for_adjacent_cell(action)
        return (adjacent_node[0] > 0 and adjacent_node[0] < self.__size and not self.__board(adjacent_node) is None) \
            and (adjacent_node[1] > 0 and adjacent_node[1] < self.__size and not self.__board(adjacent_node) is None)

    @staticmethod
    def get_coordinates_for_adjacent_cell(action: Action) -> Tuple[int, int]:
        return action.start_coordinates[0] + action.direction_vector[0], action.start_coordinates[1] + action.direction_vector[1]

    def __get_legal_actions_for_coordinates(self, coordinates: Tuple[int, int]) -> Action:
        legal_actions = []
        for direction_vector in self.__edges:
            if self.__is_legal_action(coordinates, direction_vector):
                legal_actions.append(Action(coordinates, direction_vector))
        return tuple(legal_actions)

    def get_all_legal_actions(self) -> Tuple(Action):
        legal_moves = []
        for i in self.__board:
            for j in self.__board:
                legal_moves.append(self.__get_legal_actions_for_coordinates((i, j)))
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


class DrawBoard(HexagonalBoard):
    def __init__(self, board_type: Shape, size: int, holes: list(int)):
        self.graph = nx.Graph()
        super().__init__(
            self,
            board_type,
            size,
            holes,
            board,
        ),
    )

    def add_node(self, position):
        self.graph.add_node(position)

    def add_edge(self, x, y):
        self.graph.add_edge(x, y)

    def get_filled_nodes(self, board):
        filled_positions = []
        for i in board:
            for j in board:
                if board[i][j] == 1:
                    filled_positions.append((i,j))
        return filled_positions
    
    def get_empty_nodes(self, board):
        empty_positions = []
        for i in board:
            for j in board:
                if board[i][j] == 0:
                    empty_positions.append((i,j))
        return empty_positions

    def get_legal_positions(self, board):
        legal_positions = []
        for i in board:
            for j in board:
                if board[i][j] != None:
                    legal_positions.append((i,j))
        return legal_positions


    def draw_board(self, positions = None, board, board_type, active_nodes = []):

        # List of all node positions currently filled
        filled_nodes = self.get_filled_nodes()
        legal_positions = self.get_legal_positions

        # Remove nodes currently active
        for nodes in active_nodes:
            filled_nodes.remove(nodes)

        # Creates a Hex Diamon grid
        if board_type == Diamond:
            shape = Diamond()

            for i in range(size):
                for j in range(size):
                    self.add_node((i,j))

            for position in legal_positions:
                for neighbor_position in shape.edges:   # USIKKER på hvordan å targete edges i Diamond
                    neighbor_node = (position[0] + neighbor_position[0], position[1] + neighbor_position[1])
                    if neighbor_node in legal_positions:
                        self.add_edge(position, neighbor_node)
        
        # Create a Hex Triangle grid
        elif board_type == Triangle:
            shape = Triangle()
            for i in range(size):
                for j in range(i + 1):
                    self.add_node((i,j))
            
            for position in legal_positions:
                for neighbor_position in shape.edges:
                    neighbor_node = (position[0] + neighbor_position[0], position[1] + neighbor_position[1])
                    if neighbor_node in legal_positions:
                        self.add_edge(position, neighbor_node)

        # Draw the resulting grid
        nx.draw_networkx_nodes(self.graph, positions, nodelist=self.get_empty_nodes(), node_color='black')
        nx.draw_networkx_nodes(self.graph, positions, nodelist=filled_nodes, node_color='blue')
        nx.draw_networkx_nodes(self.graph, positions, nodelist=active_nodes, node_color='green')
        nx.draw_networkx_edges(self.graph, positions, width=1)

        plt.axis('off')
        plt.draw()
        plt.clf()
        plt.show()


        





