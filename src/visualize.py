from networkx.classes.graph import Graph
from data_classes import Shape
import networkx as nx
import matplotlib.pyplot as plt
import time


class Visualize():

    graph = nx.Graph()

    @staticmethod
    def add_node_to_graph(graph, position):
        graph.add_node(position)

    @staticmethod
    def add_edge_to_graph(graph, u, v):
        graph.add_edge(u, v)

    @staticmethod
    def get_filled_nodes(board):
        filled_positions = []
        for i in board:
            for j in board:
                if board[i][j] == 1:
                    filled_positions.append((i, j))
        return filled_positions

    @staticmethod
    def get_empty_nodes(board):
        empty_positions = []
        for i in board:
            for j in board:
                if board[i][j] == 0:
                    empty_positions.append((i, j))
        return empty_positions

    @staticmethod
    def get_legal_positions(board):
        legal_positions = []
        for i in board:
            for j in board:
                if board[i][j] != None:
                    legal_positions.append((i, j))
        return legal_positions

    @classmethod
    def initialize_board(cls, board, size, edges, board_type):
        cls.graph = nx.Graph()

        legal_positions = Visualize.get_legal_positions(board)

        # Creates a Hex Diamon grid
        if board_type == Shape.Diamond:
            for i in range(size):
                for j in range(size):
                    Visualize.add_node_to_graph(Visualize.graph, (i, j))

        # Create a Hex Triangle grid
        elif board_type == Shape.Triangle:
            for i in range(size):
                for j in range(i + 1):
                    Visualize.add_node_to_graph(Visualize.graph, (i, j))

        for x, y in legal_positions:
            for row_offset, column_offset in edges:
                neighbor_node = (x + row_offset, y + column_offset)
                if neighbor_node in legal_positions:
                    Visualize.add_edge_to_graph(
                        Visualize.graph, (x, y), neighbor_node)

    @staticmethod
    def draw_board(board, action_nodes, delay, positions=None):

        # List of all node positions currently filled
        filled_nodes = Visualize.get_filled_nodes(board)
        empty_nodes = Visualize.get_empty_nodes(board)

        # Remove nodes currently active
        for nodes in action_nodes:
            filled_nodes.remove(nodes)

        # Draw the resulting grid
        nx.draw_networkx_nodes(Visualize.graph, positions,
                               nodelist=empty_nodes, node_color='black')
        nx.draw_networkx_nodes(Visualize.graph, positions,
                               nodelist=filled_nodes, node_color='blue')
        nx.draw_networkx_nodes(Visualize.graph, positions,
                               nodelist=action_nodes[0], node_color='green')
        nx.draw_networkx_nodes(Visualize.graph, positions,
                               nodelist=action_nodes[1], node_color='red')
        nx.draw_networkx_edges(Visualize.graph, positions, width=1)

        # Takes in delay for each move. Delay given in seconds
        time.sleep(delay * 1000)

        plt.axis('off')
        plt.draw()
        plt.clf()
