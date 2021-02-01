from networkx.classes.graph import Graph
from hexagonal_board import Shape
import networkx as nx
import matplotlib.pyplot as plt


class DrawBoard():

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

    @staticmethod
    def draw_board(board, board_type, action_nodes, size, edges, positions=None):
        graph = nx.Graph()

        # List of all node positions currently filled
        filled_nodes = DrawBoard.get_filled_nodes(board)
        empty_nodes = DrawBoard.get_empty_nodes(board)
        legal_positions = DrawBoard.get_legal_positions(board)


        # Remove nodes currently active
        for nodes in action_nodes:
            filled_nodes.remove(nodes)

        # Creates a Hex Diamon grid
        if board_type == Shape.Diamond:
            for i in range(size):
                for j in range(size):
                    DrawBoard.add_node_to_graph(graph, (i,j))

        # Create a Hex Triangle grid
        elif board_type == Shape.Triangle:
            for i in range(size):
                for j in range(i + 1):
                    DrawBoard.add_node_to_graph(graph, (i, j))

        for x, y in legal_positions:
            for row_offset, column_offset in edges:
                neighbor_node = (x + row_offset, y + column_offset)
                if neighbor_node in legal_positions:
                    DrawBoard.add_edge_to_graph(graph, (x, y), neighbor_node)

        # Draw the resulting grid
        nx.draw_networkx_nodes(graph, positions, nodelist=empty_nodes, node_color='black')
        nx.draw_networkx_nodes(graph, positions, nodelist=filled_nodes, node_color='blue')
        nx.draw_networkx_nodes(graph, positions, nodelist=action_nodes[0], node_color='green')
        nx.draw_networkx_nodes(graph, positions, nodelist=action_nodes[1], node_color='red')
        nx.draw_networkx_edges(graph, positions, width=1)

        plt.axis('off')
        plt.draw()
        plt.clf()
        plt.show()
