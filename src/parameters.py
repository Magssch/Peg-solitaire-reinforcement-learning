import json

from data_classes import Shape


def get_parameters():
    parameters = Parameters()
    with open('src/pivotal_parameters.json', 'r') as f:
        pivotal_parameters = json.load(f)
        for attr, value in pivotal_parameters.items():
            setattr(parameters, attr, value)

        if parameters.board_type == 1:
            parameters.board_type = Shape.Diamond
        else:
            parameters.board_type = Shape.Triangle
    return parameters


class Parameters:

    episodes: int
    visualize_games: bool
    frame_delay: float

    # Simulated World
    board_type: Shape
    size: int
    holes: list

    # Actor
    actor_learning_rate: float
    actor_discount_factor: float
    actor_trace_decay: float
    actor_epsilon: float
    actor_epsilon_decay: float

    # Critic
    use_table_critic: bool
    critic_learning_rate: float
    critic_discount_factor: float
    critic_trace_decay: float
    critic_nn_dimensions: tuple
