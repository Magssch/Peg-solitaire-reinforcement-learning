import json
from typing import Tuple

from data_classes import Shape


class Parameters:

    __is_init = False

    @classmethod
    def get_parameters(cls):
        if cls.__is_init:
            return cls
        with open('src/pivotal_parameters.json', 'r') as f:
            pivotal_parameters = json.load(f)
            for attr, value in pivotal_parameters.items():
                setattr(cls, attr, value)

            if cls.board_type == 1:
                cls.board_type = Shape.Diamond
            else:
                cls.board_type = Shape.Triangle
        cls.__is_init = True
        return cls

    episodes: int
    visualize_games: bool
    frame_delay: float

    # Simulated World
    board_type: Shape
    size: int
    holes: Tuple[int]

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
