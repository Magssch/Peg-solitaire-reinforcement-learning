from typing import Tuple

from data_classes import Shape


class Parameters:

    episodes: int = 500
    visualize_games: bool = False
    frame_delay: float = 0.3

    # Simulated World
    board_type: Shape = Shape.Diamond
    size: int = 5
    holes: Tuple[int] = tuple([0, 1])

    # Actor
    actor_learning_rate: float = 0.1
    actor_discount_factor: float = 0.9
    actor_trace_decay: float = 0.8
    actor_epsilon: float = 0.5
    actor_epsilon_decay: float = 0.9

    # Critic
    use_table_critic: bool = True
    critic_learning_rate: float = 0.001
    critic_discount_factor: float = 0.9
    critic_trace_decay: float = 0.8
    critic_nn_dimensions: tuple = (15, 20, 30, 5, 1)
