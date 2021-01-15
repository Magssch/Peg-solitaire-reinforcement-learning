from simulated_world import Shape


class Environment:
    board_type: Shape
    size: int
    holes: list

    episodes: int
    ANN_dimentions: tuple

    actor_learning_rate: float
    actor_trace_decay: float
    actor_discount_factor: float
    actor_epsilon: float
    actor_epsilon_decay_rate: float

    critic_learning_rate: float
    critic_trace_decay: float
    critic_discount_factor: float

    visualize_games: bool
