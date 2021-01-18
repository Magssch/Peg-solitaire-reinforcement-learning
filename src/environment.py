from simulated_world import Shape


class Environment:

    episodes: int
    visualize_games: bool

    # Simulated World
    board_type: Shape
    size: int
    holes: list

    # Actor
    actor_learning_rate: float
    actor_trace_decay: float
    actor_discount_factor: float
    actor_epsilon: float
    actor_epsilon_decay_rate: float

    # Critic
    critic_learning_rate: float
    critic_trace_decay: float
    critic_discount_factor: float
    critic_nn_dimentions: tuple
