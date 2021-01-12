from simulated_world import Shape

if __name__ == "__main__":
    # Pivotal paramters / env
    board_type = Shape.Diamond
    size = 4
    holes = []

    episodes = 500
    value_function = 'neural_network'
    ANN_dimentions = (15, 20, 30, 5, 1)

    actor_learning_rate = 0.4
    actor_trace_decay = 0.9
    actor_discount_factor = 0.9
    actor_epsilon = 0.5
    actor_epsilon_decay_rate = 0.5

    critic_learning_rate = 0.1
    critic_trace_decay = 0.9
    critic_discount_factor = 0.9

    visualize_games = False
