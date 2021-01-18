from environment import Environment
from reinforcement_learner import ReinforcementLearner
from simulated_world import Shape, SimulatedWorld

if __name__ == "__main__":
    # Pivotal paramters / env
    env = Environment()

    env.episodes = 500
    env.visualize_games = False

    # Simulated World
    env.board_type = Shape.Diamond
    env.size = 4
    env.holes = []

    # Actor
    env.actor_learning_rate = 0.4
    env.actor_trace_decay = 0.9
    env.actor_discount_factor = 0.9
    env.actor_epsilon = 0.5
    env.actor_epsilon_decay_rate = 0.5

    # Critic
    env.critic_learning_rate = 0.1
    env.critic_trace_decay = 0.9
    env.critic_discount_factor = 0.9
    env.critic_nn_dimentions = (15, 20, 30, 5, 1)

    world = SimulatedWorld(env.board_type, env.size)
    agent = ReinforcementLearner(env)
