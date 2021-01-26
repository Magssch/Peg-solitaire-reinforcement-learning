from parameters import get_parameters
from reinforcement_learner import ReinforcementLearner
from simulated_world import SimulatedWorld

if __name__ == "__main__":
    parameters = get_parameters()

    agent = ReinforcementLearner(parameters)
    print(agent.critic.reset_eligibilities.__doc__)
    simulated_world = SimulatedWorld(parameters.board_type, parameters.size)
    agent.run()
