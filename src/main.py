from parameters import get_parameters
from reinforcement_learner import ReinforcementLearner

if __name__ == "__main__":
    parameters = get_parameters()

    agent = ReinforcementLearner(parameters)
    agent.run()
