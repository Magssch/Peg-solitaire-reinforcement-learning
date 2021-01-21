import json

from parameters import Parameters
from reinforcement_learner import ReinforcementLearner
from simulated_world import Shape

if __name__ == "__main__":
    paramters = Parameters()
    with open('src/pivotal_parameters.json', 'r') as f:
        pivotal_parameters = json.load(f)
        for attr, value in pivotal_parameters.items():
            setattr(paramters, attr, value)
        if paramters.board_type == 1:
            paramters.board_type = Shape.Diamond
        else:
            paramters.board_type = Shape.Triangle
    print(paramters.board_type)

    agent = ReinforcementLearner(paramters)
    agent.run()
