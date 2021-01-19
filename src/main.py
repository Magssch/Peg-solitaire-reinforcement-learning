import json

from environment import Environment
from reinforcement_learner import ReinforcementLearner
from simulated_world import Shape

if __name__ == "__main__":
    env = Environment()
    with open('src/pivotal_parameters.json', 'r') as f:
        pivotal_parameters = json.load(f)
        for attr, value in pivotal_parameters.items():
            setattr(env, attr, value)
        if env.board_type == 1:
            env.board_type = Shape.Diamond
        else:
            env.board_type = Shape.Triangle
    print(env.board_type)

    agent = ReinforcementLearner(env)
    agent.run()
