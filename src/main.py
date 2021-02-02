import numpy as np

from parameters import get_parameters
from reinforcement_learner import ReinforcementLearner
from simulated_world import SimulatedWorld
from visualize import TrainingData

if __name__ == "__main__":
    parameters = get_parameters()

    training_data = np.random.choice(range(10), parameters.episodes)
    TrainingData.plot(training_data)
