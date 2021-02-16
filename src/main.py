from reinforcement_learner import ReinforcementLearner
import parameters

if __name__ == "__main__":
    for batch_number in range(parameters.BATCHES):
        agent = ReinforcementLearner()
        agent.run(batch_number)
