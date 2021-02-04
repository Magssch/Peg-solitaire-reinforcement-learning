from actor import Actor
from critic.critic_factory import CriticFactory
import parameters
from simulated_world import SimulatedWorld


class ReinforcementLearner:
    """
    Reinforcement Learner agent using the Actor-Critic architecture

    ...

    Attributes
    ----------

    Methods
    -------
    run():
        Runs all episodes with pivotal parameters
    """

    def __init__(self):
        self.__actor = Actor(
            parameters.ACTOR_LEARNING_RATE,
            parameters.ACTOR_DISCOUNT_FACTOR,
            parameters.ACTOR_TRACE_DECAY,
            parameters.ACTOR_EPSILON,
            parameters.ACTOR_EPSILON_DECAY,
        )
        self.__critic = CriticFactory.get_critic(
            parameters.USE_TABLE_CRITIC,
            parameters.CRITIC_LEARNING_RATE,
            parameters.CRITIC_DISCOUNT_FACTOR,
            parameters.CRITIC_TRACE_DECAY,
            parameters.CRITIC_NN_DIMENSIONS,
        )

        self.__simulated_world = SimulatedWorld()
        self.__episodes = parameters.EPISODES

    def run(self) -> None:
        """Runs all episodes with pivotal parameters"""
        for episode in range(self.__episodes):

            print('Episode:', episode)

            self.__actor.reset_eligibilities()
            self.__critic.reset_eligibilities()

            state, possible_actions = self.__simulated_world.reset()
            action = self.__actor.choose_action(state, possible_actions)

            done = False

            while not done:
                next_state, reward, done, possible_actions = self.__simulated_world.step(action)

                self.__actor.replace_eligibilities(state, action)
                self.__critic.replace_eligibilities(state)

                td_error = self.__critic.td_error(state, next_state, reward)
                print(f'  td_error={td_error:>7.4f}')

                self.__critic.update(state, next_state, reward)
                self.__actor.update(td_error)

                if done:
                    break

                state, action = next_state, self.__actor.choose_action(next_state, possible_actions)

        self.__simulated_world.exit()

        print('Session completed.')
