from actor import Actor
from critic.critic_factory import CriticFactory
from parameters import Parameters
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
            Parameters.actor_learning_rate,
            Parameters.actor_discount_factor,
            Parameters.actor_trace_decay,
            Parameters.actor_epsilon,
            Parameters.actor_epsilon_decay,
        )
        self.__critic = CriticFactory.get_critic(
            Parameters.use_table_critic,
            Parameters.critic_learning_rate,
            Parameters.critic_discount_factor,
            Parameters.critic_trace_decay,
            Parameters.critic_nn_dimensions,
        )

        self.__simulated_world = SimulatedWorld()
        self.__episodes = Parameters.episodes

    def run(self) -> None:
        """Runs all episodes with pivotal parameters"""
        for episode in range(self.__episodes):

            print('Episode: ', episode, end='')

            self.__actor.reset_eligibilities()
            self.__critic.reset_eligibilities()

            state, possible_actions = self.__simulated_world.reset()
            action = self.__actor.choose_action(state, possible_actions)

            done = False

            while not done:

                next_state, reward, done, possible_actions = self.__simulated_world.step(action)
                next_action = self.__actor.choose_action(next_state, possible_actions)

                self.__actor.replace_eligibilities(state, action)
                self.__critic.replace_eligibilities(state)

                td_error = self.__critic.td_error(state, next_state, reward)
                print(f' - td_error={td_error:.4f}')

                self.__critic.update(state, next_state, reward)
                self.__actor.update(td_error)

                state, action = next_state, next_action

        self.__simulated_world.exit()

        print('Session completed.')
