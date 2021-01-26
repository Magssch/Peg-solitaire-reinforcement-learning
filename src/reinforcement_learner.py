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

    def __init__(self, parameters: Parameters):
        self.__actor = Actor(
            parameters.actor_learning_rate,
            parameters.actor_discount_factor,
            parameters.actor_trace_decay,
            parameters.actor_epsilon,
            parameters.actor_epsilon_decay,
        )
        self.__critic = CriticFactory.get_critic(
            parameters.use_table_critic,
            parameters.critic_learning_rate,
            parameters.critic_discount_factor,
            parameters.critic_trace_decay,
            parameters.critic_nn_dimensions,
        )

        self.simulated_world = SimulatedWorld(parameters.board_type, parameters.size)
        self.episodes = parameters.episodes

    def run(self) -> None:
        """Runs all episodes with pivotal parameters"""
        episode = 0
        while episode < self.episodes:

            self.__actor.reset_eligibilities()
            self.__critic.reset_eligibilities()

            state, possible_actions = self.simulated_world.reset()
            action = self.__actor.choose_action(state, possible_actions)

            done = False

            while not done:

                next_state, reward, done, possible_actions = self.simulated_world.step(action)
                next_action = self.__actor.choose_action(next_state, possible_actions)

                self.__actor.replace_eligibilities(state, action)
                self.__critic.replace_eligibilities(state)

                td_error = self.__critic.td_error(state, next_state, reward)

                self.__critic.update(state, next_state, reward)
                self.__actor.update(td_error)

                state, action = next_state, next_action

            episode += 1
