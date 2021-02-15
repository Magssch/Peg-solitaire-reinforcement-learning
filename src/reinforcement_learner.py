import parameters
from actor import Actor
from critic.critic_factory import CriticFactory
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

    def __run_one_episode(self, visualize: bool = False) -> None:
        self.__actor.reset_eligibilities()
        self.__critic.reset_eligibilities()

        state, possible_actions = self.__simulated_world.reset()
        action = self.__actor.choose_action(state, possible_actions)

        done = False

        while not done:
            next_state, reward, done, possible_actions = self.__simulated_world.step(action, visualize)
            next_action = self.__actor.choose_action(next_state, possible_actions)

            self.__actor.replace_eligibilities(state, action)
            self.__critic.replace_eligibilities(state)

            td_error = self.__critic.td_error(reward, next_state, state)

            self.__critic.update(reward, next_state, state)
            self.__actor.update(td_error)

            state, action = next_state, next_action

    def run(self) -> None:
        """
        Runs all episodes with pivotal parameters.
        Visualizes one round at the end.
        """
        for episode in range(self.__episodes):
            print('Episode:', episode + 1)
            self.__run_one_episode()

        print('Training completed.')
        self.__simulated_world.plot_training_data()
        self.__actor.plot_training_data()
        self.__critic.plot_training_data()

        if parameters.VISUALIZE_GAMES:
            print('Showing one episode with the greedy strategy.')
            self.__actor.set_epsilon(0)
            self.__run_one_episode(True)
