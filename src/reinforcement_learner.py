from actor import Actor
from critic.critic_factory import CriticFactory
from parameters import Parameters
from simulated_world import SimulatedWorld


class ReinforcementLearner:

    def __init__(self, parameters: Parameters):
        self.actor = Actor(
            parameters.actor_learning_rate,
            parameters.actor_discount_factor,
            parameters.actor_trace_decay,
            parameters.actor_epsilon,
            parameters.actor_epsilon_decay,
        )
        self.critic = CriticFactory.get_critic(
            parameters.use_table_critic,
            parameters.critic_learning_rate,
            parameters.critic_discount_factor,
            parameters.critic_trace_decay,
            parameters.critic_nn_dimensions,
        )

        self.simulated_world = SimulatedWorld(parameters.board_type, parameters.size)
        self.episodes = parameters.episodes

    def run(self) -> None:
        episode = 0
        while episode < self.episodes:

            self.actor.reset_eligibilities()
            self.critic.reset_eligibilities()

            state, possible_actions = self.simulated_world.reset()
            action = self.actor.choose_action(state, possible_actions)

            done = False

            while not done:

                next_state, reward, done, possible_actions = self.simulated_world.step(action)
                next_action = self.actor.choose_action(next_state, possible_actions)

                self.actor.replace_eligibilities(state, action)
                self.critic.replace_eligibilities(state)

                td_error = self.critic.td_error(state, next_state, reward)

                self.critic.update(state, next_state, reward)
                self.actor.update(td_error)

                state, action = next_state, next_action

            episode += 1
