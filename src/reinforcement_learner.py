from actor import Actor
from critic import Critic
from main import Environment
from simulated_world import Shape, SimulatedWorld


class ReinforcementLearner:

    def __init__(self, env: Environment):
        self.actor = Actor(
            env.actor_learning_rate,
            env.actor_trace_decay,
            env.actor_discount_factor,
            env.actor_epsilon,
            env.actor_nn_dimensions,
        )
        self.critic = Critic(
            env.critic_learning_rate,
            env.critic_trace_decay,
            env.critic_discount_factor,
            env.critic_nn_dimensions,
        )
        self.simulated_world = SimulatedWorld(Shape.Diamond, 4)
        self.episodes = env.episodes

    def run(self) -> None:
        episode = 0
        while episode < self.episodes:

            self.actor.reset_eligibilities()
            self.critic.reset_eligibilities()

            state = self.simulated_world.produce_initial_state()
            action = self.actor.choose_mixed(state)

            while not self.simulated_world.is_final_state():

                next_state, reward = self.simulated_world.do_action(action)
                next_action = self.actor.choose_mixed(next_state)

                self.actor.replace_trace(state, action)
                self.critic.replace_trace(state)

                td_error = self.critic.td_error(state, next_state, reward)

                # for (s, a) in episode:
                self.actor.update_policy(state, action, td_error)
                self.actor.update_trace(state, action)
                self.critic.update_value(state, td_error)
                self.critic.update_trace(state)

                state, action = next_state, next_action

            episode += 1
