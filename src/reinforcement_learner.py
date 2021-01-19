from actor import Actor
from critic import Critic, CriticFactory
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
        self.critic = CriticFactory.get_critic(env)

        self.simulated_world = SimulatedWorld(Shape.Diamond, 4)
        self.episodes = env.episodes

    def run(self) -> None:
        episode = 0
        while episode < self.episodes:

            self.actor.reset_eligibilities()
            self.critic.reset_eligibilities()

            state = self.simulated_world.produce_initial_state()
            action = self.actor.choose_action(state)

            done = False

            while not done:

                next_state, reward, done = self.simulated_world.step(action)
                next_action = self.actor.choose_action(next_state)

                self.actor.replace_eligibilities(state, action)
                self.critic.replace_eligibilities(state)

                td_error = self.critic.td_error(state, next_state, reward)

                # for (s, a) in episode:
                self.actor.update_policy(state, action, td_error)
                self.actor.update_eligibilities(state, action)
                self.critic.update_value(state, td_error)
                self.critic.update_eligibilities(state)

                state, action = next_state, next_action

            episode += 1
