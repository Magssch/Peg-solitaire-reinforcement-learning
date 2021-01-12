from actor import Actor
from critic import Critic
from simulated_world import Shape, SimulatedWorld


class ReinforcementLearner:

    def __init__(self, episodes):
        self.actor = Actor(0.5, 0.5, 0.5, 0.5)
        self.critic = Critic(0.5, 0.5, True, None)
        self.simulated_world = SimulatedWorld(Shape.Diamond, 4)
        self.episodes = episodes

    def run(self):
        episode = 0
        while episode < self.episodes:

            self.actor.reset_traces()
            self.critic.reset_traces()

            state = self.simulated_world.produce_initial_state()
            action = self.actor.choose_mixed(state)

            while not self.simulated_world.is_final_state():

                next_state, reward = self.simulated_world.do_action(action)
                next_action = self.actor.choose_mixed(state)

                self.actor.replace_trace(state, action)
                self.critic.replace_trace(state)

                td_error = self.critic.td_error(state, next_state, reward)

                # for (s, a) in episode: ??
                self.actor.update_policy(state, action, td_error)
                self.actor.update_trace(state, action)
                self.critic.update_value(state, td_error)
                self.critic.update_trace(state)

                state, action = next_state, next_action

            episode += 1
