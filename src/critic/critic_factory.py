from .critic import Critic
from .nn_critic import NNCritic
from .table_critic import TableCritic


class CriticFactory:

    @classmethod
    def get_critic(
        cls,
        critic_learning_rate: float,
        critic_discount_factor: float,
        critic_trace_decay: float,
        critic_nn_dimensions: tuple = None,
    ) -> Critic:
        if critic_nn_dimensions is None:
            return TableCritic(
                critic_learning_rate,
                critic_discount_factor,
                critic_trace_decay,
            )
        else:
            return NNCritic(
                critic_learning_rate,
                critic_discount_factor,
                critic_trace_decay,
                critic_nn_dimensions,
            )
