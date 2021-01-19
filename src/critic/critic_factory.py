from .table_critic import TableCritic
from .nn_critic import NNCritic


class CriticFactory:

    @classmethod
    def get_critic(
        cls,
        critic_learning_rate: float,
        critic_trace_decay: float,
        critic_discount_factor: float,
        critic_nn_dimensions: tuple = None,
    ):
        if critic_nn_dimensions is None:
            return TableCritic(
                critic_learning_rate,
                critic_trace_decay,
                critic_discount_factor,
            )
        else:
            return NNCritic(
                critic_learning_rate,
                critic_trace_decay,
                critic_discount_factor,
                critic_nn_dimensions,
            )
