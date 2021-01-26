from .critic import Critic
from .nn_critic import NNCritic
from .table_critic import TableCritic


class CriticFactory:

    @classmethod
    def get_critic(
        cls,
        use_table_critic: bool,
        critic_learning_rate: float,
        critic_discount_factor: float,
        critic_trace_decay: float,
        critic_nn_dimensions: tuple,
    ) -> Critic:
        if use_table_critic:
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
