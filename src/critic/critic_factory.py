from .critic import Critic
from .nn_critic import NNCritic
from .table_critic import TableCritic


class CriticFactory:

    @staticmethod
    def get_critic(
        use_table_critic: bool,
        critic_learning_rate: float,
        critic_discount_factor: float,
        critic_trace_decay: float,
        critic_nn_dimensions: tuple,
    ) -> Critic:
        """
        Constructs either a TableCritic or NNCritic based on use_table_critic.

        Parameters
        ----------
            use_table_critic : bool
                Whether or not to use table Critic or NN-based Critic
            critic_learning_rate : float
                Learning rate for the Critic
            critic_discount_factor : float
                Discount factor for the Critic
            critic_trace_decay : float
                Trace decay for the Critic
            critic_nn_dimensions : tuple
                Dimensions for the neural network (if used)

        Returns
        -------
        Critic
        """
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
