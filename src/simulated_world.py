# This is the specialized domain class for Peg Solitaire,
# it should do the following:
#
# - Understands game states and the operators that convert one game state to
# another.
# - Produces initial game states.
# - Generates child states from parent states using the legal operators of the
#  domain.
# - Recognizes final (winning, losing and neutral) states
class SimulatedWorld:

    def __init__(self):
        pass

    def produce_initial_state(self):
        pass

    def generate_child_states(self):
        pass

    def is_final_state(self):
        pass
