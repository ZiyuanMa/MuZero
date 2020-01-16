

round = 15
episodes = 10000
epoch = 3
memory_size = 3     # store 3 rounds game 30000 games
update_rate = 0.65
min_visit_times = 3


batch_size = 1024
filter_num = 64
search_times = 200


board_size = 8
import collections
# MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

# # noinspection PyArgumentList
# Winner = enum.Enum("Winner", "black white draw")

# # noinspection PyArgumentList
# Player = enum.Enum("Player", "black white")

# class MinMaxStats(object):

#     """A class that holds the min-max values of the tree."""
#     def __init__(self, known_bounds: Optional[KnownBounds]):
#         self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
#         self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

#     def update(self, value: float):
#         self.maximum = max(self.maximum, value)
#         self.minimum = min(self.minimum, value)

#     def normalize(self, value: float) -> float:
#         if self.maximum > self.minimum:
#             # We normalize only when we have set the maximum and minimum values.
#             return (value - self.minimum) / (self.maximum - self.minimum)
#         return value
class MuZeroConfig(object):

    def __init__(self,
                action_space_size: int,
                max_moves: int,
                discount: float,
                dirichlet_alpha: float,
                num_simulations: int,
                batch_size: int,
                td_steps: int,
                num_actors: int,
                lr_init: float,
                lr_decay_steps: float,
                visit_softmax_temperature_fn,
                known_bounds = None):
        ### Self-Play
        self.action_space_size = action_space_size
        self.num_actors = num_actors

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.training_steps = int(1e6)
        self.checkpoint_interval = int(100)
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.num_unroll_steps = 4
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

    def new_game(self):
        return Game(self.action_space_size, self.discount)


def make_board_game_config(action_space_size: int, max_moves: int,
                           dirichlet_alpha: float,
                           lr_init: float) -> MuZeroConfig:

    def visit_softmax_temperature(num_moves, training_steps):
        if num_moves < 6:
            return 1.0
        else:
            return 0.0  # Play according to the max.

    return MuZeroConfig(
        action_space_size=action_space_size,
        max_moves=max_moves,
        discount=1.0,
        dirichlet_alpha=dirichlet_alpha,
        num_simulations=10,
        batch_size=64,
        td_steps=max_moves,  # Always use Monte Carlo return.
        num_actors=1,
        lr_init=lr_init,
        lr_decay_steps=400e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds=KnownBounds(-1, 1))

def make_reversi_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=64, max_moves=60, dirichlet_alpha=0.03, lr_init=0.01)
