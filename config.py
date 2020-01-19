
import collections
KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

# class MuZeroConfig(object):

#     def __init__(self,
#                 action_space_size: int,
#                 max_moves: int,
#                 discount: float,
#                 dirichlet_alpha: float,
#                 num_simulations: int,
#                 batch_size: int,
#                 td_steps: int,
#                 num_actors: int,
#                 lr_init: float,
#                 lr_decay_steps: float,
#                 visit_softmax_temperature_fn,
#                 known_bounds = None):
#         ### Self-Play
#         self.action_space_size = action_space_size
#         self.num_actors = num_actors

#         self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
#         self.max_moves = max_moves
#         self.num_simulations = num_simulations
#         self.discount = discount

#         # Root prior exploration noise.
#         self.root_dirichlet_alpha = dirichlet_alpha
#         self.root_exploration_fraction = 0.25

#         # UCB formula
#         self.pb_c_base = 19652
#         self.pb_c_init = 1.25

#         # If we already have some information about which values occur in the
#         # environment, we can use them to initialize the rescaling.
#         # This is not strictly necessary, but establishes identical behaviour to
#         # AlphaZero in board games.
#         self.known_bounds = known_bounds

#         ### Training
#         self.training_steps = int(1e6)
#         self.checkpoint_interval = int(100)
#         self.window_size = int(1e6)
#         self.batch_size = batch_size
#         self.num_unroll_steps = 4
#         self.td_steps = td_steps

#         self.weight_decay = 1e-4
#         self.momentum = 0.9

#         # Exponential learning rate schedule
#         self.lr_init = lr_init
#         self.lr_decay_rate = 0.1
#         self.lr_decay_steps = lr_decay_steps

def visit_softmax_temperature(num_moves, training_steps):
    if num_moves < 6:
        return 1.0
    else:
        return 0.0  # Play according to the max.

# def make_reversi_config() -> MuZeroConfig:

#     return MuZeroConfig(
#         action_space_size=65,
#         max_moves=60,
#         discount=1.0,
#         dirichlet_alpha=0.03,
#         num_simulations=50,
#         batch_size=64,
#         td_steps=60,  # Always use Monte Carlo return.
#         num_actors=1,
#         lr_init=0.01,
#         lr_decay_steps=400e3,
#         visit_softmax_temperature_fn=visit_softmax_temperature,
#         known_bounds=KnownBounds(-1, 1))

    ### Self-Play
action_space_size = 65

visit_softmax_temperature_fn = visit_softmax_temperature
max_moves = 60
num_simulations = 200
discount = 1.0

    # Root prior exploration noise.
root_dirichlet_alpha = 0.03
root_exploration_fraction = 0.25

    # UCB formula
pb_c_base = 19652
pb_c_init = 1.25

    # If we already have some information about which values occur in the
    # environment, we can use them to initialize the rescaling.
    # This is not strictly necessary, but establishes identical behaviour to
    # AlphaZero in board games.
known_bounds = KnownBounds(-1, 1)

    ### Training
training_steps = 1000
checkpoint_interval = 50
window_size = 1024
batch_size = 1024
mini_batch_size = 128
num_unroll_steps = 4
td_steps = 60

weight_decay = 1e-4
momentum = 0.9

    # Exponential learning rate schedule
lr_init = 0.01
lr_decay_rate = 0.1
lr_decay_steps = 400e3

episodes=1024