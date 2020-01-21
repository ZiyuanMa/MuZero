
import collections
KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


def visit_softmax_temperature(num_moves, training_steps):
    if num_moves < 6:
        return 1.0
    else:
        return 0.0  # Play according to the max.


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
training_steps = 1500
checkpoint_interval = 100
window_size = 1024
batch_size = 96
mini_batch_size = 24
num_unroll_steps = 4
td_steps = 60

weight_decay = 1e-4
momentum = 0.9

# Exponential learning rate schedule
lr_init = 0.01
lr_decay_rate = 0.1
lr_decay_steps = 400e3

episodes=12