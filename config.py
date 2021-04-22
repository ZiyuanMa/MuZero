

def visit_softmax_temperature(num_moves, training_steps):
    if num_moves < 6:
        return 1.0
    else:
        return 0.0  # Play according to the max.


action_space_size = 65

visit_softmax_temperature_fn = visit_softmax_temperature
max_moves = 60
num_simulations = 60
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
known_bounds = (-1, 1)

# Training
training_steps = 100000
checkpoint_interval = 1
batch_size = 512
window_size = 512
num_unroll_steps = 3
td_steps = 60
training_interval = 0.5

weight_decay = 1e-4
momentum = 0.9

# Exponential learning rate schedule
lr_init = 0.01
lr_decay_rate = 0.1
lr_decay_steps = 400e3

state_history_len = 4
state_shape = (state_history_len*2+1, 8, 8)
num_channels = 128