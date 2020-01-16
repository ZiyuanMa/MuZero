
from environment import *
from network import *
from config import *
from typing import Dict, List, Optional
import math
import numpy as np
import enum
import collections
MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

# noinspection PyArgumentList
Winner = enum.Enum("Winner", "black white draw")

# noinspection PyArgumentList
Player = enum.Enum("Player", "black white")

class MinMaxStats(object):

  """A class that holds the min-max values of the tree."""
  def __init__(self, known_bounds: Optional[KnownBounds]):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value


# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def muzero(config: MuZeroConfig):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    for _ in range(config.num_actors):
        launch_job(run_selfplay, config, storage, replay_buffer)

    train_network(config, storage, replay_buffer)

    return storage.latest_network()

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: MuZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
    while True:
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)

# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, network: Network) -> Game:
    game = Game(64, 1)

    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        expand_node(root, game.to_play(), game.legal_actions(),
                network.initial_inference(current_observation))
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network)
        game.apply(action)
        game.store_search_statistics(root)
    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
@torch.no_grad()
def run_mcts(config: MuZeroConfig, root: Node, action_history: ActionHistory,
             network: Network):
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        action_board = history.last_action().encode()
        network_output = network.recurrent_inference(parent.hidden_state,
                                                action_board)
        expand_node(node, history.to_play(), history.action_space(), network_output)

        backpropagate(search_path, network_output.value, history.to_play(),
                    config.discount, min_max_stats)


def select_action(config: MuZeroConfig, num_moves: int, node: Node,
                  network: Network):
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    t = config.visit_softmax_temperature_fn(
        num_moves=num_moves, training_steps=network.training_steps())
    action = softmax_sample(visit_counts, t)
    return action


# Select the child with the highest UCB score.
def select_child(config: MuZeroConfig, node: Node,
                 min_max_stats: MinMaxStats):
    _, action, child = max(
        (ucb_score(config, node, child, min_max_stats), action,
        child) for action, child in node.children.items())
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: MuZeroConfig, parent: Node, child: Node,
                min_max_stats: MinMaxStats) -> float:
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value())
    return prior_score + value_score


# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(node: Node, to_play: Player, actions: List[Action],
                network_output: NetworkOutput):
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward

    # softmax the policy values
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play: Player,
                  discount: float, min_max_stats: MinMaxStats):
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: MuZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########


def train_network(config: MuZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
    network = Network(config.action_space_size).to(device)

    optimizer = optim.SGD(network.parameters(), lr=0.01, weight_decay=config.lr_decay_rate,
                            momentum=config.momentum)

    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        update_weights(optimizer, network, batch, config.weight_decay)
    storage.save_network(config.training_steps, network)


def update_weights(optimizer: torch.optim, network: Network, batch,
                   weight_decay: float):
    network.train()
    
    p_loss, v_loss = 0, 0

    for image, actions, targets in batch:
        # Initial step, from the real observation.
        value, reward, policy_logits, hidden_state = network.initial_inference(image)
        predictions = [(1.0, value, reward, policy_logits)]

    # Recurrent steps, from action and previous hidden state.
        for action in actions:
            value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, action)
            predictions.append((1.0 / len(actions), value, reward, policy_logits))

        for prediction, target in zip(predictions, targets):
            if(len(target[2]) > 0):
                _ , value, reward, policy_logits = prediction
                target_value, target_reward, target_policy = target

                p_loss += torch.sum(-torch.Tensor(np.array(target_policy)).to(device) * torch.log(policy_logits))
                v_loss += torch.sum((torch.Tensor([target_value]).to(device) - value) ** 2)
  
    optimizer.zero_grad()    
    total_loss = (p_loss + v_loss)
    total_loss.backward()
    optimizer.step()
    network.steps += 1
    print('p_loss %f v_loss %f' % (p_loss / len(batch), v_loss / len(batch)))


def scalar_loss(prediction, target) -> float:
    # MSE in board games, cross entropy between categorical values in Atari.
    return -1

def softmax_sample(distribution, temperature: float):
    visits = [i[0] for i in distribution]
    actions = [i[1] for i in distribution]
    if temperature == 0:
        return actions[visits.index(max(visits))]
    elif temperature == 1:
        visits_sum = sum(visits)
        visits_prob = [i/visits_sum for i in visits]
        return np.random.choice(actions, 1, visits_prob).item()
    else:
        raise NotImplementedError
def launch_job(f, *args):
    f(*args)

def make_uniform_network():
    return Network(make_reversi_config().action_space_size).to(device)
config = make_reversi_config()
# vs_random_once = random_vs_random()
# print('random_vs_random = ', sorted(vs_random_once.items()), end='\n')
network = muzero(config)