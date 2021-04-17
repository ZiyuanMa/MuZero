import os
os.environ["OMP_NUM_THREADS"] = "1"
from utilities import *
from network import *
import config
from typing import Dict, List, Optional
import math
import numpy as np
import torch
from torch.utils.data import Subset
import torch.multiprocessing as mp
import ray
import subprocess
import fnmatch
from tqdm import tqdm
import random
from reversi import *
torch.manual_seed(1261)
random.seed(1261)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# MuZero training is split into two independent parts: Network training and
# self-play data generation.
def muzero():
    ray.init()
    replay_buffer = ReplayBuffer.remote()
    
    network, optimizer = load_model()
    network_id = ray.put(network.state_dict())
    storage = SharedStorage(network_id)
    for _ in range(20):
        
        run_selfplay.remote(network, storage, replay_buffer)

    while not ray.get(replay_buffer.ready.remote()):
        time.sleep(5)

    train_network(network, optimizer, storage, replay_buffer)




def load_model():
    
    network = Network()
    network.share_memory()
    optimizer = optim.SGD(network.parameters(), lr=0.01, weight_decay=config.lr_decay_rate, momentum=config.momentum)
    model_name = None

    for filename in os.listdir('.'):
        if fnmatch.fnmatch(filename, 'model*.pth'):
            model_name = filename

    if model_name:
        checkpoint = torch.load(model_name)
        network.load_state_dict(checkpoint['network'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('find model, load %d steps model' % network.steps)
    else:
        print('no model, create new model')
        torch.save({
            'network': network.state_dict(),
            'optimizer': optimizer.state_dict()
        }, 'model0.pth')

    return network, optimizer

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
@ray.remote(num_cpus=1)
def run_selfplay(network, storage, replay_buffer: ReplayBuffer):
    print('start self-play')

    while True:
        network_id = storage.get_network()
        network.load_state_dict(ray.get(network_id))
        game = play_game(network)
        replay_buffer.save_game.remote(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(network) -> Game:
    game = Game()

    while not game.terminal() and len(game.action_history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0, game.to_play())
        current_observation = game.make_image(-1)
        current_observation = torch.from_numpy(current_observation)
        with torch.no_grad():
            # print(current_observation.size())
            net_output = network.initial_inference(current_observation)
        # print(game.legal_actions())
        expand_node(root, game.legal_actions(), net_output)
        add_exploration_noise(root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(root, game.action_history, network)
        action = select_action(len(game.action_history), root, network)
        game.apply(Action(action))
        game.store_search_statistics(root)

    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(root: Node, action_history: List, network: Network):
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        history = action_history.copy()
        node = root
        search_path = [node]

        # choose node based on score if it already exists in tree
        while node.expanded():
            action, node = select_child(node, min_max_stats)
            history.append(action)
            search_path.append(node)

        # go untill a unexpanded node, expand by using recurrent inference then backup
        parent = search_path[-2]
        last_action = history[-1]

        encoded_action = np.zeros((2, 8, 8), dtype=np.float32)
        player_idx = 0 if last_action is Player.WHITE else 1
        encoded_action[player_idx, last_action.position] = 1
        with torch.no_grad():
            network_output = network.recurrent_inference(parent.hidden_state, torch.from_numpy(encoded_action))

        expand_node(node, [i for i in range(config.action_space_size)], network_output)

        backpropagate(search_path, network_output[0], history[-1].player,
                    config.discount, min_max_stats)


def select_action(num_moves: int, node: Node, network: Network):
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]

    if num_moves < 6:
        t = 1.0
    else:
        t = 0.0  # Play according to the max.

    action = softmax_sample(visit_counts, t)
    return action


# Select the child with the highest UCB score.
def select_child(node: Node, min_max_stats: MinMaxStats):
    _, action_idx, child = max(
        (ucb_score(node, child, min_max_stats), action,
        child) for action, child in node.children.items())
    return Action(action_idx), child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value())
    return prior_score + value_score


# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(node: Node, actions: List[int], network_output):
    node.hidden_state = network_output[2]
    # node.reward = network_output.reward
    node.reward = 0

    # filter illegal actions only for first expansion of mcts
    policy = {a: network_output[1][0, a].item() for a in actions}
    policy_sum = sum(policy.values())
    next_player = Player.WHITE if node.to_play is Player.BLACK else Player.BLACK
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum, next_player)


# At the end of a simulation, we propagate the evaluation all the way up to the
# tree of the root.
def backpropagate(search_path: List[Node], value: float, to_play: int, discount: float, min_max_stats: MinMaxStats):
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

##################################
####### Part 2: Training #########

def train_network(network, optimizer, storage:SharedStorage, replay_buffer: ReplayBuffer):

    network.train()
    network.to(device)

    # data_set = Dataset(data)

    print('start training')
    for i in range(1, config.training_steps+1):

        # sub_data_set = Subset(data_set, random.sample(range(len(data_set)), config.batch_size))
        # data_loader = DataLoader(dataset=sub_data_set,
        #                     num_workers=4,
        #                     batch_size=config.mini_batch_size,
        #                     pin_memory = True,
        #                     shuffle=True)
        batch = replay_buffer.sample_batch.remote(config.num_unroll_steps, config.td_steps)
        batch = ray.get(batch)

        update_weights(optimizer, network, batch)

        if i % config.checkpoint_interval == 0:
            state_dict = network.state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
            network_id = ray.put(state_dict)
            storage.save_network(network_id)


def update_weights(optimizer: torch.optim, network: Network, batch):

    optimizer.zero_grad()
    p_loss, v_loss = 0, 0

    image, actions, target_values, target_policies = batch
    image, actions, target_values, target_policies = torch.FloatTensor(image), torch.FloatTensor(actions), torch.FloatTensor(target_values), torch.FloatTensor(target_policies)
    image, actions, target_values, target_policies = image.to(device), actions.to(device), target_values.to(device), target_policies.to(device)

    # Initial step, from the real observation.
    value, policy, hidden_state = network.initial_inference(image)

    p_value, p_policy = [], []
    p_value.append(value)
    p_policy.append(policy)

    # Recurrent steps, from action and previous hidden state.
    for action in actions:

        value, policy, hidden_state = network.recurrent_inference(hidden_state, action)

        p_value.append(value)
        p_policy.append(policy)

    p_value = torch.stack(p_value).squeeze()
    p_policy = torch.stack(p_policy)

    # p_value = p_value.view(config.batch_size, config.num_unroll_steps+1)
    # p_policy = p_policy.view(config.batch_size, config.num_unroll_steps+1, config.action_space_size)

    target_policies = target_policies.transpose(0, 1)
    p_policy = p_policy.transpose(0, 1)
    target_values = target_values.transpose(0, 1)
    p_value = p_value.transpose(0, 1)
    p_loss += torch.mean(torch.sum(-target_policies * torch.log(p_policy), dim=2))
    v_loss += torch.mean(torch.sum((target_values - p_value) ** 2, dim=1))
  
    total_loss = (p_loss + v_loss)
    total_loss.backward()
    optimizer.step()
    print('step {}: p_loss {:.4f} v_loss {:.4f}'.format(network.steps, p_loss, v_loss))
    network.steps += 1


def softmax_sample(distribution, temperature: float):
    visits = [i[0] for i in distribution]
    actions = [i[1] for i in distribution]
    if temperature == 0:
        return actions[visits.index(max(visits))]
    elif temperature == 1:
        visits_sum = sum(visits)
        visits_prob = [i/visits_sum for i in visits]
        return np.random.choice(actions, p=visits_prob)
    else:
        raise NotImplementedError
        

if __name__ == '__main__':

    muzero()

# %%
