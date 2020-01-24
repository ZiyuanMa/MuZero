from network import *
from environment import *
import config
import torch
import torch.multiprocessing as mp
import os
import math
import fnmatch
import pickle
from tqdm import tqdm
os.environ["OMP_NUM_THREADS"] = "1"




def load_network():
    network = Network()
    model_name = None

    for filename in os.listdir('.'):
        if fnmatch.fnmatch(filename, 'model*.pth'):
            model_name = filename

    if model_name: 
        checkpoint = torch.load(model_name)
        network.load_state_dict(checkpoint['network'])
        network.eval()
    else:
        raise RuntimeError('no model')

    return network

# def load_buffer():


def run_selfplay():
    # for _ in tqdm(range(30)):
    #     network = storage.latest_network()
    #     game = play_game(network)
    #     replay_buffer.save_game(game)

    # network = load_network()
    # network.share_memory()
    # replay_buffer = ReplayBuffer()
    # with mp.Pool(4) as p:
    #     for _ in tqdm(range(config.episodes)):
    #         p.apply(play_game, args=(network,))
    print('start self play')
    network = load_network()
    network.share_memory()
    replay_buffer = ReplayBuffer()
    with mp.Pool(os.cpu_count()) as p:
        pbar = tqdm(total=config.episodes)
        def update(ret):
            pbar.update()
            replay_buffer.save_game(ret)

        for _ in range(config.episodes):
            p.apply_async(play_game, args=(network,), callback= update)
        p.close()
        p.join()
        pbar.close()

    data = replay_buffer.generate_data()
    data_set = Dataset(data)
    with open('./data_set.pth','w+') as f:
        pickle.dump(data_set, f)

# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
@torch.no_grad()
def play_game(network: Network) -> Game:
    game = Game()

    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0, game.to_play())
        current_observation = game.make_image(-1)
        current_observation = torch.from_numpy(current_observation)
        net_output = network.initial_inference(current_observation)
        expand_node(root, game.legal_actions(), net_output)
        add_exploration_noise(root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(root, game.action_history(), network)
        action = select_action(len(game.history), root, network)
        game.apply(action)
        game.store_search_statistics(root)

    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
@torch.no_grad()
def run_mcts(root: Node, action_history: ActionHistory, network: Network):
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        # choose node based on score if it already exists in tree
        while node.expanded():
            action, node = select_child(node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # go untill a unexpanded node, expand by using recurrent inference then backup
        parent = search_path[-2]
        encoded_action = history.last_action().encode()
        encoded_action = torch.from_numpy(encoded_action)
        network_output = network.recurrent_inference(parent.hidden_state, encoded_action)
        expand_node(node, history.action_space(), network_output)

        backpropagate(search_path, network_output.value, history.to_play(),
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
    _, action, child = max(
        (ucb_score(node, child, min_max_stats), action,
        child) for action, child in node.children.items())
    return action, child


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
def expand_node(node: Node, actions: List[Action], network_output: NetworkOutput):
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward

    # filter illegal actions only for first expansion of mcts
    policy = {a: network_output.policy_logits[0][a.index].item() for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum, -node.to_play)


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

if __name__ == '__main__':
    mp.set_start_method('spawn')
    run_selfplay()