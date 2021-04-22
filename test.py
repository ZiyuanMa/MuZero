from typing import List
import math
import multiprocessing as mp
import torch
import numpy as np
from reversi import Environment, Game, Node, Action, Player
from network import Network
import config

torch.set_num_threads(1)

class Controller:
    def __init__(self, player: Player, network: Network):
        self.player = player
        self.network = network
        self.state_history = [np.zeros((2, 8, 8), dtype=np.bool) for _ in range(config.state_history_len)]

    def step(self, state, player: Player, legal_actions):
        self.state_history.pop(0)
        self.state_history.append(state)

        if player == self.player:
            root = Node(0, player)
            # print(self.state_history)
            obs = np.zeros(config.state_shape, dtype=np.float32)
            obs[:-1] = np.concatenate(self.state_history)
            # print(obs.shape)

            if player is Player.BLACK: obs[-1] = 1

            with torch.no_grad():
                net_output = self.network.initial_inference(torch.from_numpy(obs))
            
            # print(legal_actions)
            legal_actions = [a.index for a in legal_actions]
            # print(legal_actions)
            expand_node(root, legal_actions, net_output)

            # We then run a Monte Carlo Tree Search using only action sequences and the
            # model learned by the network.
            run_mcts(root, self.network)
            action = select_action(root, self.network)
            return Action(action, player)

def test1(args):

    base, target = args
    env = Environment()
    black_c = Controller(Player.BLACK, base)
    white_c = Controller(Player.WHITE, target)
    while not env.done:
        obs = env.observe()
        # print(env.player)
        a_b = black_c.step(obs, env.player, env.legal_actions)
        a_w = white_c.step(obs, env.player, env.legal_actions)
        # print(a_b)
        # print(a_w)
        if a_b:
            # print(a_b)
            env.step(a_b)
        else:
            # print(a_w)
            env.step(a_w)
        
        # print(env.steps)

    white_score = np.sum(env.board==1)
    black_score = np.sum(env.board==-1)
    print(white_score)
    if white_score > black_score:
        return 1
    else:
        return 0

def test2(args):
    base, target = args
    env = Environment()
    black_c = Controller(Player.BLACK, target)
    white_c = Controller(Player.WHITE, base)
    while not env.done:
        obs = env.observe()
        # print(env.player)
        a_b = black_c.step(obs, env.player, env.legal_actions)
        a_w = white_c.step(obs, env.player, env.legal_actions)
        # print(a_b)
        # print(a_w)
        if a_b:
            # print(a_b)
            env.step(a_b)
        else:
            # print(a_w)
            env.step(a_w)
        
        

    white_score = np.sum(env.board==1)
    black_score = np.sum(env.board==-1)
    print(black_score)
    if white_score < black_score:
        return 1
    else:
        return 0


def test(base, target):
    test1((base, target))
    pool = mp.Pool(mp.cpu_count())
    params = [(base, target) for _ in range(10)]
    ret1 = pool.map(test1, params)
    ret2 = pool.map(test2, params)
    print(sum(ret1)+sum(ret2))



            


class MinMaxStats:

    """A class that holds the min-max values of the tree."""
    def __init__(self, known_bounds: tuple = (-1, 1)):
        self.minimum = known_bounds[0]
        self.maximum = known_bounds[1]

    def update(self, value: float):
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(root: Node, network: Network):
    min_max_stats = MinMaxStats(config.known_bounds)

    num_simulations = 100
    for _ in range(num_simulations):
        history = []
        node = root
        search_path = [node]

        # choose node based on score if it already exists in tree
        while node.expanded():
            action_idx, node = select_child(node, min_max_stats)
            history.append(Action(action_idx, node.player))
            search_path.append(node)

        # go until a unexpanded node, expand by using recurrent inference then backup
        parent = search_path[-2]
        last_action = history[-1]

        encoded_action = last_action.encode()
        
        with torch.no_grad():
            network_output = network.recurrent_inference(parent.hidden_state, torch.from_numpy(encoded_action))

        expand_node(node, [i for i in range(config.action_space_size)], network_output)

        backpropagate(search_path, network_output[0], history[-1].player, config.discount, min_max_stats)


def select_action(node: Node, network: Network):
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]

    t = 0.25  # Play according to the max.

    action = softmax_sample(visit_counts, t)
    return action


# Select the child with the highest UCB score.
def select_child(node: Node, min_max_stats: MinMaxStats):
    _, action_idx, child = max(
        (ucb_score(node, child, min_max_stats), action,
        child) for action, child in node.children.items())
    return action_idx, child


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
    # print(network_output[1].size())
    # print(actions)
    policy = {a: network_output[1][0, a].item() for a in actions}
    policy_sum = sum(policy.values())
    next_player = Player.WHITE if node.player is Player.BLACK else Player.BLACK
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum, next_player)


# At the end of a simulation, we propagate the evaluation all the way up to the
# tree of the root.
def backpropagate(search_path: List[Node], value: float, to_play: int, discount: float, min_max_stats: MinMaxStats):
    for node in search_path:
        node.value_sum += value if node.player == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


def softmax_sample(distribution, temperature: float):
    visits = [i[0] for i in distribution]
    actions = [i[1] for i in distribution]
    if temperature == 0:
        return actions[visits.index(max(visits))]
    elif temperature == 1:
        visits_sum = sum(visits)
        visits_prob = [i/visits_sum for i in visits]
        return np.random.choice(actions, p=visits_prob)
    elif temperature > 0 and temperature < 1:
        visits = [visit**(1/temperature) for visit in visits]
        visits_sum = sum(visits)
        visits_prob = [i/visits_sum for i in visits]
        return np.random.choice(actions, p=visits_prob)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    base_network = Network()
    base_network.load_state_dict(torch.load('./models/1000model.pth', map_location='cpu'))

    target_network = Network()
    target_network.load_state_dict(torch.load('./models/100000model.pth', map_location='cpu'))
    
    test(base_network, target_network)

    

    