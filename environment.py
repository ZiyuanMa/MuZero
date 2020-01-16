from reversi import *
from config import *
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
# class Action(object):

#   def __init__(self, index: int):
#     self.index = index

#   def __hash__(self):
#     return self.index

#   def __eq__(self, other):
#     return self.index == other.index

#   def __gt__(self, other):
#     return self.index > other.index


@dataclass(order=True, unsafe_hash=True)
class Action():
    index: int

    def get_coord(self):
        return self.index // 8, self.index % 8

    def encode(self):
        # encode to netword input 
        board = np.zeros((8, 8))
        board[self.index//8][self.index%8] = 1
        return board

class Node:

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class ActionHistory:
    # def __init__(self, history: List[Action], action_space_size: int):
    #     self.history = list(history)
    #     self.action_space_size = action_space_size
    """Simple history container used inside the search.

    Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> int:
        if len(self.history) % 2 == 0:
            return 1
        else:
            return -1


class Environment:
    # The environment MuZero is interacting with
    def __init__(self):
        self.board = np.zeros([8,8])
        self.board[3][3] = 1
        self.board[4][4] = 1
        self.board[3][4] = -1
        self.board[4][3] = -1
        self.turn = 1
        self.done = False
        self.winner = None  # type: Winner
        self.resigned = False
        self.actions = [Action(row*8+column) for row, column in available_pos(self.board, self.turn)]
    def reset(self):
        self.board = np.zeros([8,8])
        self.board[3][3] = 1
        self.board[4][4] = 1
        self.board[3][4] = -1
        self.board[4][3] = -1
        self.turn = 1
        self.done = False
        self.winner = None
        self.resigned = False
        return self
    # def update(self, board):
    #     self.board = np.copy(board)
    #     self.turn = self.turn_n()
    #     self.done = False
    #     self.winner = None
    #     self.resigned = False
    #     return self


    def turn_n(self):
        return np.count_nonzero(self.board!=0)

    def player_turn(self):
        return self.turn
    # def get_action(self):


    def step(self, action: Action):
        row, column = action.get_coord()
        self.board[row][column] = self.turn

        self.turn = -self.turn

        self.actions = [Action(row*8+column) for row, column in available_pos(self.board, self.turn)]
        if not self.actions:
            self.turn = -self.turn
            self.actions = [Action(row*8+column) for row, column in available_pos(self.board, self.turn)]
            if not self.actions:
                self.turn = 0
                self.done = True

        reward = 0
        if self.done:
            white_score = np.count_nonzero(self.board==1)
            black_score = np.count_nonzero(self.board==-1)
            if white_score > black_score:
                reward = 1
            elif white_score < black_score:
                reward = -1
        return reward

    def legal_actions(self):
        return self.actions.copy()

    def get_board(self):
        
        return np.copy(self.board)

class Game:
    """A single episode of interaction with the environment."""

    def __init__(self, action_space_size: int, discount: float):
        self.environment = Environment()  # Game specific environment.
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount

    def terminal(self) -> bool:
        return self.environment.done

    def legal_actions(self) -> List[Action]:
        return self.environment.legal_actions()

    def apply(self, action: Action):
        reward = self.environment.step(action)
        reward = reward if self.environment.turn % 2 != 0 and reward == 1 else -reward
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def make_image(self, state_index: int) -> np.array:
        # convert state to Representation network input
        # layer 1: board with white stone
        # layer 2: board with black stone
        # layer 3: binary board for it is white's turn
        # layer 4: binary board for it is black's turn
        o = Environment().reset()

        for current_index in range(0, state_index):
            o.step(self.history[current_index])

        board = o.get_board()
        white_board = np.expand_dims(board==1, axis=0)
        black_board = np.expand_dims(board==-1, axis=0)
        turn_board = np.zeros((2,board.shape[0], board.shape[1]))
        if o.player_turn == 1:
            turn_board[0,:,:] = 1
        elif o.player_turn == -1:
            turn_board[1,:,:] = 1
        
        return np.concatenate((white_board, black_board, turn_board))

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int):
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount**td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i  # pytype: disable=unsupported-operands

            if current_index < len(self.root_values):
                targets.append((value, self.rewards[current_index],
                        self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, 0, []))
        return targets

    def to_play(self) -> int:
        return self.environment.player_turn

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)

class ReplayBuffer:

    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            del self.buffer[0]
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(i), g.history[i:i + num_unroll_steps],
            g.make_target(i, num_unroll_steps, td_steps))
            for (g, i) in game_pos]

    def sample_game(self) -> Game:
        # Sample game from buffer either uniformly or according to some priority.
        return np.random.choice(self.buffer)

    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        print(len(game.history))
        return np.random.choice(range(len(game.history)))

