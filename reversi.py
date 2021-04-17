
"""useful functions for reversi"""
import math
from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import config


DIRECTIONS = (np.array([-1, -1]), np.array([-1, 0]), np.array([-1, 1]), np.array([0, -1]),
            np.array([0, 1]), np.array([1, -1]), np.array([1, 0]), np.array([1, 1]))

class Player(Enum):
    WHITE = 1
    BLACK = -1


@dataclass
class Action:
    index: int
    player: Player = None
    position: Optional[Tuple[int, int]] = field(init=False)

    def __post_init__(self):
        if self.index < 64:
            self.position = divmod(self.index, 8)
        else:
            self.position = None


class Environment:
    """The environment MuZero is interacting with."""
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.board[3, 3] = 1
        self.board[4, 4] = 1
        self.board[3, 4] = -1
        self.board[4, 3] = -1
        
        # position around the chess pieces
        self.possible_pos = {(2, 2), (2, 3), (2, 4), (2, 5), (3, 2), (3, 5),
                            (4, 2), (4, 5), (5, 2), (5, 3), (5, 4), (5,5)}

        self.legal_positions = {(1, 2), (2, 1), (5, 4), (4, 5)}

        self.player = Player.BLACK

        self.done = False

    def reset(self):

        self.board = np.zeros((8, 8), dtype=np.int8)
        self.board[3, 3] = 1
        self.board[4, 4] = 1
        self.board[3, 4] = -1
        self.board[4, 3] = -1
        
        # position around the chess pieces
        self.possible_pos = {(2, 2), (2, 3), (2, 4), (2, 5), (3, 2), (3, 5),
                            (4, 2), (4, 5), (5, 2), (5, 3), (5, 4), (5,5)}

        self.legal_positions = {(2, 3), (3, 2), (5, 4), (4, 5)}

        self.player = Player.BLACK

        self.done = False


    def check_direction(self, position, next_player: Player, direction):

        next_position = position + direction
        step = 0

        while np.all(next_position>=0) and np.all(next_position<=7) and self.board[tuple(next_position)] == -next_player.value:
            step += 1
            next_position += direction

        if step != 0 and np.all(next_position>=0) and np.all(next_position<=7) and self.board[tuple(next_position)] == next_player.value:
            return True
        else:
            return False


    def check_position(self, position):

        next_player = self.player
        position = np.array(position)

        for direction in DIRECTIONS:
            if self.check_direction(position, next_player, direction):
                # print('{}, {}, {}'.format(position, direction, next_mark))
                return True
        
        return False
    

    def get_legal_actions(self):

        legal_pos = []

        for position in self.possible_pos:
            if self.check_position(position):
                legal_pos.append(position)

        return legal_pos
    
    def step(self, action: Action, player: Player = None):
        if player: assert player is self.player

        if action.index == 64:
            assert not self.legal_positions
            self.player = Player.WHITE if self.player is Player.BLACK else Player.BLACK
            self.legal_positions = self.get_legal_actions()
            assert self.legal_positions
            return self.observe(), 0, False

        position = np.array(action.position)

        assert action.position in self.possible_pos

        opponent_player = Player.WHITE if self.player is Player.BLACK else Player.BLACK

        token = False
        for direction in DIRECTIONS:

            next_position = position + direction
            step = 0

            while np.all(next_position>=0) and np.all(next_position<=7) and self.board[tuple(next_position)]==opponent_player.value:
                next_position = next_position + direction
                step += 1

            if step != 0 and np.all(next_position>=0) and np.all(next_position<=7) and self.board[tuple(next_position)] == self.player.value:
                for _ in range(step):
                    next_position = next_position - direction
                    self.board[tuple(next_position)] = self.player.value
                    token = True

        if token == False:
            raise RuntimeError('nothing changed at {} for player {} in\n {}'.format(action.position, self.player.value, self.board))
        else:
        
            self.board[action.position] = self.player.value

            self.possible_pos.remove(action.position)

            for direction in DIRECTIONS:
                around_position = position + direction
                if np.all(around_position>=0) and np.all(around_position<=7) and self.board[tuple(around_position)] == 0 and tuple(around_position) not in self.possible_pos:
                    self.possible_pos.add(tuple(around_position))

            self.player = Player.WHITE if self.player is Player.BLACK else Player.BLACK

            self.legal_positions = self.get_legal_actions()
            if not self.legal_positions:
                self.player = Player.WHITE if self.player is Player.BLACK else Player.BLACK
                self.legal_positions = self.get_legal_actions()
                if not self.legal_positions:
                    self.done = True
                    return self.observe(), 1, True
                else:
                    self.player = Player.WHITE if self.player is Player.BLACK else Player.BLACK
                    self.legal_positions = self.get_legal_actions()
            
            self.done = False
            return self.observe(), 0, False
    
    def observe(self):

        obs = np.stack((self.board==Player.WHITE.value, self.board==Player.BLACK.value), axis=0).astype(np.bool)

        return obs

    def print_board(self):
        char_board = np.vectorize(index_to_char)(self.board)

        for i in range(8):

            print("     -----------------------------------------------------------------")
            print("     |       |       |       |       |       |       |       |       |")
            
            print("     ", end='')
            for j in range(8):
                print("|   " + char_board[i][j], end='   ')
            
            print('|')
            print("     |       |       |       |       |       |       |       |       |")
        
        print("     -----------------------------------------------------------------\n\n")

def index_to_char(index):
    if index == -1:
        return 'X'
    elif index == 0:
        return ' '
    elif index == 1:
        return 'O'


class Node:

    def __init__(self, prior: float, player: Player):
        self.visit_count = 0
        self.to_play = player
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


class Game:
    """A single episode of interaction with the environment."""

    def __init__(self):
        self.environment = Environment()  # Game specific environment.
        self.state_history = [self.environment.observe()]
        self.player_history = [Player.BLACK]
        self.action_history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = 65 # actually only 61 actions are legal
        self.discount = 1

    def terminal(self) -> bool:
        return self.environment.done

    def legal_actions(self) -> List[int]:
        positions = self.environment.get_legal_actions()
        if positions:
            actions = [ x*8+y for x, y in positions ]
        else:
            actions = [ 64 ]

        return actions

    def apply(self, action: Action):
        if self.environment.player is not None:
            self.player_history.append(self.environment.player)

        state, reward, done = self.environment.step(action)
        
        self.rewards.append(reward)
        self.state_history.append(state)

        encoded_action = np.zeros((2, 8, 8), dtype=np.bool)
        if action is Player.WHITE:
            encoded_action[0, action.position] = 1
        else:
            encoded_action[1, action.position] = 1
        self.action_history.append(encoded_action)

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (index for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def make_image(self, state_index: int) -> np.array:
        """ convert state to Representation network input """
        image = np.zeros(config.state_shape, dtype=np.bool)

        shift = 0
        if state_index < config.state_history_steps:
            shift =  config.state_history_steps-state_index
            for i in range(shift):
                image[i*4:i*4+2] = self.state_history[0]
        
        for i in range(shift, config.state_history_steps):
            image[i*4:i*4+2] = self.state_history[i-shift]

            image[i*4+2:(i+1)*4] = self.action_history[i-shift]
        
        image[-4:-2] = self.state_history[state_index]
        if self.player_history[state_index] is Player.WHITE:
            image[-2] = 1
        else:
            image[-1] = 1

        return image.astype(np.float32)

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int):
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        # targets = []
        target_value, target_policy = [],  []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            # always MCTS, no need to do that much in this case
            # bootstrap_index = current_index + td_steps
            # if bootstrap_index < len(self.root_values):
            #     value = self.root_values[bootstrap_index] * self.discount**td_steps
            # else:
            #     value = 0
            
            # l = self.rewards[current_index:bootstrap_index]
            # for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
            #     value += reward * self.discount**i

            if current_index < len(self.root_values):
                value = self.rewards[-1] if self.player_history[current_index] is self.player_history[-1] else -self.rewards[-1]
                # targets.append((value, self.child_visits[current_index]))
                target_value.append(value)
                target_policy.append(self.child_visits[current_index])
            else:
                # targets.append((0, [ 0 for _ in range(config.action_space_size)]))
                target_value.append(0)
                target_policy.append([ 0 for _ in range(config.action_space_size)])

        return target_value, target_policy

    def to_play(self) -> Player:
        return self.environment.player

