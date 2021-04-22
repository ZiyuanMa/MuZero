
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
    player: Player
    position: Optional[Tuple[int, int]] = field(init=False)

    def __post_init__(self):
        assert self.index >= 0 and self.index <= 64
        if self.index < 64:
            self.position = divmod(self.index, 8)
        else:
            self.position = None
    
    def encode(self):
        encoded_action = np.zeros((2, 8, 8), dtype=np.bool)
        layer = 0 if self.player is Player.WHITE else 1
        encoded_action[layer, self.position] = 1
        return encoded_action


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

        self.legal_actions = [Action(2*8+3, Player.BLACK), Action(3*8+2, Player.BLACK), Action(5*8+4, Player.BLACK), Action(4*8+5, Player.BLACK)]

        self.player = Player.BLACK

        self.steps = 0

        self.done = False

    # def reset(self):

    #     self.board = np.zeros((8, 8), dtype=np.int8)
    #     self.board[3, 3] = 1
    #     self.board[4, 4] = 1
    #     self.board[3, 4] = -1
    #     self.board[4, 3] = -1
        
    #     # position around the chess pieces
    #     self.possible_pos = {(2, 2), (2, 3), (2, 4), (2, 5), (3, 2), (3, 5),
    #                         (4, 2), (4, 5), (5, 2), (5, 3), (5, 4), (5, 5)}

    #     # self.legal_actions = {(2, 3), (3, 2), (5, 4), (4, 5)}

    #     self.legal_actions = [Action(2*8+3, Player.BLACK), Action(3*8+2, Player.BLACK), Action(5*8+4, Player.BLACK), Action(4*8+5, Player.BLACK)]

    #     self.player = Player.BLACK

    #     self.done = False


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

    def update_legal_actions(self):
        self.legal_actions = []
        for position in self.possible_pos:
            if self.check_position(position):
                self.legal_actions.append(Action(position[0]*8+position[1], self.player))
        
        if not self.legal_actions:
            self.legal_actions.append(Action(64, self.player))


    
    def step(self, action: Action, player: Player = None):
        assert action.player is self.player

        self.steps += 1

        if action.index == 64:

            self.player = Player.WHITE if self.player is Player.BLACK else Player.BLACK
            self.update_legal_actions()
            return self.observe(), 0, False

        position = np.array(action.position)

        assert action.position in self.possible_pos, '{} out of {} at {}'.format(action.position, self.possible_pos, self.steps)
        assert action in self.legal_actions, '{} out of {}'.format(action, self.legal_actions)

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

            if self.steps == 60:
                self.done = True
                return self.observe(), self.check_win(), True

            self.update_possible_position(action.position)

            self.player = Player.WHITE if self.player is Player.BLACK else Player.BLACK

            self.update_legal_actions()

            return self.observe(), 0, False
    
    def observe(self):

        obs = np.stack((self.board==Player.WHITE.value, self.board==Player.BLACK.value), axis=0).astype(np.bool)

        return obs
    
    def update_possible_position(self, position):
        self.possible_pos.remove(position)

        for direction in DIRECTIONS:
            around_position = position + direction
            if np.all(around_position>=0) and np.all(around_position<=7) and self.board[tuple(around_position)] == 0:
                self.possible_pos.add(tuple(around_position))

    
    def check_win(self):
        white_score = np.sum(self.board==1)
        black_score = np.sum(self.board==-1)
        if white_score == black_score:
            return 0
        elif white_score > black_score:
            return self.player.value
        else:
            return -self.player.value


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
        self.player = player
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
        # self.player_history = [Player.BLACK]
        self.action_history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = 65 # actually only 61 actions are legal
        self.discount = 1

    def terminal(self) -> bool:
        return self.environment.done

    def legal_actions(self) -> List[Action]:

        return self.environment.legal_actions

    def apply(self, action: Action):
        # if self.environment.player is not None:
        #     self.player_history.append(self.environment.player)
        # print(action)
        state, reward, _ = self.environment.step(action)
        
        self.rewards.append(reward)
        self.state_history.append(state)

        # encoded_action = np.zeros((2, 8, 8), dtype=np.bool)
        # if action is Player.WHITE:
        #     encoded_action[0, action.position] = 1
        # else:
        #     encoded_action[1, action.position] = 1

        self.action_history.append(action)

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

        for i in range(min(state_index, config.state_history_len)):
            image[(config.state_history_len-1-i)*2:(config.state_history_len-i)*2] = self.state_history[state_index-i]

        if state_index % 2 == 0:
            # black tern
            image[-1] = 1

        return image.astype(np.float32)

    def make_target(self, state_index: int, num_unroll_steps: int):
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
                value = self.rewards[-1] if self.action_history[current_index].player is self.action_history[-1].player else -self.rewards[-1]
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

