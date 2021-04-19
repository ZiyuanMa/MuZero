
""" useful classes for Muzero """
from enum import Enum
import time
from reversi import *
import config
import random
import numpy as np
import ray
from dataclasses import dataclass
from typing import List




class ActionHistory:
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



@ray.remote(num_cpus=1)
class ReplayBuffer:

    def __init__(self):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer_size = 0
        self.buffer = []
        self.counter = 0

    def save_game(self, game):
        # with self.lock:
        if len(self.buffer) == self.window_size:
            del self.buffer[0]
            self.buffer_size -= 1
        self.buffer.append(game)
        self.buffer_size += 1
        self.counter += len(game.action_history)

    def ready(self):
        if self.buffer_size < self.batch_size:
            return False
        else:
            return True

    
    def sample_batch(self, num_unroll_steps: int = config.num_unroll_steps):
        # while self.buffer_size < self.batch_size:
        #     print('waiting')
        #     time.sleep(1)
        game_idxes = random.sample(range(self.buffer_size), self.batch_size)
        games = [self.buffer[idx] for idx in game_idxes]
        game_pos = [(g, self.sample_position(g)) for g in games]
        # batch = [(g.make_image(i), g.history[i:i + num_unroll_steps],
        #         g.make_target(i, num_unroll_steps, td_steps))
        #         for (g, i) in game_pos]

        init_state = []
        actions = []
        target_values = []
        target_policies = []
        for g, i in game_pos:
            init_state.append(g.make_image(i))
            actions.append(np.array([ g.encode() for g in g.action_history[i:i + num_unroll_steps]]))
            target_value, target_policy = g.make_target(i, num_unroll_steps)
            target_values.append(target_value)
            target_policies.append(np.array(target_policy))
        
        init_state = np.stack(init_state, axis=0)
        actions = np.stack(actions, axis=0).swapaxes(0, 1)
        target_values = np.stack(target_values, axis=0).swapaxes(0, 1)
        target_policies = np.stack(target_policies, axis=0).swapaxes(0, 1)

        return init_state, actions, target_values, target_policies

    def sample_position(self, game) -> int:
        
        # Sample position from game either uniformly or according to some priority.
        return random.randint(0, len(game.action_history)-config.num_unroll_steps)
    
    def count(self):
        temp = self.counter
        self.counter = 0
        return temp


MAXIMUM_FLOAT_VALUE = float('inf')

class MinMaxStats:

    """A class that holds the min-max values of the tree."""
    def __init__(self, known_bounds: tuple):
        self.minimum = known_bounds[0] if known_bounds else MAXIMUM_FLOAT_VALUE
        self.maximum = known_bounds[1] if known_bounds else -MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
    


