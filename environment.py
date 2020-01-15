
from reversi import *
from dataclasses import dataclass

class Action(object):

  def __init__(self, index: int):
    self.index = index

  def __hash__(self):
    return self.index

  def __eq__(self, other):
    return self.index == other.index

  def __gt__(self, other):
    return self.index > other.index


class Action(dataclass):
    row: int
    column: int
    tern: int


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
        self.actions = [Action(row, column, self.tern) for row, column in available_pos(self.board, self.tern)]
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
    def update(self, board):
        self.board = numpy.copy(board)
        self.turn = self.turn_n()
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    def turn_n(self):
        return np.count_nonzero(self.board==0)

    def player_turn(self):
        if self.turn == 1:
            return Player.white
        else:
            return Player.black
    # def get_action(self):


    def step(self, action: Action):
        if self.tern != action.tern:
            raise RuntimeError('tern mismatch')
        self.board[action.row][action.column] = action.tern

        self.tern = -self.tern

        self.actions = [Action(row, column, self.tern) for row, column in available_pos(self.board, self.tern)]
        if not self.actions:
            self.tern = -self.tern
            self.actions = [Action(row, column, self.tern) for row, column in available_pos(self.board, self.tern)]
            if not self.actions:
                self.tern = 0
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

    def black_and_white_plane(self):
        white_board = self.board==1
        black_board = self.board==-1
        return white_board, black_board

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

    def make_image(self, state_index: int):
        # Game specific feature planes.    
        o = Environment().reset()

        for current_index in range(0, state_index):
            o.step(self.history[current_index])

        black_ary, white_ary = o.black_and_white_plane()
        state = [black_ary, white_ary] if o.player_turn() == Player.black else [white_ary, black_ary]
        return numpy.array(state)

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int,
                  to_play: Player):
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

    def to_play(self) -> Player:
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
             g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
            for (g, i) in game_pos]

    def sample_game(self) -> Game:
        # Sample game from buffer either uniformly or according to some priority.
        return numpy.random.choice(self.buffer)

    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        return numpy.random.choice(game.history)