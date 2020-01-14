
class Action(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]


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

    def step(self, action):
        row = action // 8
        column = action % 8
        self.board[row][column] == 