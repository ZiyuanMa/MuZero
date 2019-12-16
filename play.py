
import time
from reversi import *
from MCTS import MCT_search

board = [0 for _ in range(64)]

board[27] = 1
board[36] = 1
board[28] = -1
board[35] = -1
curr = 1

while len(available_pos(board, curr)) != 0:
    MCT_search(board, curr)
    print_board(board)
    curr = -curr
    time.sleep(0.5)