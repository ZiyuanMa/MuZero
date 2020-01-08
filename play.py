
import time
from reversi import *
from MCTS import MCT_search
import numpy as np

# board = np.zeros([8,8])

# board[3][3] = 1
# board[4][4] = 1
# board[3][4] = -1
# board[4][3] = -1
# curr = 1

# while len(available_pos(board, curr)) != 0:
#     MCT_search(board, curr)
#     print_board(board)
#     curr = -curr
#     time.sleep(0.5)

# l = [1,2,3]
# del l[0]
# print(l[0])