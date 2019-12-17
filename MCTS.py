
from reversi import *
import random
import numpy as np
import multiprocessing
pool_num = multiprocessing.cpu_count()
g_mark = None

def random_search(board):
    score = 0
    global g_mark
    for _ in range(1000):
        temp_board = np.copy(board)

        current_mark = -g_mark
        while True:

            positions = available_pos(temp_board, current_mark)
            if len(positions)==0:
                current_mark = -current_mark
                positions = available_pos(temp_board, current_mark)
            
            if len(positions)==0:
                break
            else:
                row, column = random.choice(positions)
                set_position(temp_board, row, column, current_mark)
                current_mark = -current_mark 

    # check score
        program_score = np.count_nonzero(temp_board==g_mark)
        player_score = np.count_nonzero(temp_board==-g_mark)

        if program_score > player_score:
            score += 1
        elif program_score < player_score:
            score -= 1

    return score


def MCT_search(board, mark):

    positions = available_pos(board, mark)
    global g_mark
    g_mark = mark

    temp_boards = []

    for row, column in positions:
        temp_board = np.copy(board)
        set_position(temp_board, row, column, mark)
        temp_boards.append(temp_board)
    

    with multiprocessing.Pool(pool_num) as p:
        scores = p.map(random_search, temp_boards)


    index = scores.index(max(scores))
    row, column = positions[index]
    set_position(board, row, column, mark)


