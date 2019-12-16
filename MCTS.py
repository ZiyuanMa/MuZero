
from reversi import *
import random
import multiprocessing
pool_num = multiprocessing.cpu_count()
g_mark = None

def random_search(board):
    score = 0
    global g_mark
    for _ in range(1000):
        temp_board = board.copy()

        current_mark = -g_mark
        while True:

            positions = available_pos(temp_board, current_mark)
            if len(positions)==0:
                current_mark = -current_mark
                positions = available_pos(temp_board, current_mark)
            
            if len(positions)==0:
                break
            else:
                position = random.choice(positions)
                set_position(temp_board, position, current_mark)
                current_mark = -current_mark 

    # check score
        program_score = len(list(filter(lambda x: x==g_mark, temp_board)))
        player_score = len(list(filter(lambda x: x==-g_mark, temp_board)))

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

    for i in range(len(positions)):
        temp_board = board.copy()
        set_position(temp_board, positions[i], mark)
        temp_boards.append(temp_board)
    

    with multiprocessing.Pool(pool_num) as p:
        scores = p.map(random_search, temp_boards)


    index = scores.index(max(scores))

    set_position(board, positions[index], mark)


