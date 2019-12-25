
from reversi import available_pos, set_position
import random
import numpy as np
import multiprocessing as mp


def self_play(board, mark):

    temp_board = np.copy(board)

    current_mark = -mark
    while True:

        positions = available_pos(temp_board, current_mark)
        if len(positions)==0:
            current_mark = -current_mark
            positions = available_pos(temp_board, current_mark)
            
        if len(positions)==0:
            break
        
        row, column = random.choice(positions)
        set_position(temp_board, row, column, current_mark)
        current_mark = -current_mark 

    # check score
    program_score = np.count_nonzero(temp_board==mark)
    player_score = np.count_nonzero(temp_board==-mark)

    if program_score > player_score:
        return 1
    elif program_score < player_score:
        return -1
    else:
        return 0


def MCT_search(board, mark):

    positions = available_pos(board, mark)

    #p = mp.Pool(4)
    scores = []

    for row, column in positions:
        temp_board = np.copy(board)
        set_position(temp_board, row, column, mark)
        results = []

        for _ in range(500):
            # re = p.apply_async(self_play, args=(temp_board,mark))
            # results.append(re.get())
            results.append(self_play(temp_board,mark))

        # p.close()
        # p.join()

        scores.append(sum(results))


    index = scores.index(max(scores))
    row, column = positions[index]
    set_position(board, row, column, mark)


