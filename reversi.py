
"""useful functions for reversi"""
import math
import numpy as np

# check if a direction of a position is avaliable
def check_direction(board, row, column, mark, row_direc, col_direc):
    next_row = row + row_direc
    next_column = column + col_direc

    while board[next_row][next_column] == -mark:

        next_row += row_direc
        next_column += col_direc
        if next_row<0 or next_row>7 or next_column<0 or next_column>7:
            return False
          
    if board[next_row][next_column] == mark:
        return True
    
    return False
    
# check if this position is avaliable
def check_position(board, row, column, mark):

    available_direc = []
    if row+1<=7 and board[row+1][column]==-mark and check_direction(board, row, column, mark, 1, 0):
        available_direc.append((1, 0))

    if row-1>=0 and column+1<=7 and board[row-1][column+1]==-mark and check_direction(board, row, column, mark, -1, 1):
        available_direc.append((-1, 1))

    if column+1<=7 and board[row][column+1]==-mark and check_direction(board, row, column, mark, 0, 1):
        available_direc.append((0, 1))

    if row+1<=7 and column+1<=7 and board[row+1][column+1]==-mark and check_direction(board, row, column, mark, 1, 1):
        available_direc.append((1, 1))

    if row-1>=0 and board[row-1][column]==-mark and check_direction(board, row, column, mark, -1, 0):
        available_direc.append((-1, 0))

    if row+1<=7 and column-1>=0 and board[row+1][column-1]==-mark and check_direction(board, row, column, mark, 1, -1):
        available_direc.append((1, -1))

    if column-1>=0 and board[row][column-1]==-mark and check_direction(board, row, column, mark, 0, -1):
        available_direc.append((0, -1))

    if row-1>=0 and column-1>=0 and board[row-1][column-1]==-mark and check_direction(board, row, column, mark, -1, -1):
        available_direc.append((-1, -1))

    return available_direc

# 
def change_direction(board, row, column, row_direc, col_direc):
    non_curr = -board[row][column]
    next_row = row + row_direc
    next_column = column + col_direc
    count = 0


    while board[next_row][next_column] == non_curr:
        board[next_row][next_column] = -non_curr
        count += 1
        next_row += row_direc
        next_column += col_direc

        if board[next_row][next_column] == -non_curr:
            return count
    

def set_position(board, row, column, mark):
    count = 0
    available_direc = check_position(board, row, column, mark)
    if not available_direc:
        raise RuntimeError("set_position: not vaild position")
    
    board[row][column] = mark

    for row_direc, col_direc in available_direc:
        count += change_direction(board, row, column, row_direc, col_direc)

    return count


def available_pos(board, current):
    possible_pos = []
    positions = [(row, column) for row in range(8) for column in range(8) if board[row][column] == 0]
    
    for row, column in positions:
        if check_position(board, row, column, current):
            possible_pos.append((row, column))

    return possible_pos

def index_to_char(index):
    if index == -1:
        return 'X'
    elif index == 0:
        return ' '
    elif index == 1:
        return 'O'

def print_board(board):
    char_board = np.vectorize(index_to_char)(board)

    for i in range(8):

        print("     -----------------------------------------------------------------")
        print("     |       |       |       |       |       |       |       |       |")
        
        print("     ", end='')
        for j in range(8):
            print("|   " + char_board[i][j], end='   ')
        
        print('|')
        print("     |       |       |       |       |       |       |       |       |")
    
    print("     -----------------------------------------------------------------\n\n")

def init_board():
    board = np.zeros([8,8], dtype=np.float32)
    board[3][3] = 1
    board[4][4] = 1
    board[3][4] = -1
    board[4][3] = -1
    return board