

def check_direction(board, position, row_direc, col_direc):
    non_curr = board[position] * -1
    row = position/8 + row_direc
    column = position%8 + col_direc

    if row<0 or row>7 or column<0 or column>7:
        return False
    
    if board[row*8+column] != non_curr:
        return False

    while board[row*8+column] == non_curr:

        row += row_direc
        column += col_direc
        if row<0 or row>7 or column<0 or column>7:
            return False
          
    if board[row*8+column] == board[position]:
        return True
    
    return False


def check_position(board, position):

    if check_direction(board, position, -1, 0):
        return True
    
    if check_direction(board, position, -1, 1):
        return True

    if check_direction(board, position, 0, 1):
        return True

    if check_direction(board, position, 1, 1):
        return True

    if check_direction(board, position, 1, 0):
        return True

    if check_direction(board, position, 1, -1):
        return True

    if check_direction(board, position, 0, -1):
        return True

    if check_direction(board, position, -1, -1):
        return True

    return False


def change_direction(board, position, row_direc, col_direc):
    non_curr = board[position] * -1
    row = position/8 + row_direc
    column = position%8 + col_direc
    count = 0

    if row<0 or row>7 or column<0 or column>7:
        return 0
    
    if board[row*8+column] != non_curr:
        return 0

    while board[row*8+column] == non_curr:
        row += row_direc
        column += col_direc
        if row<0 or row>7 or column<0 or column>7:
            return 0
        
    if board[row*8+column] == board[position]:
        row -= row_direc
        column -= col_direc
        while row*8+column != position:
            board[row*8+column] = board[position]
            row -= row_direc
            column -= col_direc
            count += 1
        return count

    return 0

def set_position(board, position, mark):
    count = 0
    board[position] = mark
    if check_position(board, position) == False:
        raise RuntimeError("set_position: not vaild position")
 
    count += change_direction(board, position, -1, 0)

    count += change_direction(board, position, -1, 1)

    count += change_direction(board, position, 0, 1)

    count += change_direction(board, position, 1, 1)

    count += change_direction(board, position, 1, 0)

    count += change_direction(board, position, 1, -1)

    count += change_direction(board, position, 0, -1)

    count += change_direction(board, position, -1, -1)

    return count


def get_pos_position(board, current):
    possible_pos = []

    for i in range(64):
        if board[i] == 0:
            board[i] == current
            if check_position(board, i):
                possible_pos.append(i)
            board[i] = 0

    return possible_pos

def index_to_char(index):
    if index == -1:
        return 'X'
    elif index == 0:
        return ' '
    elif index == 1:
        return 'O'

def print(board):

    print("\n\n        A      B      C      D      E      F      G      H\n")
    
    char_board = list(map(lambda x: index_to_char(x), board))
    for i in range(8):

        print("     ---------------------------------------------------------")
        print("     |      |      |      |      |      |      |      |      |")

        for j in range(8):
            print("|  " + char_board[i*8+j], end=' ')
        
        print('|')
        print("     |      |      |      |      |      |      |      |      |")
    
    print("     ---------------------------------------------------------\n\n")

