import numpy as np

class container:
    def __init__(self):
        self.dict_list = [dict()]*61
        # for _ in range(61):
        #     self.dict_list.append(dict())
        board = np.zeros([8,8], dtype='double')
        board[3][3] = 1
        board[4][4] = 1
        board[3][4] = -1
        board[4][3] = -1
        self.dict_list[0][board.tobytes()] = [0, 0]
        print(self.dict_list)
if __name__ == '__main__':
    m = container()
    