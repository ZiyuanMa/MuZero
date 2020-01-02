import numpy as np

class container(object):
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

    def __getitem__(self, key):
        num = np.nonzero(key)-4
        bytes_board = key.tobytes()
        if bytes_board in self.dict_list[num]:
            return self.dict_list[num][bytes_board]
        else:
            raise RuntimeError("this board does not exist")

    def __setitem__(self, key, value):
        num = np.nonzero(key)-4
        bytes_board = key.tobytes()

        self.dict_list[num][bytes_board] = value


    def __len__(self):
        length = 0
        for d in self.dict_list:
            length += len(d)
        return length


if __name__ == '__main__':
    m = container()
    