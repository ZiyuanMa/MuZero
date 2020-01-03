import numpy as np
import random

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
        self.init_layer = 0
        self.aval_num = 1
        #print(self.dict_list)

    def __getitem__(self, key):
        num = np.count_nonzero(key)-4

        bytes_board = key.tobytes()
        if bytes_board in self.dict_list[num]:
            return self.dict_list[num][bytes_board]
        else:
            raise RuntimeError("this board does not exist")

    def __setitem__(self, key, value):
 
        num = np.count_nonzero(key)-4
        bytes_board = key.tobytes()

        self.dict_list[num][bytes_board] = value

    def get_init_board(self):
        bytes_board = random.choice(list(self.dict_list[self.init_layer].keys()))
        while self.dict_list[self.init_layer][bytes_board][1] >= 100:
            bytes_board = random.choice(list(self.dict_list[self.init_layer].keys()))

        if self.dict_list[self.init_layer][bytes_board][1] == 99:
            self.aval_num -= 1
            if self.aval_num == 0:
                self.init_layer += 1
                self.aval_num = len([key for key, value in  self.dict_list[self.init_layer].items() if value[1] < 100])

                if self.aval_num == 0:
                    raise RuntimeError('no avaliable board')

        board = np.frombuffer(bytes_board, dtype='double').reshape(8,8)
        if np.count_nonzero(board) % 2 == 0:
            curr = 1
        else:
            curr = -1
        return board, curr
    
    def meet(self, board):

        num = np.count_nonzero(board)-4
        bytes_board = board.tobytes()
        if bytes_board not in self.dict_list[num]:
            self.dict_list[num][bytes_board] = [0, 1]
        else:
            print('1')
            self.dict_list[num][bytes_board][1] += 1

    def __len__(self):
        length = 0
        for d in self.dict_list:
            length += len(d)
        return length

    def to_filtered_list(self):
        l = []
        for d in self.dict_list:
            for key, value in d.items():
                if value[1] > 1:
                    l.append((np.frombuffer(key, dtype='double').reshape(8,8), value[0]/value[1]))
        print(l)
        return l

    def __str__(self):
        string = ''
        for i, d in enumerate(self.dict_list):
            if len(d) != 0:
                string += 'layer ' + str(i+1) + ':\n'
                for value in d:
                    string += '\t' + str(value[0]) + '  ' + str(value[0]) + '\n'
        return string

    def __iter__(self):
        self.layer_list = list(self.dict_list[0].keys())
        self.layer_ptr = 0
        self.ptr = -1
        self.layer_len = 1
        return self

    def __next__(self):
        self.ptr += 1
        if self.ptr == self.layer_len:
            if self.layer_ptr == 60:
                raise StopIteration()
            self.layer_ptr += 1
            self.layer_list = list(self.dict_list[self.layer_ptr].keys())
            self.layer_len = len(self.dict_list[self.layer_ptr])
            
            self.ptr = 0
        if self.layer_len == 0:
            raise StopIteration()
        else:
            return self.layer_list[self.ptr]
        
if __name__ == '__main__':
    m = container()
    