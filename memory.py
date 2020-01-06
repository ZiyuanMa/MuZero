import numpy as np
import random

class Memory:
    def __init__(self):
        self.dict_list = [dict() for _ in range(61)]

        board = np.zeros([8,8], dtype='int8')
        board[3][3] = 1
        board[4][4] = 1
        board[3][4] = -1
        board[4][3] = -1
        self.dict_list[0][board.tobytes()] = [0, 0, 1]
        self.init_layer = 0
        self.init_list = [(board.tobytes(), 1)]
        #print(self.dict_list)

        #print(len(self.dict_list[0]))

    def get_init_board(self):
        if len(self.init_list) == 0:
            raise RuntimeError('no board in init list')

        bytes_board, next = random.choice(self.init_list)

        if self.dict_list[self.init_layer][bytes_board][1] == 99:
            self.init_list.remove((bytes_board, next))
            if len(self.init_list) == 0:
                self.init_layer += 1
                self.init_list = [ (key, value[2]) for key, value in self.dict_list[self.init_layer].items() 
                                        if self.dict_list[self.init_layer][key][1] < 100 ]

                if len(self.init_list) == 0:
                    raise RuntimeError('no avaliable board')

        board = np.frombuffer(bytes_board, dtype='int8').reshape(8,8)
        #print(np.count_nonzero(board))

        return np.copy(board), next
    
    def meet(self, board, next):

        num = np.count_nonzero(board)-4
        bytes_board = board.tobytes()
        if bytes_board not in self.dict_list[num]:

            self.dict_list[num][bytes_board] = [0, 1, next]

        else:
            if self.dict_list[num][bytes_board][2] != next:
                raise RuntimeError('wrong next')
            self.dict_list[num][bytes_board][1] += 1


    def __len__(self):
        length = 0
        for d in self.dict_list:
            length += len(d)

        return length

    def to_list(self, min=2):
        l = []
        for d in self.dict_list:
            for key, value in d.items():
                if value[1] >= 2:
                    l.append((np.frombuffer(key, dtype='int8').reshape(8,8), value[0]/value[1]))

        return l

    def __str__(self):
        string = ''
        for i, d in enumerate(self.dict_list):
            if len(d) != 0 and i < 4:
                string += 'layer ' + str(i+1) + ':\n'
                for key, value in d.items():
                    #board = np.frombuffer(key, dtype='double').reshape(8,8)
                    #string += board.__str__()
                    string += '\t' + str(value[0]) + '  ' + str(value[1]) + '\n'

        return string

        
    def exist(self, board):
        num = np.count_nonzero(board)-4
        bytes_board = board.tobytes()
        if bytes_board in self.dict_list[num]:
            return True
        else:
            return  False

    def round_result(self, round_boards, result):

        for board in round_boards:
            num = np.count_nonzero(board)-4
            bytes_board = board.tobytes()
            self.dict_list[num][bytes_board][0] += result

            

    def get_value(self, board):
        num = np.count_nonzero(board)-4
        bytes_board = board.tobytes()
        if bytes_board not in self.dict_list[num]:
            raise RuntimeError('this board D.N.E.')
        
        return self.dict_list[num][bytes_board]

if __name__ == '__main__':
    m = Memory()
    