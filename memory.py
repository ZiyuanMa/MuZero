import config
import numpy as np
import random
import math

class Board:
    def __init__(self, round_info: list):
        round_info = round_info.copy()
        round_info[0] = round_info[0]/round_info[1]
        round_info.append(config.memory_size)
        self.info_list = [round_info]
        self.visit_times = round_info[1]

    def update(self, round_info=None):

        if self.info_list[0][2] == 1:
            self.visit_times -= self.info_list[0][1]
            del self.info_list[0]

        for info in self.info_list:
            info[2] -= 1

        if round_info:
            self.info_list.append(round_info+[config.memory_size])
            self.visit_times += round_info[1]


    def __len__(self):
        return len(self.info_list)

    def get_value(self):
        size = len(self.info_list)
        value = self.info_list[0][0]
        if size == 1:
            return value

        else:
            for info in self.info_list[1:]:
                value = value * (1-config.update_rate) + info[0] * config.update_rate
            return value


class Memory:
    def __init__(self):
        self.storage = dict()
        self.buffer = dict()
        #print(self.storage)

        #print(len(self.storage[0]))
    
    def meet(self, board, next):

        bytes_board = board.tobytes()
        if (bytes_board, next) not in self.buffer:

            self.buffer[(bytes_board, next)] = [0, 1]

        else:

            self.buffer[(bytes_board, next)][1] += 1


    def __len__(self):

        return len(self.storage)

    def to_list(self, min=2):
        l = []
        for key, value in self.storage.items():
            if value.visit_times >= config.min_visit_times or key[1]==0:
                l.append(np.concatenate((np.frombuffer(key[0], dtype='int8').reshape(8,8), key[1]*np.ones((8, 8))), value[0]/value[1]))
            
        return l

    # def __str__(self):
    #     string = ''
    #     for i, d in enumerate(self.storage):
    #         if len(d) != 0 and i < 4:
    #             string += 'layer ' + str(i+1) + ':\n'
    #             for _, value in d.items():
    #                 #board = np.frombuffer(key, dtype='double').reshape(8,8)
    #                 #string += board.__str__()
    #                 string += '\t' + str(value[0]) + '  ' + str(value[1]) + '\n'

    #     return string


    def round_result(self, round_boards, result):

        for board, next in round_boards:
            bytes_board = board.tobytes()
            self.buffer[(bytes_board, next)][0] += result


    def buffer_to_storage(self):
        # update storage
        for key, value in self.storage.items():
            if key in self.buffer:
                value.update(self.buffer[key])
                del self.buffer[key]
            else:
                value.update()
                if len(value) == 0:
                    del self.storage[key]
        
        # write new board in buffer to storage
        for key, value in self.buffer.items():
            self.storage[key] = Board(value)
        
        self.buffer.clear()


if __name__ == '__main__':
    m = Memory()
    