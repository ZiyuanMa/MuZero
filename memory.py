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

    def update(self, round_info=None):

        if self.info_list[0][2] == 0:
            del self.info_list[0]

        for info in self.info_list:
            info[2] -= 1

        if round_info:
            round_info = round_info.copy()
            round_info[0] = round_info[0]/round_info[1] 
            round_info.append(config.memory_size)
            self.info_list.append(round_info)

    def __len__(self):
        return len(self.info_list)

    def get_value(self):

        value = self.info_list[0][0]
        if len(self.info_list) == 1:
            return value

        else:
            for info in self.info_list[1:]:
                value = value * (1-config.update_rate) + info[0] * config.update_rate
            return value

    def get_visit_times(self):
        # visit_times = self.info_list[0][1]
        # if len(self.info_list) == 1:
        #     return visit_times

        # else:
        #     for info in self.info_list[1:]:
        #         visit_times = visit_times * (1-config.update_rate) + info[1] * config.update_rate
        #     return round(visit_times)
        
        visit_times = 0
        for info in self.info_list:
            visit_times += info[1]
        return visit_times

class Memory:
    def __init__(self):
        self.storage = dict()

        #print(self.storage)

        #print(len(self.storage[0]))
    def __len__(self):

        return len(self.storage)

    def get_batch(self):
        sub_keys1 = []
        sub_keys2 = []
        for key, value in self.storage.items():
            if key[1]==0:
                sub_keys1.append(key)
            elif value.get_visit_times() >= config.min_visit_times:
                sub_keys2.append(key)

        print('avaliable: '+str(len(sub_keys1)+len(sub_keys2)))
        if len(sub_keys2) < config.batch_size*0.75:
            keys = random.sample(sub_keys1, config.batch_size-len(sub_keys2))
            keys.extend(sub_keys2)
        else:
            keys = random.sample(sub_keys1, config.batch_size*0.25)
            keys.extend(random.sample(sub_keys2, config.batch_size*0.75))

        return list(map(lambda key: (np.concatenate((np.frombuffer(key[0]).reshape(8,8), key[1]*np.ones([8, 8]))), self.storage[key].get_value()), keys))

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

    def buffer_to_storage(self, buffer: dict):
        # update storage
        for key, value in self.storage.items():
            if key in buffer:
                value.update(buffer[key].copy())
                del buffer[key]
            else:
                value.update()
                if len(value) == 0:
                    del self.storage[key]
        
        # write new board in buffer to storage
        for key, value in buffer.items():

            self.storage[key] = Board(value)
        


if __name__ == '__main__':
    m = Memory()
    