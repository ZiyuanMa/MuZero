from MCTS import MCT_search
import config
from reversi import available_pos, set_position
from memory import Memory
import random
from math import sqrt, log
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
import pickle
import os
torch.manual_seed(1261)
random.seed(1261)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
# torch.set_num_interop_threads(2)
# torch.set_num_threads(12)

@torch.no_grad()
def self_play(net, buffer):

    board = np.zeros([8,8])
    board[3][3] = 1
    board[4][4] = 1
    board[3][4] = -1
    board[4][3] = -1
    next = 1
    chess_piece = 4
    round_boards = []
    while True:
        bytes_board = board.tobytes()
        if (bytes_board, next) not in buffer:
            buffer[(bytes_board, next)] = [0, 1]
        else:
            tmp = buffer[(bytes_board, next)]
            tmp[1] += 1
            buffer[(bytes_board, next)] = tmp

        if next == 0:
            break
        else:
            positions = available_pos(board, next)
        

        # visit_times = []
        next_list = []
        next_boards = np.empty([len(positions), 2, 8, 8])
        for i, position in enumerate(positions):
            row, column = position
            temp_board = np.copy(board)
            set_position(temp_board, row, column, next)

            temp_next = -next
            temp_pos = available_pos(temp_board, temp_next)
            if len(temp_pos) == 0:
                temp_next = -temp_next
                temp_pos = available_pos(temp_board, temp_next)

                if len(temp_pos) == 0:
                    temp_next = 0


            next_list.append(temp_next)
            next_boards[i,0,:,:] = temp_board
            next_boards[i,1,:,:] = np.ones((8,8))*temp_next
            # if self.memory_pool.exist(temp_board):
            #     visit_times.append(self.memory_pool.get_value(temp_board)[1])
            # else:
            #     visit_times.append(0)

        net_input = torch.from_numpy(next_boards).float().view(-1,2,8,8)
        net_output = net(net_input)
        net_output = net_output.view(-1)* next
        # values = np.asarray(net_output)
        # values = values.tolist()
        if chess_piece < 40:

            prob = np.asarray(torch.softmax(net_output, dim=0))
            index = np.random.choice(range(len(prob)), p = prob)
            # sum_visit = sum(visit_times)+1
            # scores = [value + sqrt(log(sum_visit)/(visit+1)) for value, visit in zip(values, visit_times)]
            # index = scores.index(max(scores))
        else:
            _, index = torch.max(net_output, 0)
            index = index.item()
            
        set_position(board, positions[index][0], positions[index][1], next)
        chess_piece += 1

        next = next_list[index]

    white_score = np.count_nonzero(board==1)
    black_score = np.count_nonzero(board==-1)

    if white_score > black_score:

        for key in round_boards:
            tmp = buffer[key]
            tmp[0] += 1
            buffer[key] = tmp


    elif white_score < black_score:

        for key in round_boards:
            tmp = buffer[key]
            tmp[0] -= 1
            buffer[key] = tmp
def random_play(data_dict):
    board = np.zeros([8,8])
    board[3][3] = 1
    board[4][4] = 1
    board[3][4] = -1
    board[4][3] = -1
    next = 1
    round = 0
    while True:
        bytes_board = board.tobytes()
        positions = available_pos(board, next)
        if len(positions) == 0:
            next = -next
            positions = available_pos(board, next)
            if len(positions) == 0:
                break
        key = (bytes_board, next)
        if key not in data_dict:
            data_dict[key] = []

        row, column = random.choice(positions)
        set_position(board, row, column, next)
        round += 1
        next = -next

@torch.no_grad()
def model_against(white_net, black_net):
    board = np.zeros([8,8])
    board[3][3] = 1
    board[4][4] = 1
    board[3][4] = -1
    board[4][3] = -1
    next = 1

    while True:

        positions = available_pos(board, next)
        if len(positions) == 0:
            next = -next
            positions = available_pos(board, next)
        if len(positions) == 0:
            
            break

        next_list = []
        next_boards = np.empty([len(positions), 2, 8, 8])
        for i, position in enumerate(positions):
            row, column = position
            temp_board = np.copy(board)
            set_position(temp_board, row, column, next)

            temp_next = -next
            temp_pos = available_pos(temp_board, temp_next)
            if len(temp_pos) == 0:
                temp_next = -temp_next
                temp_pos = available_pos(temp_board, temp_next)

                if len(temp_pos) == 0:
                    temp_next = 0


            next_list.append(temp_next)
            next_boards[i,0,:,:] = temp_board
            next_boards[i,1,:,:] = np.ones((8,8))*temp_next

        net_input = torch.from_numpy(next_boards).float().view(-1,2,8,8)
        if next == 1:
            net_output = white_net(net_input)
        else:
            net_output = black_net(net_input)

        net_output = net_output.view(-1)* next

        _, index = torch.max(net_output, 0)
        index = index.item()
            
        set_position(board, positions[index][0], positions[index][1], next)

        next = next_list[index]

    white_score = np.count_nonzero(board==1)
    black_score = np.count_nonzero(board==-1)

    return white_score, black_score



def test_new_model(last_model_num, new_net):
    new_net.eval()
    checkpoint = torch.load('./model'+str(last_model_num)+'.pth')
    last_net = Network()
    last_net.load_state_dict(checkpoint['net'])
    last_net.eval()

    last_white, last_black = model_against(last_net, last_net)

    print('last model\nwhite score: %i, black score: %i\n'%(last_white, last_black))

    new_white, last_black = model_against(new_net, last_net)

    print('new model with white\nwhite score: %i, black score: %i'%(new_white, last_black))

    last_white, new_black = model_against(last_net, new_net)

    print('new model with black\nwhite score: %i, black score: %i'%(last_white, new_black))
    if new_white < last_white or new_black < last_black:
        return False
    else:
        return True



class Dataset(Dataset):

    def __init__(self, data):

        self.x_data = [board for board, _ in data]
        self.y_data = [torch.tensor([value], dtype=torch.float) for _, value in data]
        self.len = len(data)

    def __getitem__(self, index):

        return self.transform(self.x_data[index]), self.y_data[index]


    def __len__(self):
        return self.len

    def transform(self, narray):
        if random.choice([True, False]):
            
            return torch.from_numpy(np.rot90(narray, 2).copy()).float().view(2,8,8)
        else:
            return torch.from_numpy(narray).float().view(2,8,8)

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.LeakyReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.LeakyReLU(out)
        return out

class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # 8x8 input 6x6 output
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2,
                            out_channels=config.filter_num,
                            kernel_size=3,
                            stride=1,
                            padding=0),
            nn.BatchNorm2d(config.filter_num),
            nn.LeakyReLU()
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(config.filter_num, config.filter_num, 3, 1, 1),
            nn.BatchNorm2d(config.filter_num),
            nn.LeakyReLU(),
            nn.Conv2d(config.filter_num, config.filter_num, 3, 1, 1),
            nn.BatchNorm2d(config.filter_num)
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(config.filter_num, config.filter_num, 3, 1, 1),
            nn.BatchNorm2d(config.filter_num),
            nn.LeakyReLU(),
            nn.Conv2d(config.filter_num, config.filter_num, 3, 1, 1),
            nn.BatchNorm2d(config.filter_num)
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(config.filter_num, config.filter_num, 3, 1, 1),
            nn.BatchNorm2d(config.filter_num),
            nn.LeakyReLU(),
            nn.Conv2d(config.filter_num, config.filter_num, 3, 1, 1),
            nn.BatchNorm2d(config.filter_num)
        )
        self.res4 = nn.Sequential(
            nn.Conv2d(config.filter_num, config.filter_num, 3, 1, 1),
            nn.BatchNorm2d(config.filter_num),
            nn.LeakyReLU(),
            nn.Conv2d(config.filter_num, config.filter_num, 3, 1, 1),
            nn.BatchNorm2d(config.filter_num)
        )

        # # 8x8 input 2x2 output
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(1, 32, 5, 3, 0)
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU()
        # )

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(32, 64, 2, 1, 0)
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU()
        # )

        # fully-connected layer
        self.value = nn.Sequential(
            nn.Conv2d(config.filter_num, 1, 1, 1, 0),
            nn.BatchNorm2d(config.filter_num),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(config.filter_num,config.filter_num),
            nn.LeakyReLU(),
            nn.Linear(config.filter_num,1),
            nn.Tanh()
        )

    def forward(self, x):

        x = self.conv1(x)

        residual = x
        x = self.res1(x)
        x += residual
        x = nn.LeakyReLU(x)

        residual = x
        x = self.res2(x)
        x += residual
        x = nn.LeakyReLU(x)

        residual = x
        x = self.res3(x)
        x += residual
        x = nn.LeakyReLU(x)

        residual = x
        x = self.res4(x)
        x += residual
        x = nn.LeakyReLU(x)

        x = self.value(x)

        return x


class model:

    def __init__(self):
        self.net = Network()
        self.net.share_memory()
        
        self.loss = nn.MSELoss()
        self.optim = torch.optim.AdamW(self.net.parameters())
        self.round = config.round
        # if not self.load():
        #     self.start_round = 0
        #     self.episodes = config.episodes
        #     self.epoch = config.epoch
        #     self.memory_pool = Memory()
        #     with open('./memory.pth', 'wb') as pickle_file:
        #             pickle.dump(self.memory_pool, pickle_file)
        #     torch.save({
        #             'net': self.net.state_dict(),
        #             'optim': self.optim.state_dict(),
        #             'start_round': 0,
        #             'episodes': self.episodes,
        #             'epoch': self.epoch
        #     }, './model0.pth')

    def load(self):
        if not os.path.exists('./model.pth') or not os.path.exists('./memory.pth'):
            return False

        checkpoint = torch.load('./model.pth')
        if checkpoint['episodes'] != config.episodes or checkpoint['epoch'] != config.epoch:
            return False

        print('load model, continue training')
        self.net.load_state_dict(checkpoint['net'])
        self.optim.load_state_dict(checkpoint['optim'])
        self.start_round = checkpoint['start_round']
        self.episodes = checkpoint['episodes']
        self.epoch = checkpoint['epoch']

        with open('./memory.pth', 'rb') as pickle_file:

            self.memory_pool = pickle.load(pickle_file)

        return True

    def generate_data(self):

        self.data = mp.Manager().dict()
        # with mp.Pool(mp.cpu_count()) as p:
        #     pbar = tqdm(total=100000)
        #     def update(ret):
        #         pbar.update()
        #     for _ in range(100000):
        #         p.apply_async(random_play, args=(self.data,), callback= update)
        #     p.close()
        #     p.join()
        #     pbar.close()
        # with mp.Pool(mp.cpu_count()) as p:
        for _ in tqdm(range(10000)):
            random_play(self.data)
        print(len(self.data))
        with open('./board_data.pth', 'wb') as pickle_file:
            pickle.dump(self.data, pickle_file)
    def train(self):
        i = self.start_round
        while i < self.round:
            print('round ' + str(i+1) + ' start')
            self.net.eval()
            
            buffer = dict()

            for _ in tqdm(range(config.episodes)):
                self_play(self.net, buffer)
            # buffer = mp.Manager().dict()



            # with mp.Pool(2) as p:
            #     pbar = tqdm(total=config.episodes)
            #     def update(ret):
            #         pbar.update()

            #     for _ in range(config.episodes):
            #         p.apply_async(self_play, args=(self.net, buffer), callback= update)


            #     p.close()
            #     p.join()
            #     pbar.close()

            self.memory_pool.buffer_to_storage(buffer.copy())

            print(len(self.memory_pool))


            board_batch = self.memory_pool.get_batch()
            print(len(board_batch))

            data_set = Dataset(board_batch)
            data_loader = DataLoader(dataset=data_set,
                            num_workers=4,
                            pin_memory=True,
                            batch_size=256,
                            shuffle=True)


            self.net.train()
            for _ in range(self.epoch):
                epoch_loss = 0
                for boards, values in data_loader:
                    self.optim.zero_grad()

                    outputs = self.net(boards)
                    loss = self.loss(outputs, values)
                    epoch_loss += loss.item()
                    loss.backward()
                    self.optim.step()
                epoch_loss /= len(data_loader)
                print('loss: %.6f' %epoch_loss)


            if test_new_model(i, self.net):
                print('new model pass, start next round')
                with open('./memory.pth', 'wb') as pickle_file:
                    pickle.dump(self.memory_pool, pickle_file)

                if i < self.round-1:
                    model_suffix = str(i+1)
                else:
                    model_suffix = ''

                torch.save({
                        'net': self.net.state_dict(),
                        'optim': self.optim.state_dict(),
                        'start_round': i+1,
                        'episodes': self.episodes,
                        'epoch': self.epoch
                }, './model'+model_suffix+'.pth')
                i += 1
            else:
                print('new model fail, retrain model')
                checkpoint = torch.load('./model'+str(i)+'.pth')
                self.net.load_state_dict(checkpoint['net'])
                self.optim.load_state_dict(checkpoint['optim'])

                with open('./memory.pth', 'rb') as pickle_file:
                    self.memory_pool = pickle.load(pickle_file)

        self.test()

    def test(self):
        self.net.eval()
        
        scores = []
        for _ in range(10):
            scores.append(self.against_MCTS())

        print('test score: %d' %sum(scores))


    @torch.no_grad()
    def against_MCTS(self):
        board = np.zeros([8,8], dtype='int8')
        board[3][3] = 1
        board[4][4] = 1
        board[3][4] = -1
        board[4][3] = -1
        curr = 1
        self_mark = random.choice([1, -1])

        while True:
            positions = available_pos(board, curr)
            if len(positions) == 0:
                curr = -curr
                positions = available_pos(board, curr)

            if len(positions) == 0:
                break

            if curr == self_mark:

                next_boards = np.empty([len(positions), 8, 8])
                for i, position in enumerate(positions):
                    row, column = position
                    temp_board = np.copy(board)
                    set_position(temp_board, row, column, curr)
                    next_boards[i,:,:] = temp_board

                net_input = torch.from_numpy(next_boards).float().view(-1,1,8,8)
                net_output = self.net(net_input)
                net_output = net_output.view(-1)
                values = np.asarray(net_output) * curr
                values = values.tolist()

                index = values.index(max(values))
                set_position(board, positions[index][0], positions[index][1], curr)
            else:
                MCT_search(board, curr)

            curr = -curr

        white_score = np.count_nonzero(board==1)
        black_score = np.count_nonzero(board==-1)

        if white_score > black_score:
            return self_mark
        elif white_score < black_score:
            return -self_mark
        else:
            return 0

if __name__ == '__main__':
    mp.set_start_method('spawn')
    m = model()
    m.generate_data()

    # m.test()
    # m.s_play()
    # board_dict = Memory()
    # net = Network()
    # net.eval()
    # net.share_memory()
    # loss = nn.MSELoss()
    # optim = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    # # game_num = 200
    # # torch.set_num_interop_threads(2)
    # # BaseManager.register('Memory', Memory, exposed=['get_init_board', 'meet', '__len__', 'to_list', '__str__', 'exist', 'round_result', 'get_value'])
    # # manager = BaseManager()
    # # manager.start()
    # # board_dict = manager.Memory()

    # for _ in range(2):
    #     buffer = mp.Manager().dict()

    #             # with mp.Pool(2) as p:
    #             #     for _ in tqdm(range(config.episodes)):
    #             #         p.apply(self_play, args=(self.net, buffer)
    #     pbar = tqdm(total=100)
    #     def update(ret):
    #             pbar.update()
    #     with mp.Pool(2) as p:

    #         for _ in range(100):
    #             p.apply_async(self_play, args=(net, buffer), callback= update)


    #         p.close()
    #         p.join()
    #     pbar.close()

    #     board_dict.buffer_to_storage(buffer)
    #             # del buffer
    #     print(len(board_dict))

    #             # for i in board_dict.values():
    #             #     if i[1] != 1:
    #             #         print(i)

    #     board_batch = board_dict.to_list()
    #     print(len(board_batch))
    #             # # with open('./memory.pth', 'wb') as pickle_file:

    #             # #     pickle.dump(self.memory_pool, pickle_file)

    #     data_set = Dataset(board_batch)
    #     data_loader = DataLoader(dataset=data_set,
    #                     num_workers=4,
    #                     pin_memory=True,
    #                     batch_size=256,
    #                     shuffle=True)

    #     # self.net.to(device)
    #     net.train()
    #     for _ in range(3):
    #         epoch_loss = 0
    #         for boards, values in data_loader:
    #             boards, values = boards.to(device), values.to(device)
    #             optim.zero_grad()

    #             outputs = net(boards)
    #             loss_ret = loss(outputs, values)
    #             epoch_loss += loss_ret.item()
    #             loss_ret.backward()
    #             optim.step()
    #         epoch_loss /= len(data_loader)
    #         print('loss: %.6f' %epoch_loss)
    # torch.set_num_threads(12)
    # torch.set_num_interop_threads
    # board_dict = Memory()
    # for _ in tqdm(range(game_num)):
    #     self_play(board_dict,net)
