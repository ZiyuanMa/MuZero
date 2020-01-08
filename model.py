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
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
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
        # if len(positions) == 0:
        #     next = -next
        #     positions = available_pos(board, next)

        #     if len(positions) == 0:

        #         if (bytes_board, 0) not in buffer:
        #             buffer[(bytes_board, 0)] = [0, 1]
        #         else:
        #             tmp = buffer[(bytes_board, 0)]
        #             tmp[1] += 1
        #             buffer[(bytes_board, 0)] = tmp
        #         round_boards.append((bytes_board, 0))
        #         break
        #     else:

        #         if (bytes_board, next) not in buffer:
        #             buffer[(bytes_board, next)] = [0, 1]
        #         else:
        #             tmp = buffer[(bytes_board, next)]
        #             tmp[1] += 1
        #             buffer[(bytes_board, next)] = tmp

        #         round_boards.append((bytes_board, next))
        # else:

        #     if (bytes_board, next) not in buffer:
        #         buffer[(bytes_board, next)] = [0, 1]
        #     else:
        #         tmp = buffer[(bytes_board, next)]
        #         tmp[1] += 1
        #         buffer[(bytes_board, next)] = tmp

        #     round_boards.append((bytes_board, next))


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
            # if self.board_dict.exist(temp_board):
            #     visit_times.append(self.board_dict.get_value(temp_board)[1])
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


class DealDataset(Dataset):

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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 8x8 input 6x6 output
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2,
                            out_channels=256,
                            kernel_size=3,
                            stride=1,
                            padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        # 6x6 input 4x4 output
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        # 4x4 input 1x1 output
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
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
        self.FC = nn.Sequential(
            nn.Linear(256,256),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(256,256),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(256,1),
            nn.Tanh()
        )

    def forward(self, x):

        x1_out = self.conv1(x)
        x1_out = self.conv2(x1_out)
        x1_out = self.conv3(x1_out)

        # x2_out = self.conv4(x)
        # x2_out = self.conv5(x2_out)


        out = self.FC(x1_out.view(x1_out.size(0),-1))

        return out


class model:

    def __init__(self):
        self.net = CNN()
        self.net.share_memory()
        self.loss = nn.MSELoss()
        self.optim = torch.optim.AdamW(self.net.parameters())
        self.round = config.round
        if not self.load():
            self.start_round = 0
            self.episodes = config.episodes
            self.epoch = config.epoch
            self.board_dict = Memory()

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

            self.board_dict = pickle.load(pickle_file)

        return True


    def train(self):

        self.board_dict = Memory()

        for i in range(self.start_round, self.round):
            print('round ' + str(i+1) + ' start')
            self.net.eval()
            
            # self.board_dict = Memory()

            # for _ in tqdm(range(config.episodes)):
            #     self_play(self.net, self.board_dict)
            buffer = mp.Manager().dict()

            # with mp.Pool(2) as p:
            #     for _ in tqdm(range(config.episodes)):
            #         p.apply(self_play, args=(self.net, buffer))



            with mp.Pool(2) as p:
                pbar = tqdm(total=config.episodes)
                def update(ret):
                    pbar.update()

                for _ in range(config.episodes):
                    p.apply_async(self_play, args=(self.net, buffer), callback= update)


                p.close()
                p.join()
                pbar.close()

            self.board_dict.buffer_to_storage(buffer.copy())

            print(len(self.board_dict))

            # for i in board_dict.values():
            #     if i[1] != 1:
            #         print(i)

            board_list = self.board_dict.to_list()
            print(len(board_list))

            data_set = DealDataset(board_list)
            data_loader = DataLoader(dataset=data_set,
                            num_workers=4,
                            pin_memory=True,
                            batch_size=256,
                            shuffle=True)

            # self.net.to(device)
            self.net.train()
            for _ in range(self.epoch):
                epoch_loss = 0
                for boards, values in data_loader:
                    boards, values = boards.to(device), values.to(device)
                    self.optim.zero_grad()

                    outputs = self.net(boards)
                    loss = self.loss(outputs, values)
                    epoch_loss += loss.item()
                    loss.backward()
                    self.optim.step()
                epoch_loss /= len(data_loader)
                print('loss: %.6f' %epoch_loss)

            # self.net.to(torch.device('cpu'))

            with open('./memory.pth', 'wb') as pickle_file:
                pickle.dump(self.board_dict, pickle_file)

            torch.save({
                    'net': self.net.state_dict(),
                    'optim': self.optim.state_dict(),
                    'start_round': i+1,
                    'episodes': self.episodes,
                    'epoch': self.epoch
            }, './model.pth')

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
    m.train()

    # m.test()
    # m.s_play()
    # board_dict = Memory()
    # net = CNN()
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

    #     board_list = board_dict.to_list()
    #     print(len(board_list))
    #             # # with open('./memory.pth', 'wb') as pickle_file:

    #             # #     pickle.dump(self.board_dict, pickle_file)

    #     data_set = DealDataset(board_list)
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
