from random import choice
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.multiprocessing as mp
pool_num = round(mp.cpu_count()/4)
from reversi import available_pos, set_position
from MCTS import MCT_search

def self_play(board_dict, net):
    board = np.zeros([8,8], dtype='double')
    board[3][3] = 1
    board[4][4] = 1
    board[3][4] = -1
    board[4][3] = -1
    curr = 1
    round_boards = list()
    while True:
        positions = available_pos(board, curr)
        if len(positions) == 0:
            curr = -curr
            positions = available_pos(board, curr)

        if len(positions) == 0:
            break

        values = []
        visit_times = []
        for row, column in positions:
            temp_board = np.copy(board)
            set_position(temp_board, row, column, curr)
            bytes_board = temp_board.tobytes()
            if bytes_board not in board_dict:
                visit_times.append(0)
            else:
                visit_times.append(board_dict[bytes_board][1])

            net_input = torch.from_numpy(temp_board).view(1,1,8,8)
            
            value = net(net_input).item() * curr
            values.append(value)

        sum_visit = math.sqrt(sum(visit_times))
        scores = [value * sum_visit / (visit+1) for value, visit in zip(values, visit_times)]
        index = scores.index(max(scores))
        set_position(board, positions[index][0], positions[index][1], curr)

        if board.tobytes() not in board_dict:
            board_dict[board.tobytes()] = [0, 1, values[index]*curr]
        else:
            board_dict[board.tobytes()][1] += 1

        round_boards.append(board.tobytes())
        curr = -curr

    white_score = np.count_nonzero(board==1)
    black_score = np.count_nonzero(board==-1)

    if white_score > black_score:
        round_score = 1
    elif white_score < black_score:
        round_score = -1
    else:
        round_score = 0

    for board in round_boards:

        board_dict[board][0] += round_score

def against_MCTS(net):
    board = np.zeros([8,8], dtype='double')
    board[3][3] = 1
    board[4][4] = 1
    board[3][4] = -1
    board[4][3] = -1
    curr = 1
    self_mark = choice([1, -1])
    while True:
        positions = available_pos(board, curr)
        if len(positions) == 0:
            curr = -curr
            positions = available_pos(board, curr)

        if len(positions) == 0:
            break

        if curr == self_mark:
            values = []

            for row, column in positions:
                temp_board = np.copy(board)
                set_position(temp_board, row, column, curr)
                net_input = torch.from_numpy(temp_board).view(1,1,8,8)

                value = net(net_input).item() * curr
                values.append(value)

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

class DealDataset(Dataset):

    def __init__(self, data):

        self.x_data = [torch.from_numpy(board) for board, _ in data]
        self.y_data = [torch.tensor([value], dtype=torch.double) for _, value in data]
        self.len = len(self.x_data)
    
    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 8x8 input 6x6 output
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )

        # 6x6 input 4x4 output
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, 3, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        # 4x4 input 1x1 output
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
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
            nn.Linear(128,128),
            nn.Dropout(0.3),
            nn.Linear(128,1)
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
        self.net = self.net.double()
        self.loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.net.parameters())
        self.epch = 3
    def train(self):
        for _ in range(10):
            self.net.eval()
            self.net.share_memory()

            p = mp.Pool(pool_num)
            board_dict = mp.Manager().dict()

            with torch.no_grad():
                for _ in range(10000):
                    p.apply_async(self_play, args=(board_dict,self.net))

            p.close()
            p.join()

            print(len(board_dict))
            # for board, value in board_dict.items():
            #     print(str(value[0])+' '+str(value[1]) + ' ' + str(value[2]))
            board_list = [(np.frombuffer(board, dtype='double').reshape(1,8,8), value[0]/value[1]) for board, value in board_dict.items() if abs(value[2]-value[0]/value[1]) > 0.05]
            print(len(board_list))

            data_set = DealDataset(board_list)
            data_loader = DataLoader(dataset=data_set,
                            batch_size=128,
                            shuffle=True)

            self.net.train()
            for _ in range(self.epch):
                epch_loss = 0
                for boards, values in data_loader:

                    self.opt.zero_grad()

                    outputs = self.net(boards)
                    loss = self.loss(outputs, values)
                    epch_loss += loss.item()
                    loss.backward()
                    self.opt.step()
                epch_loss /= len(data_loader)
                print('loss: %.6f' %epch_loss)

            self.test()
            torch.save(self.net.state_dict(), './model.pth')

    def test(self):
        self.net.eval()
        self.net.share_memory()

        p = mp.Pool(pool_num)

        scores = []
        with torch.no_grad():
            for _ in range(10):
                re = p.apply_async(against_MCTS, args=(self.net))
                scores.append(re.get())

        p.close()
        p.join()

        print('test score: %d' %sum(scores))



if __name__ == '__main__':
    m = model()
    m.train()