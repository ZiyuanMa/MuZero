import random
import math
import torch
import torch.nn as nn
import numpy as np
from reversi import *
from MCTS import MCT_search

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 8x8 input 6x6 output
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                            out_channels=8,
                            kernel_size=3,
                            stride=1,
                            padding=0),
            nn.BatchNorm2d(8),
            nn.LeakyReLU()
        )

        # 6x6 input 4x4 output
        self.conv2 = nn.Sequential(
            nn.Conv2d(8,32,3,1,0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        # 4x4 input 1x1 output
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,128,3,1,0),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )

        # fully-connected layer
        self.FC = nn.Sequential(
            nn.Linear(128,128),
            nn.Dropout(0.3),
            nn.Linear(128,1)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.FC(x.view(x.size(0),-1))

        return x


class model:
    def __init__(self):
        self.network = CNN()
        self.network = self.network.double()
        self.loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.network.parameters())

    def train(self):
        self.network.eval()

        # value total_number predict_value
        board_dict = dict()
        for _ in range(100):
            round_boards = dict()
            self.init_board()
            while True:
                positions = available_pos(self.board, self.curr)
                if len(positions) == 0:
                    self.curr = -self.curr
                    positions = available_pos(self.board, self.curr)

                if len(positions) == 0:
                    break

                values = []
                visit_times = []
                for row, column in positions:
                    temp_board = np.copy(self.board)
                    set_position(temp_board, row, column, self.curr)
                    bytes_board = temp_board.tobytes('F')
                    if bytes_board not in board_dict:
                        visit_times.append(0)
                    else:
                        visit_times.append(board_dict[bytes_board][1])

                    input = torch.from_numpy(temp_board).view(1,1,8,8)
                    #print(input)
                    value = self.network(input).item() * self.curr
                    values.append(value)

                sum_visit = math.sqrt(sum(visit_times)+1)
                scores = [value * sum_visit / (visit+1) for value, visit in zip(values, visit_times)]
                index = scores.index(max(scores))
                set_position(self.board, positions[index][0], positions[index][1], self.curr)
                round_boards[self.board.tobytes('F')] = values[index]
                self.curr = -self.curr

            white_score = np.count_nonzero(self.board==1)
            black_score = np.count_nonzero(self.board==-1)

            if white_score > black_score:
                round_score = 1
            elif white_score < black_score:
                round_score = -1
            else:
                round_score = 0

            for board, value in round_boards.items():

                if board in board_dict:
                    board_dict[board][0] += round_score
                    board_dict[board][1] += 1
                else:

                    board_dict[board] = [round_score, 1, value]

        print(len(board_dict))
        board_list = [(board, value[0]/value[1]) for board, value in board_dict.items() if abs(value[2]-value[0]/value[1]) > 0.5]
        print(len(board_list))

    def init_board(self):
        self.board = np.zeros([8,8], dtype='double')
        self.board[3][3] = 1
        self.board[4][4] = 1
        self.board[3][4] = -1
        self.board[4][3] = -1
        self.curr = 1



if __name__ == '__main__':
    m = model()
    m.train()