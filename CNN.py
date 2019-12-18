import random
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
            nn.Conv2d(32,64,3,1,0),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        # fully-connected layer
        self.FC = nn.Sequential(
            nn.Linear(64,64),
            nn.Dropout(0.3),
            nn.Linear(64,1)
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
        self.loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.network.parameters())

    def train(self):
        self.network.eval()
        for _ in range(100):
            self.init_board()
            while True:
                positions = available_pos(self.board, self.curr)
                if len(positions) == 0:
                    self.curr = -self.curr
                    positions = available_pos(self.board, self.curr)

                if len(positions) == 0:
                    break

                scores = []
                for row, column in positions:
                    temp_board = np.copy(self.board)
                    set_position(temp_board, row, column, self.curr)
                    score = self.network(torch.from_numpy(temp_board))
                    scores.append(score)

    def init_board(self):
        self.board = np.zeros([8,8])
        self.board[3][3] = 1
        self.board[4][4] = 1
        self.board[3][4] = -1
        self.board[4][3] = -1
        self.curr = 1