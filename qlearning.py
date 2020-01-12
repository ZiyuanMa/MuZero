import config
from reversi import available_pos, set_position
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pickle
torch.manual_seed(1261)
random.seed(1261)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def board_to_input(board, tern):
    self_board = board==tern
    opponent_board = board==-tern
    np_input = np.concatenate(self_board, opponent_board)
    return torch.from_numpy(np_input).view(1,2,8,8)

class Node:
    def __init__(self, board, tern, parent, net):
        self.visit_times = 0
        self.total_value = 0
        self.parent = parent
        self.children = []
        self.board = board
        self.tern = tern
        if self.tern == 0:
            white_score = np.count_nonzero(board==1)
            black_score = np.count_nonzero(board==-1)
            if white_score > black_score:
                self.reward = 1
            elif white_score < black_score:
                self.reward = -1
            else:
                self.reward = 0
            self.available_pos = []
        else:
        
            self.available_pos = available_pos(self.board, self.tern)
        self.next_prob = []
        self.net = net
    def __len__(self):
        count = 0
        for child in self.children:
            count += len(child)
        return 1+count
    def expand(self):
        for row, column in self.available_pos:
            next_board = np.copy(self.board)
            set_position(next_board, row, column, self.tern)
            next_tern = -self.tern
            next_pos = available_pos(next_board, next_tern)
            if len(next_pos) == 0:
                next_tern = -next_tern
                next_pos = available_pos(next_board, next_tern)
                if len(next_pos) == 0:
                    child = Node(next_board, 0, self, None)
                    self.children.append(child)
                    self.next_prob.append(child.reward*self.tern)
                else:
                    input = board_to_input(next_board, next_tern)
                    value = self.net(input)
                    self.next_prob.append(value.item())
            else:
                input = board_to_input(next_board, next_tern)
                value = self.net(input)
                self.next_prob.append(value.item())
                
    def update_prob(self):
    def search(self):
        self.visit_times += 1
        if self.tern == 0:
            self.total_value += self.reward
            return self.reward
        else:
            if not self.children:
                self.expand()
            



class Tree:
    def __init__(self, board, tern, net):
        self.root = Node(board, tern, None, net)

    def __len__(self):
        return len(self.root)

    def search(self, num):
        for _ in range(num):
            self.root.search()
    



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


        # value head
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

def random_play(data_dict):
    board = np.zeros([8,8])
    board[3][3] = 1
    board[4][4] = 1
    board[3][4] = -1
    board[4][3] = -1
    next = 1
    while True:
        bytes_board = board.tobytes()
        positions = available_pos(board, next)
        if len(positions) == 0:
            next = -next
            positions = available_pos(board, next)
            if len(positions) == 0:
                break
        key = (bytes_board, next)
        data_dict.add(key)

        row, column = random.choice(positions)
        set_position(board, row, column, next)
        next = -next


def generate_data():

    data =  set()
    for _ in tqdm(range(10000)):
        random_play(data)
    print(len(data))
    with open('./board_data.pth', 'wb') as pickle_file:
        pickle.dump(data, pickle_file)