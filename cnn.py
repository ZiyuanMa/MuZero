import random
from math import sqrt, log
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager
pool_num = round(mp.cpu_count()/4)
from tqdm import tqdm
from reversi import available_pos, set_position
from MCTS import MCT_search
from data_struct import container
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#device = torch.device('cpu')
torch.manual_seed(1261)
random.seed(1261)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

@torch.no_grad()
def self_play(board_dict, net):
    board, curr = board_dict.get_init_board()
    round_boards = []
    while True:

        board_dict.meet(board)
        round_boards.append(np.copy(board))

        positions = available_pos(board, curr)
        if len(positions) == 0:
            curr = -curr
            positions = available_pos(board, curr)

        if len(positions) == 0:
            break

        visit_times = []
        next_boards = np.empty([len(positions), 8, 8])
        for i, position in enumerate(positions):
            row, column = position
            temp_board = np.copy(board)
            set_position(temp_board, row, column, curr)

            next_boards[i,:,:] = temp_board
            if board_dict.exist(temp_board):
                visit_times.append(board_dict.get_value(temp_board)[1])
            else:
                visit_times.append(0)
            # net_input = torch.from_numpy(temp_board).view(1,1,8,8).to(device)

            # net_output = net(net_input)
            # out_value = net_output.item()
            # value = out_value * curr
            # values.append(value)
        net_input = torch.from_numpy(next_boards).view(-1,1,8,8).to(device)
        net_output = net(net_input)
        net_output = net_output.cpu().view(-1)
        values = np.asarray(net_output) * curr

        values = values.tolist()

        sum_visit = sum(visit_times)+1
        scores = [value + sqrt(log(sum_visit)/(visit+1)) for value, visit in zip(values, visit_times)]
        index = scores.index(max(scores))
        set_position(board, positions[index][0], positions[index][1], curr)

        curr = -curr

    white_score = np.count_nonzero(board==1)
    black_score = np.count_nonzero(board==-1)

    if white_score > black_score:
        board_dict.round_result(round_boards, 1)
    elif white_score < black_score:
        board_dict.round_result(round_boards, -1)



@torch.no_grad()
def against_MCTS(scores, net):
    board = np.zeros([8,8], dtype='double')
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
            values = []

            for row, column in positions:
                temp_board = np.copy(board)
                set_position(temp_board, row, column, curr)
                net_input = torch.from_numpy(temp_board).view(1,1,8,8).to(device)
                net_output = net(net_input)
                value = net_output.item() * curr
                values.append(value)

            index = values.index(max(values))
            set_position(board, positions[index][0], positions[index][1], curr)
        else:
            MCT_search(board, curr)

        curr = -curr

    white_score = np.count_nonzero(board==1)
    black_score = np.count_nonzero(board==-1)

    if white_score > black_score:
        scores.append(self_mark)
    elif white_score < black_score:
        scores.append(-self_mark)
    else:
        scores.append(0)

class DealDataset(Dataset):

    def __init__(self, data):

        self.x_data = [board for board, _ in data]
        self.y_data = [torch.tensor([value], dtype=torch.double, device=device) for _, value in data]
        self.len = len(data)

    def __getitem__(self, index):

        return self.transform(self.x_data[index]), self.y_data[index]


    def __len__(self):
        return self.len

    def transform(self, narray):
        if random.choice([True, False]):
            
            return torch.from_numpy(np.rot90(narray, 2).copy()).view(1,8,8).to(device)
        else:
            return torch.from_numpy(narray).view(1,8,8).to(device)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 8x8 input 6x6 output
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
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
            nn.Dropout(0.3),
            nn.Linear(256,256),
            nn.Dropout(0.3),
            nn.Linear(256,1)
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
        self.net = self.net.double().to(device)
        self.loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.net.parameters())
        self.epch = 3
        mp.set_start_method('forkserver')

    def train(self):

        for _ in range(10):
            self.net.eval()
            self.net.share_memory()
            
            BaseManager.register('container', container, exposed=['get_init_board', 'meet', '__len__', 'to_list', '__str__', 'exist', 'round_result', 'get_value'])
            manager = BaseManager()
            manager.start()
            board_dict = manager.container()

            with mp.Pool(3) as p:
                pbar = tqdm(total=20)
                def update(ret):
                    pbar.update()

                for _ in range(20):
                    p.apply_async(self_play, args=(board_dict,self.net), callback=update)
                    #pbar.update()

                p.close()
                p.join()
                pbar.close()


            # board_dict = container()
            # with torch.no_grad():
            #     for _ in range(110):
            #         self_play(board_dict,self.net)
            

            
            print(len(board_dict))
            print(board_dict)
            # for i in board_dict.values():
            #     if i[1] != 1:
            #         print(i)

            board_list = board_dict.to_list()
            print(len(board_list))


            data_set = DealDataset(board_list)
            data_loader = DataLoader(dataset=data_set,
                            batch_size=256,
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

            #self.test()
            self.net.eval()
            self.net.share_memory()

            scores = mp.Manager().list()
            with mp.Pool(5) as p:
                
                for _ in range(10):
                    p.apply_async(against_MCTS, args=(scores, self.net,))


                p.close()
                p.join()

            print('test score: %d' %sum(scores))

            torch.save(self.net.state_dict(), './model1.pth')

    def test(self):
        self.net.eval()
        self.net.share_memory()

        scores = []
        with mp.Pool(10) as p, torch.no_grad():
            
            for _ in range(10):
                score = p.apply_async(against_MCTS, args=(self.net,)).get()
                scores.append(score)

            p.close()
            p.join()

        print('test score: %d' %sum(scores))



if __name__ == '__main__':
    m = model()
    m.train()