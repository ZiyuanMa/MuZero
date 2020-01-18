from environment import Action
from config import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from typing import Dict, List, Optional
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = 'cpu'
filter_num = 16

@dataclass
class NetworkOutput:
    value: torch.Tensor
    reward: float
    policy_logits: torch.Tensor
    hidden_state: torch.Tensor

class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.BatchNorm2d(filters)
        )

    def forward(self, x):
        return F.leaky_relu(x + (self.block(x)))

class Representation(nn.Module):
    # from board to hidden state
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.board_size = self.input_shape[1] * self.input_shape[2]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4,
                            out_channels=filter_num,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            nn.BatchNorm2d(filter_num),
            nn.LeakyReLU()
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(filter_num) for _ in range(4)])

    def forward(self, x):
        x = self.conv1(x)
        for block in self.res_blocks:
            x = block(x)
        return x

class Prediction(nn.Module):
    # use hidden state to predict value and policy
    def __init__(self, action_shape):
        super().__init__()
        self.board_size = 64

        self.policy_head = nn.Sequential(
            nn.Conv2d(filter_num, 2, 1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(self.board_size*2, self.board_size+1),
            nn.Softmax(dim=1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(filter_num, 1, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(self.board_size, self.board_size),
            nn.LeakyReLU(),
            nn.Linear(self.board_size, 1),
            nn.Tanh()
        )

    def forward(self, x):
        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value

class Dynamics(nn.Module):
    '''Hidden state transition'''
    def __init__(self, rp_shape, act_shape):
        super().__init__()
        self.rp_shape = rp_shape
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=filter_num + 1,
                            out_channels=filter_num,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            nn.BatchNorm2d(filter_num),
            nn.LeakyReLU()
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(filter_num) for _ in range(4)])

    def forward(self, rp, a):
        h = torch.cat([rp, a], dim=1)
        h = self.conv1(h)
        for block in self.res_blocks:
            h = block(h)
        return h

class Network(nn.Module):

    def __init__(self, action_space_size: int):
        super().__init__()
        self.steps = 0
        self.action_space_size = action_space_size
        input_shape = (4, 8, 8)
        rp_shape = (filter_num, *input_shape[1:])
        self.representation = Representation(input_shape).to(device)
        self.prediction = Prediction(action_space_size).to(device)
        self.dynamics = Dynamics(rp_shape, (1, 8, 8)).to(device)
        self.eval()

    def initial_inference(self, image: torch.Tensor):
        # representation + prediction function
        hidden = self.representation(image)
        policy, value = self.prediction(hidden)
        return NetworkOutput(value, torch.Tensor([[0]]), policy, hidden)

    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor):
        # dynamics + prediction function
        hidden = self.dynamics(hidden_state, action)
        policy, value = self.prediction(hidden)
        return NetworkOutput(value, torch.Tensor([[0]]), policy, hidden)

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.steps

class SharedStorage:

    def __init__(self):
        self._networks = {}

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return Network(make_reversi_config().action_space_size).to(device)

    def old_network(self) -> Network:
        if self._networks:
            return self._networks[min(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return Network(make_reversi_config().action_space_size).to(device)
    def save_network(self, step: int, network: Network):
        self._networks[step] = network
