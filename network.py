
""" network of MuZero """
from utilities import Action
import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional
filter_num = 32

class Dataset(Dataset):

    def __init__(self, data):
        self.images = [torch.from_numpy(image) for image, _, _ in data]
        self.actions = [[torch.from_numpy(action.encode()) for action in action_list] for _, action_list, _ in data]
        self.target_values = [[torch.tensor([value], dtype=torch.float) for value, _, _ in target_list]for _, _, target_list in data]
        self.target_rewards = [[torch.tensor([reward], dtype=torch.float) for _, reward, _ in target_list]for _, _, target_list in data]
        self.target_policies = [[torch.tensor(policy, dtype=torch.float) for _, _, policy in target_list]for _, _, target_list in data]
        self.len = len(data)

    def __getitem__(self, index):

        return self.images[index], self.actions[index], self.target_values[index], self.target_rewards[index], self.target_policies[index]

    def __len__(self):
        return self.len


@dataclass
class NetworkOutput:
    value: torch.Tensor
    reward: torch.Tensor
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
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=config.state_shape[0],
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
    def __init__(self):
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
    def __init__(self):
        super().__init__()
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

    def __init__(self):
        super().__init__()
        self.steps = 0
        self.representation = Representation()
        self.prediction = Prediction()
        self.dynamics = Dynamics()

    def initial_inference(self, image: torch.FloatTensor):
        # representation + prediction function
        if image.ndim == 3:
            image = image.unsqueeze(0)
        hidden = self.representation(image)
        policy, value = self.prediction(hidden)
        return NetworkOutput(value, torch.Tensor([[0]]), policy, hidden)

    def recurrent_inference(self, hidden_state: torch.FloatTensor, action: torch.FloatTensor):
        if hidden_state.ndim == 3:
            hidden_state = hidden_state.unsqueeze(0)
        if action.ndim == 3:
            action = action.unsqueeze(0)
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
            return Network()

    def old_network(self) -> Network:
        if self._networks:
            return self._networks[min(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return Network()
    def save_network(self, step: int, network: Network):
        self._networks[step] = network
