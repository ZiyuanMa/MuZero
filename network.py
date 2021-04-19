
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

# class Dataset(Dataset):

#     def __init__(self, data):
#         self.images = [torch.from_numpy(image) for image, _, _ in data]
#         self.actions = [[torch.from_numpy(action.encode()) for action in action_list] for _, action_list, _ in data]
#         self.target_values = [[torch.tensor([value], dtype=torch.float) for value, _, _ in target_list]for _, _, target_list in data]
#         self.target_rewards = [[torch.tensor([reward], dtype=torch.float) for _, reward, _ in target_list]for _, _, target_list in data]
#         self.target_policies = [[torch.tensor(policy, dtype=torch.float) for _, _, policy in target_list]for _, _, target_list in data]
#         self.len = len(data)

#     def __getitem__(self, index):

#         return self.images[index], self.actions[index], self.target_values[index], self.target_rewards[index], self.target_policies[index]

#     def __len__(self):
#         return self.len




class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            # nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, 1, 1),
            # nn.BatchNorm2d(filters)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))

class Representation(nn.Module):
    # from board to hidden state
    def __init__(self, num_channels=config.num_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(config.state_shape[0], num_channels, 3, 1, 1),
            # nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            ResidualBlock(num_channels),
            ResidualBlock(num_channels),
            ResidualBlock(num_channels),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(num_channels, 8, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # for block in self.res_blocks:
        #     x = block(x)
        return x

class Prediction(nn.Module):
    # use hidden state to predict value and policy
    def __init__(self, num_channels=config.num_channels):
        super().__init__()
        self.board_size = 64

        self.flatten = nn.Flatten()

        self.policy_head = nn.Sequential(
            nn.Conv2d(8, 2, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.board_size*2, self.board_size*2),
            nn.ReLU(),
            nn.Linear(self.board_size*2, config.action_space_size),
            nn.Softmax(dim=1)
        )

        self.value_head = nn.Sequential(

            nn.Conv2d(8, 2, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.board_size*2, 1),
            nn.Tanh()
        )

    def forward(self, x):

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value

class Dynamics(nn.Module):
    '''Hidden state transition'''
    def __init__(self, num_channels=config.num_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(10, num_channels, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(num_channels),
            ResidualBlock(num_channels),
            ResidualBlock(num_channels),
            nn.Conv2d(num_channels, 8, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(2),
            nn.ReLU(),

        )

    def forward(self, rp, a):
        h = torch.cat((rp, a), dim=1)
        h = self.conv(h)

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
        # print(image.size())
        hidden = self.representation(image)
        policy, value = self.prediction(hidden)
        return value, policy, hidden

    def recurrent_inference(self, hidden_state: torch.FloatTensor, action: torch.FloatTensor):
        if hidden_state.ndim == 3:
            hidden_state = hidden_state.unsqueeze(0)
        if action.ndim == 3:
            action = action.unsqueeze(0)
        # dynamics + prediction function
        hidden = self.dynamics(hidden_state, action)
        policy, value = self.prediction(hidden)
        return value, policy, hidden

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.steps

class SharedStorage:

    def __init__(self, init_network_id):
        self.network_id = init_network_id

    def get_network(self) -> Network:
        return self.network_id

    def save_network(self, new_network_id):
        self.network_id = new_network_id
