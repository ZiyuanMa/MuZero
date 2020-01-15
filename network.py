from environment import Action
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from typing import Dict, List, Optional
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

filter_num = 16

@dataclass
class NetworkOutput:
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]

class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)

class Conv(nn.Module):
    def __init__(self, filters0, filters1, kernel_size, bn=False):
        super().__init__()
        self.conv = nn.Conv2d(filters0, filters1, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(filters1)

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h

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
        return F.leaky_relu(x + (self.conv(x)))

class Representation(nn.Module):
    # from board to hidden state
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.board_size = self.input_shape[1] * self.input_shape[2]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_shape[0],
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
        self.board_size = 42
        self.action_size = action_shape

        self.policy_head = nn.Sequential(
            nn.Conv2d(filter_num, 2, 1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(self.board_size*2, self.board_size),
            nn.Softmax()
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
            nn.Conv2d(in_channels=rp_shape[0] + act_shape[0],
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
        input_shape = (3, 8, 8)
        rp_shape = (filter_num, *input_shape[1:])
        self.representation = Representation(input_shape).to(device)
        self.prediction = Prediction(action_space_size).to(device)
        self.dynamics = Dynamics(rp_shape, (1, 8, 8)).to(device)
        self.eval()
  
    def predict_initial_inference(self, x):    
        assert x.ndim in (3, 4)
        assert x.shape == (3, 8, 8) or x.shape[1:] == (3, 8, 8)
        orig_x = x
        if x.ndim == 3:
            x = x.reshape(1, 3, 8, 8)
        
        x = torch.Tensor(x).to(device)
        h = self.representation(x)
        policy, value = self.prediction(h)
        
        if orig_x.ndim == 3:
            return h[0], policy[0], value[0]
        else:
            return h, policy, value

    def predict_recurrent_inference(self, x, a):

        if x.ndim == 3:
            x = x.reshape(1, 3, 8, 8)

        a = torch.Tensor(a).to(device)

        g = self.dynamics(x, a)
        policy, value = self.prediction(g)
        
        return g[0], policy[0], value[0]

    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        h, p, v = self.predict_initial_inference(image.astype(numpy.float32))
        return NetworkOutput(v, 0, p, h)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        g, p, v = self.predict_recurrent_inference(hidden_state, action)
        return NetworkOutput(v, 0, p, g) 

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
            return make_uniform_network()

    def old_network(self) -> Network:
        if self._networks:
            return self._networks[min(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network()

    def save_network(self, step: int, network: Network):
        self._networks[step] = network
