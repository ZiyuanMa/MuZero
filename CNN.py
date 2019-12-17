import torch
import torch.nn as nn

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