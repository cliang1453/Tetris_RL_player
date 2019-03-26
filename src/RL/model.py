import torch.nn as nn
import torch.nn.functional as F
import torch

NUM_FEATURES = 6


class QFunc(nn.Module):
    def __init__(self, args):
        super(QFunc, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(34 * 64, 512)
        self.fc5 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print("after conv1", x.shape)
        x = F.relu(self.conv2(x))
        # print("after conv2", x.shape)
        x = F.relu(self.conv3(x))
        # print("shape", x.shape)
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


class QFuncFeature(nn.Module):
    def __init__(self, args):
        super(QFuncFeature, self).__init__()
        self.fc4 = nn.Linear(NUM_FEATURES, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, features):
        x = F.relu(self.fc4(features))
        return self.fc5(x)


class QFuncAll(nn.Module):
    def __init__(self, args):
        super(QFuncAll, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(34 * 64, 512)
        self.fc5 = nn.Linear(512, 32)

        self.fc_feature1 = nn.Linear(NUM_FEATURES, 32)
        self.fc_final = nn.Linear(64, 1)

    def forward(self, x, features):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x = F.relu(self.fc5(x))
        y = F.relu(self.fc_feature1(features))

        x = torch.cat((x, y), dim=1)
        x = self.fc_final(x)

        return x

# class DQN_RAM(nn.Module):
#     def __init__(self, in_features=4, num_actions=18):
#         """
#         Initialize a deep Q-learning network for testing algorithm
#             in_features: number of features of input.
#             num_actions: number of action-value to output, one-to-one correspondence to action in game.
#         """
#         super(DQN_RAM, self).__init__()
#         self.fc1 = nn.Linear(in_features, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, num_actions)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         return self.fc4(x)
