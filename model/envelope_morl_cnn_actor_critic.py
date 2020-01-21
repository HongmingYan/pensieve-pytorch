## Modified by Yajie Zhou and Kasim Te, January 2020

# import torch.nn.functional as F
import torch.nn as nn
import torch
# import torch.optim as optim
# import numpy as np
# import math
from torch.nn import init

from torch.distributions.categorical import Categorical

# Code borrowed and adapted from:
#
# https://github.com/RunzheYang/MORL/blob/master/multimario/model.py

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# Envelope MORL CNN Actor Critic Network implementation, adapted for
# adaptive bitrate algorithm.
class EnvelopeMORLCnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, reward_size):
        super(EnvelopeMORLCnnActorCriticNetwork, self).__init__()

        linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            linear(
                7 * 7 * 64,
                512),
            nn.LeakyReLU(),
        )
        self.actor = nn.Sequential(
            linear(512+reward_size, 128),
            nn.LeakyReLU(),
            linear(128, output_size),
        )
        self.critic = nn.Sequential(
            linear(512+reward_size, 128),
            nn.LeakyReLU(),
            linear(128, reward_size),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                module.bias.data.zero_()

            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, a=1.0)
                module.bias.data.zero_()

    def forward(self, state, preference):
        x_value = self.feature(state)
        x_value = torch.cat((x_value, preference), dim=1)
        policy = self.actor(x_value)
        value = self.critic(x_value)
        return policy, value
