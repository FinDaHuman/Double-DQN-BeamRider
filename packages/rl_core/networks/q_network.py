"""
q_network.py

Contains the Convolutional Neural Network (CNN) architecture for the DQN.
"""

import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    """
    Purpose:
        To define the Convolutional Neural Network (CNN) architecture that approximates 
        the Q-value function Q(s, a) for the Beam Rider environment.
        
    Inputs:
        input_shape (tuple): Shape of the stacked frames, e.g., (4, 84, 84).
        num_actions (int): The number of possible discrete actions.
        
    Outputs:
        An instantiated DQN PyTorch Module.
        
    Usage:
        network = DQN(input_shape=(4, 84, 84), num_actions=9)
        q_values = network(state_tensor)
    """
    def __init__(self, input_shape: tuple, num_actions: int):
        """
        description: Initializes the deep neural network layers.
        parameters:
            input_shape (tuple): Shape of the stacked frames.
            num_actions (int): The number of possible discrete actions.
        return values: None
        """
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.feature_size = self._get_conv_output(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def _get_conv_output(self, shape):
        """
        description: A helper function to dynamically calculate the size of the tensor 
                     after passing through the convolutional layers.
        parameters:
            shape (tuple): The input shape (C, H, W).
        return values:
            int: The total number of features (flattened size).
        """
        o = self.features(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        description: Defines the forward pass of the network.
        parameters:
            x (torch.Tensor): A batch of stacked frames of shape (Batch, 4, 84, 84).
        return values:
            torch.Tensor: A tensor of shape (Batch, num_actions) containing Q-values.
        """
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
