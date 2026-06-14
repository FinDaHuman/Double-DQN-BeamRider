"""
replay_buffer.py

Contains the Experience Replay Buffer for Deep Q-Networks.
"""

import numpy as np
import collections
import random


class ReplayBuffer:
    """
    Purpose:
        To implement an Experience Replay Buffer that stores and samples game transitions.
        This breaks the correlation between sequential game frames and improves data efficiency.

    Inputs:
        capacity (int): The maximum number of transitions the buffer can hold.

    Outputs:
        An instantiated ReplayBuffer object.

    Usage:
        buffer = ReplayBuffer(capacity=10000)
        buffer.push(state, action, reward, next_state, done)
        batch = buffer.sample(batch_size=32)
    """

    def __init__(self, capacity: int):
        """
        description: Initializes the cyclic replay buffer using a deque.
        parameters:
            capacity (int): Maximum number of items the buffer can hold.
        return values: None
        """
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        description: Stores a single transition into the buffer.
        parameters:
            state (np.ndarray): The starting state (e.g., 4 stacked frames).
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The resulting state.
            done (bool): Whether the episode terminated after this step.
        return values: None
        """
        state = np.array(state, copy=False)
        next_state = np.array(next_state, copy=False)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        description: Randomly samples a mini-batch of transitions from the buffer.
        parameters:
            batch_size (int): The number of transitions to sample.
        return values:
            A tuple of numpy arrays: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        """
        description: Returns the current size of the buffer.
        parameters: None
        return values:
            int: The number of stored transitions.
        """
        return len(self.buffer)
