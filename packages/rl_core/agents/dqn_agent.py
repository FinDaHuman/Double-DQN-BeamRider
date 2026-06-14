"""
dqn_agent.py

Contains the Agent logic for action selection and network updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from networks.q_network import DQN
from replay.replay_buffer import ReplayBuffer

class DQNAgent:
    """
    Purpose:
        To implement the Deep Q-Network Agent that handles action selection, epsilon-greedy 
        exploration, and the mathematical Bellman Update step.
        
    Inputs:
        state_shape (tuple): The shape of the preprocessed state.
        num_actions (int): The number of available actions.
        lr (float): The learning rate for the optimizer.
        device (str): Device to run computation on ("cpu" or "cuda").
        
    Outputs:
        An instantiated DQNAgent object.
        
    Usage:
        agent = DQNAgent(state_shape, num_actions)
        action = agent.select_action(state, epsilon)
        loss = agent.train_step(replay_buffer, batch_size, gamma)
    """
    def __init__(self, state_shape: tuple, num_actions: int, lr: float = 1e-4, device: str = "cpu"):
        """
        description: Initializes the online and target networks, optimizer, and loss function.
        parameters:
            state_shape (tuple): The shape of the preprocessed state.
            num_actions (int): The number of available actions.
            lr (float): The learning rate.
            device (str): Device to run computation on.
        return values: None
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.device = torch.device(device)
        
        self.online_net = DQN(state_shape, num_actions).to(self.device)
        self.target_net = DQN(state_shape, num_actions).to(self.device)
        self.update_target_network()
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        description: Uses epsilon-greedy strategy to select an action.
        parameters:
            state (np.ndarray): The current preprocessed state.
            epsilon (float): Probability of taking a random action [0.0, 1.0].
        return values:
            int: The selected action index.
        """
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
            with torch.no_grad():
                q_values = self.online_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def update_target_network(self):
        """
        description: Hard update: Copies weights from Online Network to Target Network.
        parameters: None
        return values: None
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int, gamma: float) -> float:
        """
        description: Samples a batch from the replay buffer and performs one Bellman update step.
        parameters:
            replay_buffer (ReplayBuffer): The experience replay memory.
            batch_size (int): Number of transitions to sample.
            gamma (float): Discount factor for future rewards.
        return values:
            float: The loss value for this training step (or 0.0 if not enough data).
        """
        if len(replay_buffer) < batch_size:
            return 0.0
            
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device) / 255.0
        actions_t = torch.LongTensor(actions).unsqueeze(-1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device) / 255.0
        dones_t = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)
        
        q_values = self.online_net(states_t)
        current_q = q_values.gather(1, actions_t)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t)
            max_next_q = next_q_values.max(1)[0].unsqueeze(-1)
            target_q = rewards_t + (1 - dones_t) * gamma * max_next_q
            
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        for param in self.online_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
                
        self.optimizer.step()
        
        return loss.item()
