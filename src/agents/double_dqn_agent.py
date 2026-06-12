"""
double_dqn_agent.py

Contains the Double DQN Agent logic, extending the baseline DQN.
"""

import torch
import numpy as np

from agents.dqn_agent import DQNAgent
from replay.replay_buffer import ReplayBuffer

class DoubleDQNAgent(DQNAgent):
    """
    Purpose:
        To implement the Double Deep Q-Network (Double DQN) Agent.
        This extension addresses the overestimation bias present in standard Q-Learning.
        
    Why it is needed (Oral Defense):
        Standard DQN uses the max Q-value from the Target Network for the next state. 
        Because it takes the maximum over noisy estimates, it systematically overestimates 
        the true value of states. 
        Double DQN solves this by decoupling action selection from evaluation:
        - The Online Network SELECTS the best next action.
        - The Target Network EVALUATES the value of that specific action.
        
    Inputs:
        state_shape (tuple): The shape of the preprocessed state.
        num_actions (int): The number of available actions.
        lr (float): The learning rate for the optimizer.
        device (str): Device to run computation on ("cpu" or "cuda").
        
    Outputs:
        An instantiated DoubleDQNAgent object.
        
    Usage:
        agent = DoubleDQNAgent(state_shape, num_actions)
        action = agent.select_action(state, epsilon)
        loss = agent.train_step(replay_buffer, batch_size, gamma)
    """
    
    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int, gamma: float) -> float:
        """
        description: Samples a batch from the replay buffer and performs the Double DQN Bellman update.
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
        
        # 1. Calculate Current Q-Values: Q_online(s, a)
        q_values = self.online_net(states_t)
        current_q = q_values.gather(1, actions_t)
        
        # 2. Calculate Target Q-Values using Double DQN logic
        with torch.no_grad():
            # a) Use ONLINE network to SELECT the best action for the next state
            next_q_online = self.online_net(next_states_t)
            best_next_actions = next_q_online.argmax(dim=1).unsqueeze(-1)
            
            # b) Use TARGET network to EVALUATE the value of that selected action
            next_q_target = self.target_net(next_states_t)
            double_q_value = next_q_target.gather(1, best_next_actions)
            
            # Target = R + gamma * Q_target(s', argmax Q_online(s', a))
            target_q = rewards_t + (1 - dones_t) * gamma * double_q_value
            
        # 3. Compute Loss and Backpropagate
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        for param in self.online_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
                
        self.optimizer.step()
        
        return loss.item()
