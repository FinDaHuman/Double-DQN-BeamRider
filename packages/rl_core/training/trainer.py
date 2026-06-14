"""
trainer.py

Contains the training engine that orchestrates the environment, agent, and replay buffer.
"""

import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from environment.preprocessing import make_env
from replay.replay_buffer import ReplayBuffer
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent


class Trainer:
    """
    Purpose:
        To encapsulate the training loop logic, separating the orchestration layer
        from the executable script. This adheres to the architectural separation rules.

    Inputs:
        config (dict): The hyperparameter configuration dictionary.

    Outputs:
        An instantiated Trainer object.

    Usage:
        trainer = Trainer(CONFIG)
        trainer.train()
    """

    def __init__(self, config: dict):
        """
        description: Initializes the training environment, agent, replay buffer, and loggers.
        parameters:
            config (dict): Dictionary containing all training hyperparameters.
        return values: None
        """
        self.config = config

        self.env = make_env(
            config["env_name"],
            skip=config["frame_skip"],
            stack=config["frame_stack"]
        )

        state_shape = self.env.observation_space.shape
        num_actions = self.env.action_space.n

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Trainer initialized using device: {self.device}")

        agent_type = config.get("agent_type", "DQN")
        print(f"Instantiating Agent Type: {agent_type}")
        if agent_type == "DoubleDQN":
            self.agent = DoubleDQNAgent(
                state_shape,
                num_actions,
                lr=config["learning_rate"],
                device=self.device)
        else:
            self.agent = DQNAgent(
                state_shape,
                num_actions,
                lr=config["learning_rate"],
                device=self.device)

        self.replay_buffer = ReplayBuffer(config["replay_capacity"])

        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=f"logs/{agent_type.lower()}_{int(time.time())}")

    def train(self):
        """
        description: Executes the main RL training loop for the total configured timesteps.
        parameters: None
        return values: None
        """
        state, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        epsilon = self.config["epsilon_start"]

        print(
            f"Beginning training for {self.config['total_timesteps']} timesteps.")

        for step in tqdm(
                range(
                    1,
                    self.config["total_timesteps"] +
                    1),
                desc="Training"):

            # Decay Epsilon
            epsilon = max(
                self.config["epsilon_end"],
                self.config["epsilon_start"] -
                step /
                self.config["epsilon_decay_steps"] *
                (
                    self.config["epsilon_start"] -
                    self.config["epsilon_end"]))

            action = self.agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(
                action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if done:
                self.writer.add_scalar(
                    "Train/Episode_Reward", episode_reward, step)
                self.writer.add_scalar(
                    "Train/Episode_Length", episode_length, step)
                self.writer.add_scalar("Train/Epsilon", epsilon, step)

                state, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

            if step > self.config["learning_starts"] and step % self.config["train_frequency"] == 0:
                loss = self.agent.train_step(
                    self.replay_buffer,
                    self.config["batch_size"],
                    self.config["gamma"])
                if step % (self.config["train_frequency"] * 10) == 0:
                    self.writer.add_scalar("Train/Loss", loss, step)

            if step % self.config["target_update_freq"] == 0:
                self.agent.update_target_network()

            if step % 250_000 == 0:
                torch.save(
                    self.agent.online_net.state_dict(),
                    f"models/dqn_step_{step}.pth")

        save_name = self.config.get("model_save_name", "dqn_final.pth")
        torch.save(self.agent.online_net.state_dict(), f"models/{save_name}")
        print(f"Training Complete. Model saved as {save_name}.")
        self.env.close()
        self.writer.close()
