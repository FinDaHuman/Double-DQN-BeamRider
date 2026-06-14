"""
evaluator.py

Contains the Evaluation engine to load trained models and run test episodes.
"""

import torch
import numpy as np
from tqdm import tqdm

from environment.preprocessing import make_env
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent


class Evaluator:
    """
    Purpose:
        To encapsulate the evaluation logic. This class loads a saved model checkpoint,
        disables training (exploration is minimized), and runs a set number of episodes
        to accurately measure the agent's actual learned policy performance.

    Why it is needed (Oral Defense):
        During training, the agent explores using a decaying epsilon value. The rewards
        achieved during training include random, suboptimal actions. Evaluation runs
        the agent strictly on its learned Q-values (pure exploitation or near-pure
        with very small epsilon like 0.01) to provide an unbiased measurement of its capability.

    Inputs:
        config (dict): The hyperparameter configuration dictionary.
        model_path (str): The path to the saved model weights (.pth file).

    Outputs:
        An instantiated Evaluator object.

    Usage:
        evaluator = Evaluator(CONFIG, "models/dqn_final.pth")
        metrics = evaluator.evaluate(num_episodes=10)
    """

    def __init__(self, config: dict, model_path: str):
        """
        description: Initializes the environment and agent for evaluation, loading saved weights.
        parameters:
            config (dict): Dictionary containing all training hyperparameters.
            model_path (str): Path to the saved network weights.
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

        agent_type = config.get("agent_type", "DQN")
        if agent_type == "DoubleDQN":
            self.agent = DoubleDQNAgent(
                state_shape, num_actions, device=self.device)
        else:
            self.agent = DQNAgent(state_shape, num_actions, device=self.device)

        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.agent.online_net.load_state_dict(state_dict)
        self.agent.online_net.eval()  # Set network to evaluation mode

    def evaluate(self, num_episodes: int = 10, epsilon: float = 0.05) -> dict:
        """
        description: Runs multiple evaluation episodes to calculate performance metrics.
        parameters:
            num_episodes (int): The number of episodes to run.
            epsilon (float): Evaluation epsilon (small value to prevent getting stuck).
        return values:
            dict: A dictionary containing Average, Max, Min Reward, and Standard Deviation.
        """
        episode_rewards = []

        for ep in tqdm(range(num_episodes), desc="Evaluating"):
            state, _ = self.env.reset()
            done = False
            total_reward = 0.0

            while not done:
                # Use the agent's select_action, but with the fixed evaluation
                # epsilon
                action = self.agent.select_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                done = terminated or truncated

                total_reward += reward
                state = next_state

            episode_rewards.append(total_reward)

        self.env.close()

        metrics = {
            "average_reward": np.mean(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "std_deviation": np.std(episode_rewards)
        }

        return metrics
