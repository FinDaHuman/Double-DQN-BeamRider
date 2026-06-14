"""
preprocessing.py

Purpose:
To implement the standard Atari preprocessing pipeline from scratch using Gymnasium wrappers.
This converts the raw (210, 160, 3) RGB images into (4, 84, 84) stacked grayscale tensors
suitable for a Deep Q-Network.

Usage:
from environment.preprocessing import make_env
env = make_env("ALE/BeamRider-v5")
"""

import gymnasium as gym
import numpy as np
from collections import deque
import torch
import torchvision.transforms.functional as TF


class MaxAndSkipEnv(gym.Wrapper):
    """
    Purpose:
        Skips frames to accelerate training and applies a max-pooling operation
        over the last two frames.

    Why it is needed (Oral Defense):
        1. Frame Skipping: Humans don't react every 1/60th of a second. By skipping 4 frames,
           the agent plays faster and the RL algorithm learns more efficiently without losing
           critical decision time.
        2. Max-Pooling: Atari 2600 hardware limitations meant some sprites (like bullets)
           were only rendered every other frame. Taking the maximum pixel value over the
           last two frames ensures moving objects don't "disappear" from the agent's vision.

    Inputs:
        env: The gymnasium environment.
        skip (int): The number of frames to skip.

    Outputs:
        observation: The max-pooled frame.
    """

    def __init__(self, env=None, skip=4):
        """
        description: Initializes the MaxAndSkip wrapper.
        parameters:
            env (gym.Env): The environment to wrap.
            skip (int): The number of frames to skip.
        return values: None
        """
        super().__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        """
        description: Steps the environment, repeating the action skip times, and max-pools.
        parameters:
            action (int): The action to take.
        return values:
            tuple: (max_frame, total_reward, terminated, truncated, info)
        """
        total_reward = 0.0
        terminated = truncated = False

        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break

        # Max-pool over the last two frames
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class GrayscaleResizeEnv(gym.ObservationWrapper):
    """
    Purpose:
        Converts the RGB observation to Grayscale and resizes it to 84x84.

    Why it is needed (Oral Defense):
        The raw state space is 210x160x3 = 100,800 pixels. This is too computationally
        expensive for our CNN. Color is generally not required to play Beam Rider.
        Reducing to 84x84x1 reduces the state space to 7,056 pixels (a 93% reduction).

    Inputs:
        env: The gymnasium environment.

    Outputs:
        observation (np.ndarray): An 84x84 2D numpy array.
    """

    def __init__(self, env):
        """
        description: Initializes the Grayscale and Resize wrapper.
        parameters:
            env (gym.Env): The environment to wrap.
        return values: None
        """
        super().__init__(env)
        # Update the observation space to reflect the new dimensions
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def observation(self, obs):
        """
        description: Converts an RGB observation to grayscale and resizes it.
        parameters:
            obs (np.ndarray): The raw RGB observation.
        return values:
            np.ndarray: The processed 84x84 grayscale observation.
        """
        # Convert numpy array to torch tensor and permute to (Channels, Height,
        # Width)
        obs_tensor = torch.tensor(obs).permute(2, 0, 1)

        # Convert to grayscale: (1, 210, 160)
        obs_gray = TF.rgb_to_grayscale(obs_tensor)

        # Resize to 84x84. antialias=True prevents distortion.
        obs_resized = TF.resize(obs_gray, [84, 84], antialias=True)

        # Remove the channel dimension and convert back to numpy
        final_obs = obs_resized.squeeze(0).numpy().astype(np.uint8)
        return final_obs


class FrameStackEnv(gym.Wrapper):
    """
    Purpose:
        Stacks the last 'k' frames together along a new channel dimension.

    Why it is needed (Oral Defense):
        A single static frame does not satisfy the Markov Property because it contains
        no velocity or directional information (e.g., is the enemy moving left or right?).
        By stacking the last 4 frames, the agent can perceive motion, allowing the CNN
        to approximate the Markov Property.

    Inputs:
        env: The gymnasium environment.
        k (int): Number of frames to stack.

    Outputs:
        observation (np.ndarray): A (k, 84, 84) numpy array.
    """

    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)

        # Update observation space to account for the stacked channels
        obs_shape = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(
                k, obs_shape[0], obs_shape[1]), dtype=np.uint8)

    def reset(self, **kwargs):
        """
        description: Resets the environment and fills the stack with the initial frame.
        parameters:
            kwargs: Additional arguments for reset.
        return values:
            tuple: (stacked_observation, info)
        """
        obs, info = self.env.reset(**kwargs)
        # On reset, duplicate the first frame k times
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_ob(), info

    def step(self, action):
        """
        description: Steps the environment, repeating the action skip times, and max-pools.
        parameters:
            action (int): The action to take.
        return values:
            tuple: (max_frame, total_reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        # Stack frames along the first dimension (Channels, Height, Width)
        # PyTorch CNNs expect the channel dimension first.
        return np.stack(self.frames, axis=0)


def make_env(env_name="ALE/BeamRider-v5", skip=4, stack=4):
    """
    Purpose:
        Helper function to chain all preprocessing wrappers together.

    Inputs:
        env_name (str): Gym environment ID.
        skip (int): Frames to skip.
        stack (int): Frames to stack.

    Outputs:
        env: The fully wrapped and preprocessed gymnasium environment.
    """
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env, skip=skip)
    env = GrayscaleResizeEnv(env)
    env = FrameStackEnv(env, k=stack)
    return env
