"""
dqn_config.py

Purpose:
To store all hyperparameters for training the DQN agent. This avoids "magic numbers"
scattered throughout the code, strictly following Rule 4.
"""

CONFIG = {
    # Environment
    "env_name": "ALE/BeamRider-v5",
    "agent_type": "DoubleDQN",  # "DQN" or "DoubleDQN"
    "frame_skip": 4,
    "frame_stack": 4,

    # Replay Buffer
    "replay_capacity": 100000,  # Max transitions to store
    "batch_size": 32,          # Number of samples per training step

    # Agent / Network
    "learning_rate": 0.0001,   # Optimizer learning rate
    "gamma": 0.99,             # Discount factor

    # Exploration (Epsilon)
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "epsilon_decay_steps": 1000000,  # Linearly decay over 1M steps

    # Training Loop
    "total_timesteps": 2000000,  # Total frames to train on
    "learning_starts": 50000,   # Steps to fill replay buffer before training
    "train_frequency": 4,       # Train every 4 environment steps
    "target_update_freq": 10000  # Update target network every 10K steps
}
