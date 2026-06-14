"""
train.py

Purpose:
To act as the main entry point for running the baseline DQN training process.
It relies on the strictly structured Trainer class in src/training/trainer.py.

Usage:
python train.py
"""

import os
import sys

# Ensure src directory is in the path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.join(REPO_ROOT, 'packages', 'rl_core'))
sys.path.append(os.path.join(REPO_ROOT, 'packages', 'atari_env'))
sys.path.append(os.path.dirname(__file__))

from configs.dqn_config import CONFIG
from training.trainer import Trainer

def main():
    print(f"Loading Configuration for {CONFIG['env_name']}...")
    trainer = Trainer(CONFIG)
    trainer.train()

if __name__ == "__main__":
    main()
