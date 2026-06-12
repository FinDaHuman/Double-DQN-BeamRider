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
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from configs.dqn_config import CONFIG
from training.trainer import Trainer

def main():
    print(f"Loading Configuration for {CONFIG['env_name']}...")
    trainer = Trainer(CONFIG)
    trainer.train()

if __name__ == "__main__":
    main()
