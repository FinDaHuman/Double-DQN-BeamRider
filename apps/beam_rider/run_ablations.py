"""
run_ablations.py

Purpose:
To automate Phase 7 (Ablation Studies). This script runs short training sessions (150,000 steps)
with key algorithmic components disabled one by one. By observing the failure or instability
of the agent in these crippled states, we scientifically prove the necessity of each component.

Usage:
python run_ablations.py
"""

from training.trainer import Trainer
from configs.dqn_config import CONFIG
import os
import sys
import copy
from datetime import datetime

# Ensure src directory is in the path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.join(REPO_ROOT, 'packages', 'rl_core'))
sys.path.append(os.path.join(REPO_ROOT, 'packages', 'atari_env'))
sys.path.append(os.path.dirname(__file__))


def run_ablation(name, modified_config):
    print(f"\n{'='*50}")
    print(f"Starting Ablation Study: {name}")
    print(f"{'='*50}")

    # Initialize trainer with the crippled config
    trainer = Trainer(modified_config)
    trainer.train()

    # We don't necessarily need to evaluate heavily, the training logs (TensorBoard)
    # will show the divergence or failure to learn.
    print(f"Finished Ablation: {name}\n")


def main():
    ablation_steps = 150_000

    # Base config for ablations (shorter duration, faster epsilon decay)
    base_ablation_config = copy.deepcopy(CONFIG)
    base_ablation_config["total_timesteps"] = ablation_steps
    # Decay faster for short run
    base_ablation_config["epsilon_decay_steps"] = 75_000
    base_ablation_config["learning_starts"] = 10_000

    os.makedirs("../../docs/reports", exist_ok=True)
    report_path = "../../docs/reports/ablation_study.md"

    # Initialize the report file
    with open(report_path, "w") as f:
        f.write("# Phase 7: Ablation Studies\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Duration per ablation:** {ablation_steps} timesteps.\n\n")
        f.write(
            "This document summarizes the experiments demonstrating the necessity of core DQN components.\n\n")

    # --- 1. Ablation: No Target Network ---
    # We simulate 'No Target Network' by updating the target network every single step,
    # effectively making it identical to the online network at all times.
    config_no_target = copy.deepcopy(base_ablation_config)
    config_no_target["target_update_freq"] = 1
    config_no_target["model_save_name"] = "ablation_no_target.pth"
    # Use standard DQN to isolate the target network effect
    config_no_target["agent_type"] = "DQN"
    run_ablation("No Target Network", config_no_target)

    with open(report_path, "a") as f:
        f.write("## 1. Without Target Network\n")
        f.write(
            "- **Method:** Target network updated every step (`target_update_freq = 1`).\n")
        f.write("- **Expected Result:** High instability, catastrophic divergence due to the moving target problem.\n")
        f.write("- **Actual Observation:** (Check TensorBoard `Train/Episode_Reward` and `Train/Loss` curves and document here).\n\n")

    # --- 2. Ablation: No Experience Replay ---
    # We simulate 'No Replay' by making the buffer capacity equal to the batch size.
    # It will only train on the absolute most recent transitions, breaking the
    # i.i.d assumption.
    config_no_replay = copy.deepcopy(base_ablation_config)
    config_no_replay["replay_capacity"] = config_no_replay["batch_size"]
    config_no_replay["model_save_name"] = "ablation_no_replay.pth"
    config_no_replay["agent_type"] = "DQN"
    run_ablation("No Experience Replay", config_no_replay)

    with open(report_path, "a") as f:
        f.write("## 2. Without Experience Replay\n")
        f.write("- **Method:** Replay buffer capacity reduced to the batch size (32), forcing on-policy updates with highly correlated data.\n")
        f.write("- **Expected Result:** Catastrophic forgetting, agent forgets how to play states it hasn't seen in the last few seconds.\n")
        f.write("- **Actual Observation:** (Check TensorBoard and document here).\n\n")

    # --- 3. Ablation: No Frame Stacking ---
    # We pass only 1 frame instead of 4. The agent is now blind to velocity
    # and direction.
    config_no_stack = copy.deepcopy(base_ablation_config)
    config_no_stack["frame_stack"] = 1
    config_no_stack["model_save_name"] = "ablation_no_stack.pth"
    config_no_stack["agent_type"] = "DQN"
    run_ablation("No Frame Stacking", config_no_stack)

    with open(report_path, "a") as f:
        f.write("## 3. Without Frame Stacking\n")
        f.write("- **Method:** Changed `frame_stack` from 4 to 1.\n")
        f.write("- **Expected Result:** Agent fails to avoid moving enemies/bullets because a single static frame violates the Markov Property (cannot perceive motion).\n")
        f.write("- **Actual Observation:** (Check TensorBoard and document here).\n\n")

    print("\nAll ablations completed! Please check TensorBoard to analyze the training curves.")
    print("Run: tensorboard --logdir=logs/")
    print(f"Then document your findings in {report_path}")


if __name__ == "__main__":
    main()
