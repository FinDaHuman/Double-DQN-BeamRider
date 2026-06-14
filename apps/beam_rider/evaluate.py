"""
evaluate.py

Purpose:
To act as the main entry point for evaluating a trained DQN or Double DQN agent.
It calculates and prints Average, Max, Min Reward, and Standard Deviation as requested
in Phase 6 of the Project Plan.

Usage:
python evaluate.py
"""

from evaluation.evaluator import Evaluator
from configs.dqn_config import CONFIG
import os
import sys

# Ensure src directory is in the path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.join(REPO_ROOT, 'packages', 'rl_core'))
sys.path.append(os.path.join(REPO_ROOT, 'packages', 'atari_env'))
sys.path.append(os.path.dirname(__file__))


def main():
    model_path = "models/dqn_final.pth"
    num_episodes = 10
    eval_epsilon = 0.05

    if not os.path.exists(model_path):
        print(
            f"Error: Model file '{model_path}' not found. Please train the model first.")
        return

    print(f"Evaluating model: {model_path}")
    print(f"Environment: {CONFIG['env_name']}, Agent: {CONFIG['agent_type']}")
    print(f"Running {num_episodes} episodes with epsilon = {eval_epsilon}...")

    evaluator = Evaluator(CONFIG, model_path)
    metrics = evaluator.evaluate(
        num_episodes=num_episodes,
        epsilon=eval_epsilon)

    print("\n--- Evaluation Results ---")
    print(f"Average Reward:     {metrics['average_reward']:.2f}")
    print(f"Maximum Reward:     {metrics['max_reward']:.2f}")
    print(f"Minimum Reward:     {metrics['min_reward']:.2f}")
    print(f"Standard Deviation: {metrics['std_deviation']:.2f}")
    print("--------------------------\n")

    # Save the results to the markdown report
    report_path = "../../docs/reports/evaluation_report.md"
    os.makedirs("../../docs/reports", exist_ok=True)

    with open(report_path, "a") as f:
        f.write(f"## Evaluation: {CONFIG['agent_type']} (Final Model)\n")
        f.write(f"- **Episodes:** {num_episodes}\n")
        f.write(f"- **Epsilon:** {eval_epsilon}\n")
        f.write(f"- **Average Reward:** {metrics['average_reward']:.2f}\n")
        f.write(f"- **Max Reward:** {metrics['max_reward']:.2f}\n")
        f.write(f"- **Min Reward:** {metrics['min_reward']:.2f}\n")
        f.write(f"- **Std Deviation:** {metrics['std_deviation']:.2f}\n\n")

    print(f"Results appended to {report_path}")


if __name__ == "__main__":
    main()
