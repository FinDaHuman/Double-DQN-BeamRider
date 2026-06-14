"""
generate_plots.py
Extracts data from TensorBoard logs and generates PNG plots for the final report.
"""
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalars(log_dir, scalar_name):
    """Extracts a specific scalar from all tfevents files in a directory."""
    try:
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        if scalar_name in event_acc.Tags()['scalars']:
            events = event_acc.Scalars(scalar_name)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            return steps, values
    except Exception as e:
        print(f"Error reading {log_dir}: {e}")
    return [], []

def smooth(scalars, weight=0.85):  
    """Exponential moving average smoothing, similar to TensorBoard."""
    last = scalars[0]  
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  
        smoothed.append(smoothed_val)                        
        last = smoothed_val                                  
    return smoothed

def main():
    logs_base_dir = "logs"
    reports_dir = "../../docs/reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # We want to plot the main Double DQN run
    # Find the most recent doubledqn directory
    run_dirs = [os.path.join(logs_base_dir, d) for d in os.listdir(logs_base_dir) if "doubledqn" in d]
    if not run_dirs:
        print("No Double DQN logs found.")
        return
        
    main_run_dir = max(run_dirs, key=os.path.getmtime)
    print(f"Extracting data from: {main_run_dir}")

    # Extract Rewards
    steps_reward, rewards = extract_scalars(main_run_dir, "Train/Episode_Reward")
    
    if steps_reward:
        plt.figure(figsize=(10, 6))
        plt.plot(steps_reward, rewards, alpha=0.3, color='blue', label='Raw Reward')
        plt.plot(steps_reward, smooth(rewards), color='blue', linewidth=2, label='Smoothed Reward')
        plt.title('Training Learning Curve: Episode Reward')
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(reports_dir, "learning_curves.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved reward plot to {save_path}")
        plt.close()
    else:
        print("Could not find Train/Episode_Reward data.")

if __name__ == "__main__":
    main()
