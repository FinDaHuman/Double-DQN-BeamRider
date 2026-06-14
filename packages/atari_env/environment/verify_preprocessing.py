"""
verify_preprocessing.py

Purpose:
To visually verify that the preprocessing pipeline (Grayscale, Resize, Skip, Stack)
is working correctly by comparing a raw environment frame with the processed stacked frames.

Usage:
python src/environment/verify_preprocessing.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
from preprocessing import make_env
import os


def main():
    print("Verifying preprocessing pipeline...")

    # 1. Get a raw frame
    raw_env = gym.make("ALE/BeamRider-v5")
    raw_obs, _ = raw_env.reset()
    for _ in range(10):
        action = raw_env.action_space.sample()
        raw_obs, _, terminated, truncated, _ = raw_env.step(action)
        if terminated or truncated:
            raw_obs, _ = raw_env.reset()
    raw_env.close()

    # 2. Get a processed frame (stacked)
    processed_env = make_env("ALE/BeamRider-v5", skip=4, stack=4)
    processed_obs, _ = processed_env.reset()
    for _ in range(10):
        action = processed_env.action_space.sample()
        processed_obs, _, terminated, truncated, _ = processed_env.step(action)
        if terminated or truncated:
            processed_obs, _ = processed_env.reset()
    processed_env.close()

    print(f"Raw observation shape: {raw_obs.shape}")
    print(f"Processed observation shape: {processed_obs.shape}")
    print(
        f"Processed observation max value: {processed_obs.max()}, min: {processed_obs.min()}")

    # 3. Plot the results
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))

    # Plot raw frame
    axes[0].imshow(raw_obs)
    axes[0].set_title("Raw RGB Frame\\n(210x160x3)")
    axes[0].axis('off')

    # Plot the 4 stacked frames
    for i in range(4):
        axes[i + 1].imshow(processed_obs[i], cmap='gray')
        axes[i + 1].set_title(f"Processed Stack {i+1}\\n(84x84)")
        axes[i + 1].axis('off')

    plt.tight_layout()

    # Save the plot
    os.makedirs('reports', exist_ok=True)
    save_path = 'reports/preprocessing_verification.png'
    plt.savefig(save_path)
    print(f"Visual verification saved to: {save_path}")


if __name__ == "__main__":
    main()
