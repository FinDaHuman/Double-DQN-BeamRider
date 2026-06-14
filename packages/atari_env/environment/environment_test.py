"""
environment_test.py

Purpose:
To verify the Gymnasium Atari BeamRider environment is correctly installed and to analyze its
raw properties (observation space, action space, etc.) before preprocessing.

Inputs:
None (Runs standalone)

Outputs:
Prints environment specifications to the console.

Usage:
python environment_test.py
"""

import gymnasium as gym


def main():
    """
    description: Initializes the BeamRider environment, takes random actions for a few steps,
                 and prints out the structural properties of the state and action spaces.
    parameters: None
    return values: None
    """
    env_name = "ALE/BeamRider-v5"
    print(f"Loading environment: {env_name}")

    # We use v5 which is standard for modern gymnasium Atari.
    # v5 includes sticky actions by default for stochasticity.
    env = gym.make(env_name)

    observation, info = env.reset()

    print("-" * 40)
    print("Environment Analysis:")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print(f"Action Meanings: {env.unwrapped.get_action_meanings()}")
    print("-" * 40)

    print("Running random agent for 50 steps...")
    total_reward = 0.0

    for step in range(50):
        # Sample a random action
        action = env.action_space.sample()

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if step < 5:
            print(
                f"Step {step+1} | Action: {action} | Reward: {reward} | Done: {terminated or truncated}")

        if terminated or truncated:
            print("Episode ended early.")
            break

    print(f"Total reward accumulated over 50 steps: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()
