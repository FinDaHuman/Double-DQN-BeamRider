import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder, DummyVecEnv
from sb3_contrib import RecurrentPPO
import numpy as np

ENV_ID = "ALE/BeamRider-v5"
MODEL_PATH = "./models/sb3_lstm_final"
VIDEO_FOLDER = "./assets/gameplay_videos/sb3_lstm/"
NUM_EPISODES = 3

def main():
    print(f"Loading environment {ENV_ID}...")
    # Setup environment exactly as in training
    env = make_atari_env(ENV_ID, n_envs=1, seed=42, vec_env_cls=DummyVecEnv)
    env = VecFrameStack(env, n_stack=4)
    
    # Wrap with Video Recorder
    env = VecVideoRecorder(
        env,
        VIDEO_FOLDER,
        record_video_trigger=lambda x: x == 0, # Record the first episode
        video_length=5000, # Max length of the video
        name_prefix="lstm-agent"
    )

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = RecurrentPPO.load(MODEL_PATH, env=env)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Starting evaluation...")
    episode_rewards = []
    
    for ep in range(NUM_EPISODES):
        obs = env.reset()
        # LSTM hidden states
        lstm_states = None
        # Episode start signals are used to reset the lstm states
        episode_starts = np.ones((1,), dtype=bool)
        
        done = False
        total_reward = 0
        
        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            episode_starts = dones
            total_reward += rewards[0]
            
            if dones[0]:
                done = True
                
        print(f"Episode {ep + 1}: Reward = {total_reward}")
        episode_rewards.append(total_reward)

    env.close()
    
    print("\n--- Evaluation Results ---")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Video saved to: {VIDEO_FOLDER}")

if __name__ == "__main__":
    main()
