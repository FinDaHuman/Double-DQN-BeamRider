"""
record_video.py
Records a video of the trained agent playing Beam Rider.
"""
import os
import sys
import torch
import gymnasium as gym

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from configs.dqn_config import CONFIG
from agents.double_dqn_agent import DoubleDQNAgent
from environment.preprocessing import MaxAndSkipEnv, GrayscaleResizeEnv, FrameStackEnv

def make_video_env(env_name, skip, stack, video_folder):
    # Enable rgb_array rendering to capture the game screen
    env = gym.make(env_name, render_mode="rgb_array")
    
    # Wrap with RecordVideo *before* the other wrappers so it records the original RGB 
    # screen, rather than the 84x84 grayscale max-pooled stacked arrays.
    env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda x: True)
    
    # Re-apply the normal preprocessing wrappers so the agent gets the right inputs
    env = MaxAndSkipEnv(env, skip=skip)
    env = GrayscaleResizeEnv(env)
    env = FrameStackEnv(env, k=stack)
    
    return env

def main():
    model_path = "models/dqn_final.pth"
    video_dir = "gameplay_videos"
    os.makedirs(video_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return
        
    print(f"Initializing video recording environment for {CONFIG['env_name']}...")
    env = make_video_env(CONFIG["env_name"], CONFIG["frame_skip"], CONFIG["frame_stack"], video_dir)
    
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the agent
    agent = DoubleDQNAgent(state_shape, num_actions, device=device)
    agent.online_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.online_net.eval()
    
    state, _ = env.reset()
    epsilon = 0.01  # 1% exploration, 99% pure exploitation
    
    print(f"Recording 3 episodes to find the best attempt (Epsilon: {epsilon})...")
    best_reward = -float('inf')
    best_episode = 0
    
    for ep in range(3):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            action = agent.select_action(state, epsilon)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
        print(f"Episode {ep} finished with Reward: {total_reward}")
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = ep
            
    env.close()
    print(f"Finished recording. The BEST attempt was Episode {best_episode} with Reward: {best_reward}")
    print(f"Videos saved to the '{video_dir}/' directory. Use 'rl-video-episode-{best_episode}.mp4' for your presentation.")

if __name__ == "__main__":
    main()
