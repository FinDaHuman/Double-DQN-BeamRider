import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# Hardware-friendly configuration for a low-end laptop
ENV_ID = "ALE/BeamRider-v5"
NUM_ENVS = 2  # Low number of parallel environments to save CPU/RAM
SEED = 42

def main():
    print(f"Initializing {NUM_ENVS} environments for low-end hardware...")
    
    # We use DummyVecEnv instead of SubprocVecEnv. 
    # SubprocVecEnv creates separate Python processes which can freeze a low-end laptop.
    # DummyVecEnv runs everything sequentially in the same process, which is slower 
    # wall-clock time but much safer for limited RAM and CPU cores.
    # 'make_atari_env' automatically applies standard wrappers (Grayscale, Resize 84x84, MaxPool).
    env = make_atari_env(ENV_ID, n_envs=NUM_ENVS, seed=SEED, vec_env_cls=DummyVecEnv)

    # Note on Frame Stacking + LSTM: 
    # While an LSTM handles temporal memory inherently, stacking frames (e.g., 4) 
    # is still commonly used to provide immediate velocity context to the CNN, 
    # freeing the LSTM to focus purely on long-term dependencies (like the boss trap).
    env = VecFrameStack(env, n_stack=4)

    print("Initializing RecurrentPPO (LSTM) model...")
    # CnnLstmPolicy gives us visual processing (CNN) + long-term memory (LSTM)
    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        # Hardware optimization: Smaller n_steps and batch_size
        n_steps=128,      # Standard is usually 128 or 256
        batch_size=64,    # Smaller batch size to prevent Out-Of-Memory (OOM) errors
        seed=SEED,
        tensorboard_log="./logs/sb3_lstm_tensorboard/"
    )

    # Ensure model directory exists
    os.makedirs('./models/sb3_lstm_checkpoints/', exist_ok=True)

    # Save a checkpoint every 50,000 steps (adjusted for fewer envs)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // NUM_ENVS, 1),
        save_path='./models/sb3_lstm_checkpoints/',
        name_prefix='lstm_beam_rider'
    )

    print("Starting training. This is safe to run on a laptop.")
    print("Press Ctrl+C to stop manually at any time. Checkpoints will be saved automatically.")
    
    # 1 Million timesteps is a good starting point to see learning progress 
    # without running the laptop for a full week.
    try:
        model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving current model state...")
    finally:
        print("Saving final model...")
        model.save("./models/sb3_lstm_final")
        print("Done!")

if __name__ == "__main__":
    main()
