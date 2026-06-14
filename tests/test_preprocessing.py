import numpy as np
from atari_env.environment.preprocessing import make_env

def test_env_wrapping():
    """Test that the custom Gym wrappers correctly shape the Atari observation space."""
    env = make_env("ALE/BeamRider-v5", skip=4, stack=4)
    obs, info = env.reset()
    
    # Check shape: (Stack, Height, Width)
    assert obs.shape == (4, 84, 84), f"Expected shape (4, 84, 84), got {obs.shape}"
    assert obs.dtype == np.uint8, f"Expected dtype uint8, got {obs.dtype}"
    
    # Check step
    next_obs, reward, terminated, truncated, info = env.step(0)
    assert next_obs.shape == (4, 84, 84), "Step observation shape mismatch"
    
    env.close()
