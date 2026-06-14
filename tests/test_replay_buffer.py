import numpy as np
from rl_core.replay.replay_buffer import ReplayBuffer


def test_replay_buffer_push_and_sample():
    """Test that the replay buffer correctly stores and samples transitions."""
    buffer = ReplayBuffer(capacity=10)
    state = np.zeros((4, 84, 84))
    next_state = np.ones((4, 84, 84))

    # Push 5 transitions
    for _ in range(5):
        buffer.push(state, 1, 1.0, next_state, False)

    assert len(buffer) == 5, "Buffer length should be 5"

    states, actions, rewards, next_states, dones = buffer.sample(3)
    assert states.shape == (3, 4, 84, 84), "State batch shape mismatch"
    assert actions.shape == (3, 1), "Action batch shape mismatch"
    assert rewards.shape == (3, 1), "Reward batch shape mismatch"
    assert next_states.shape == (
        3, 4, 84, 84), "Next State batch shape mismatch"
    assert dones.shape == (3, 1), "Done batch shape mismatch"


def test_replay_buffer_capacity():
    """Test that the buffer correctly overwrites old transitions when capacity is reached."""
    buffer = ReplayBuffer(capacity=5)
    state = np.zeros((4, 84, 84))
    next_state = np.ones((4, 84, 84))

    # Push 10 transitions (exceeds capacity of 5)
    for _ in range(10):
        buffer.push(state, 1, 1.0, next_state, False)

    assert len(buffer) == 5, "Buffer length should not exceed capacity"
