import pytest
import numpy as np
import torch
import gymnasium as gym

from gossip_wm.replay_buffer import ReplayBuffer
from gossip_wm import config

def test_buffer_init():
    """Tests buffer initialization."""
    buffer = ReplayBuffer(capacity=100)
    assert len(buffer) == 0
    assert buffer.memory.maxlen == 100

def test_buffer_push_and_len():
    """Tests pushing items and checking the length."""
    buffer = ReplayBuffer(capacity=10)
    for i in range(5):
        buffer.push(f"obs_{i}", "action", 0.0, "next_obs", False)
    assert len(buffer) == 5

def test_buffer_capacity():
    """Tests that the buffer respects its capacity limit."""
    capacity = 10
    buffer = ReplayBuffer(capacity=capacity)
    for i in range(capacity + 5):
        # The 'obs' is the unique identifier for the transition
        buffer.push(f"obs_{i}", "action", 0.0, "next_obs", False)
    
    assert len(buffer) == capacity
    # Check that the first element is now 'obs_5' because the first 5 were pushed out
    assert buffer.memory[0][0] == "obs_5"

@pytest.mark.parametrize("action_space_def", [
    ("continuous", gym.spaces.Box(low=-1, high=1, shape=(3,))),
    ("discrete_small", gym.spaces.Discrete(7)),
    ("discrete_large", gym.spaces.Discrete(18)) # For Craftium
])
def test_sample_sequences_shape(action_space_def):
    """Tests if sampled sequences have the correct shape for different action spaces."""
    action_type, action_space = action_space_def
    action_shape = action_space.shape if action_type.startswith("continuous") else ()

    buffer = ReplayBuffer(capacity=100)
    seq_len = 10
    batch_size = 4
    
    # Populate with one long episode
    for i in range(50):
        done = (i == 49)
        action = action_space.sample()
        buffer.push(np.random.rand(1, *config.RESIZE_DIM), action, 1.0, None, done)
    
    batch = buffer.sample_sequences(batch_size, seq_len)
    assert batch is not None, "Sampling failed with enough data."
    obs_b, act_b, rew_b = batch
    
    assert obs_b.shape == (batch_size, seq_len, 1, *config.RESIZE_DIM)
    assert act_b.shape == (batch_size, seq_len, *action_shape)
    assert rew_b.shape == (batch_size, seq_len, 1)

def test_sample_sequences_avoids_episode_boundary():
    """
    Ensures that sampled sequences do not cross episode boundaries by checking
    the 'done' flags of the returned sequences.
    """
    buffer = ReplayBuffer(capacity=100)
    seq_len = 10
    batch_size = 5 # Sample a few times to increase chance of catching errors

    # Create two episodes of 20 steps each.
    # We'll use the observation `i` to track the step number.
    for i in range(20):
        buffer.push(np.array([i]), np.array([0]), 0.0, 0, done=(i == 19))
    for i in range(20, 40):
        buffer.push(np.array([i]), np.array([0]), 0.0, 0, done=(i == 39))

    # The buffer now contains transitions 0..19 (ep 1) and 20..39 (ep 2).
    # The `done` flag is True at indices 19 and 39.
    
    # Run the sampling 50 times to be thorough
    for _ in range(50):
        batch = buffer.sample_sequences(batch_size, seq_len)
        if batch is None:
            # This can happen if not enough valid sequences are found
            continue

        obs_seq, _, _ = batch
        all_sampled_obs = obs_seq.cpu().numpy()

        for seq in all_sampled_obs:
            # The 'observation' in our dummy data is just the step number.
            step_numbers = seq.flatten().astype(int)
            
            # A valid sequence should be contiguous (e.g., [5, 6, 7...]).
            # If it crosses an episode, it will jump, e.g., from 19 to 20.
            # Our sampling method is supposed to prevent this.
            is_contiguous = np.all(np.diff(step_numbers) == 1)
            assert is_contiguous, f"Sequence is not contiguous, it might have crossed an episode boundary: {step_numbers}"
        
def test_sample_returns_none_if_not_enough_data():
    """Tests that sampling returns None if the buffer is too small."""
    buffer = ReplayBuffer(capacity=100)
    # Populate with fewer steps than sequence length
    for _ in range(config.SEQUENCE_LENGTH - 1):
        buffer.push(0,0,0,0,False)
        
    assert buffer.sample_sequences(config.BATCH_SIZE, config.SEQUENCE_LENGTH) is None

def test_sample_transitions_shape_and_type():
    """Tests the shape and type of single transition samples."""
    buffer = ReplayBuffer(capacity=100)
    batch_size = config.BATCH_SIZE
    action_dim = 3
    
    # Populate buffer
    for _ in range(batch_size * 2):
        buffer.push(
            np.random.rand(1, *config.RESIZE_DIM), 
            np.random.rand(action_dim), 
            1.0, 
            np.random.rand(1, *config.RESIZE_DIM), 
            False
        )
    
    batch = buffer.sample_transitions(batch_size)
    assert batch is not None
    obs_b, act_b, rew_b, next_obs_b, done_b = batch

    assert isinstance(obs_b, torch.Tensor)
    assert obs_b.shape == (batch_size, 1, *config.RESIZE_DIM)
    assert obs_b.device.type == config.DEVICE.type

    assert isinstance(act_b, torch.Tensor)
    assert act_b.shape == (batch_size, action_dim)
    assert act_b.device.type == config.DEVICE.type

    assert isinstance(rew_b, torch.Tensor)
    assert rew_b.shape == (batch_size, 1)

    assert isinstance(next_obs_b, torch.Tensor)
    assert next_obs_b.shape == (batch_size, 1, *config.RESIZE_DIM)

    assert isinstance(done_b, torch.Tensor)
    assert done_b.shape == (batch_size, 1)