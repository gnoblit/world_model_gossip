import pytest
import numpy as np

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

def test_sample_sequences_shape():
    """Tests if sampled sequences have the correct shape."""
    buffer = ReplayBuffer(capacity=100)
    seq_len = 10
    batch_size = 4
    
    # Populate with one long episode
    for i in range(50):
        done = (i == 49)
        buffer.push(np.random.rand(1, *config.RESIZE_DIM), np.random.rand(config.ACTION_DIM), 1.0, None, done)
    
    batch = buffer.sample_sequences(batch_size, seq_len)
    assert batch is not None, "Sampling failed with enough data."
    obs_b, act_b, rew_b = batch
    
    assert obs_b.shape == (batch_size, seq_len, 1, *config.RESIZE_DIM)
    assert act_b.shape == (batch_size, seq_len, config.ACTION_DIM)
    assert rew_b.shape == (batch_size, seq_len, 1)

def test_sample_sequences_avoids_episode_boundary():
    """
    Ensures that sampled sequences do not cross episode boundaries by checking
    the 'done' flags of the returned sequences.
    """
    buffer = ReplayBuffer(capacity=100)
    seq_len = 10
    batch_size = 20 # Sample many times to increase chance of catching errors

    # Create two episodes of 20 steps each.
    # We'll use the observation `i` to track the step number.
    for i in range(20):
        buffer.push(np.array([i]), 0, 0, 0, done=(i == 19))
    for i in range(20, 40):
        buffer.push(np.array([i]), 0, 0, 0, done=(i == 39))

    # The buffer now contains transitions 0..19 (ep 1) and 20..39 (ep 2).
    # The `done` flag is True at indices 19 and 39.

    # We need to sample from the original memory to verify
    sampled_batch = []
    # In a small buffer, we might not find enough valid sequences for a large batch.
    # So we sample multiple times.
    for _ in range(50): # Run the sampling 50 times
        batch = buffer.sample_sequences(batch_size, seq_len)
        if batch:
            # We only care about the observation sequence for this test
            obs_seq, _, _ = batch
            sampled_batch.append(obs_seq.cpu().numpy())

    assert len(sampled_batch) > 0, "Failed to sample any valid batches."

    # Concatenate all sampled sequences for easier checking
    all_sampled_obs = np.concatenate(sampled_batch, axis=0)

    for seq in all_sampled_obs:
        # The 'observation' in our dummy data is just the step number.
        # Check if the sequence crosses an episode boundary.
        # e.g., a sequence like [15, 16, 17, 18, 19, 20, 21, 22, 23, 24] is invalid.
        # This is invalid because the transition from step 19 to 20 crosses an episode.
        
        # The 'done' flag applies to the transition *from* the current obs *to* the next.
        # Therefore, a sequence is invalid if any of its first `seq_len - 1` steps
        # correspond to a 'done' transition in the buffer.
        
        # Let's check the step numbers (which we stored as obs)
        step_numbers = seq.flatten().astype(int)
        
        # A sequence is valid if it's monotonically increasing by 1.
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