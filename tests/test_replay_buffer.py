# tests/test_replay_buffer.py
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
        buffer.push(f"obs_{i}", "action", 0.0, "next_obs", False)
    
    assert len(buffer) == capacity
    assert buffer.memory[0][0] == "obs_5"

@pytest.mark.parametrize("action_space_def", [
    ("continuous", gym.spaces.Box(low=-1, high=1, shape=(3,))),
    ("discrete_small", gym.spaces.Discrete(7)),
    ("discrete_large", gym.spaces.Discrete(18))
])
def test_sample_sequences_shape(action_space_def):
    action_type, action_space = action_space_def
    action_shape = action_space.shape if action_type.startswith("continuous") else ()

    buffer = ReplayBuffer(capacity=100)
    seq_len = 10
    batch_size = 4
    
    # Get resize_dim from default config for creating dummy data
    env_conf = config.ENV_CONFIGS[config.ENV_NAME]
    resize_dim = env_conf['RESIZE_DIM']
    
    for i in range(50):
        done = (i == 49)
        action = action_space.sample()
        buffer.push(np.random.rand(1, *resize_dim), action, 1.0, None, done)
    
    batch = buffer.sample_sequences(batch_size, seq_len)
    assert batch is not None, "Sampling failed with enough data."
    obs_b, act_b, rew_b, done_b = batch
    
    assert obs_b.shape == (batch_size, seq_len, 1, *resize_dim)
    assert act_b.shape == (batch_size, seq_len, *action_shape)
    assert rew_b.shape == (batch_size, seq_len, 1)
    assert done_b.shape == (batch_size, seq_len, 1)

def test_sample_sequences_avoids_episode_boundary():
    """
    Ensures that sampled sequences do not cross episode boundaries.
    """
    buffer = ReplayBuffer(capacity=100)
    seq_len = 10
    batch_size = 5

    for i in range(20):
        buffer.push(np.array([i]), np.array([0]), 0.0, 0, done=(i == 19))
    for i in range(20, 40):
        buffer.push(np.array([i]), np.array([0]), 0.0, 0, done=(i == 39))

    for _ in range(50):
        batch = buffer.sample_sequences(batch_size, seq_len)
        if batch is None:
            continue
        
        # *** FIX: Unpack 4 values now ***
        obs_seq, _, _, _ = batch
        all_sampled_obs = obs_seq.cpu().numpy()

        for seq in all_sampled_obs:
            step_numbers = seq.flatten().astype(int)
            is_contiguous = np.all(np.diff(step_numbers) == 1)
            assert is_contiguous, f"Sequence crossed episode boundary: {step_numbers}"
        
def test_sample_returns_none_if_not_enough_data():
    """Tests that sampling returns None if the buffer is too small."""
    buffer = ReplayBuffer(capacity=100)
    for _ in range(config.SEQUENCE_LENGTH - 1):
        buffer.push(0,0,0,0,False)
        
    assert buffer.sample_sequences(config.BATCH_SIZE, config.SEQUENCE_LENGTH) is None

def test_sample_transitions_shape_and_type():
    buffer = ReplayBuffer(capacity=100)
    batch_size = config.BATCH_SIZE
    action_dim = 3
    
    env_conf = config.ENV_CONFIGS[config.ENV_NAME]
    resize_dim = env_conf['RESIZE_DIM']
    
    for _ in range(batch_size * 2):
        buffer.push(
            np.random.rand(1, *resize_dim), 
            np.random.rand(action_dim), 
            1.0, 
            np.random.rand(1, *resize_dim), 
            False
        )
    
    batch = buffer.sample_transitions(batch_size)
    assert batch is not None
    obs_b, act_b, rew_b, next_obs_b, done_b = batch

    assert isinstance(obs_b, torch.Tensor)
    assert obs_b.shape == (batch_size, 1, *resize_dim)
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