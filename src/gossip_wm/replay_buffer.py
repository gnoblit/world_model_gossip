# src/gossip_wm/replay_buffer.py

import random
from collections import deque
import torch
import numpy as np
from . import config

class ReplayBuffer:
    """
    A replay buffer that stores transitions and can efficiently sample sequences.
    This version uses a much faster, vectorized method for finding valid start indices.
    """
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        """Saves a single transition."""
        self.memory.append((obs, action, reward, next_obs, done))

    def sample_sequences(self, batch_size, seq_len):
        """Efficiently samples a batch of consecutive sequences of transitions."""
        n_transitions = len(self.memory)
        if n_transitions < seq_len:
            return None

        # --- OPTIMIZATION: Fast, vectorized valid start index calculation ---
        # 1. Extract 'done' flags into a numpy array
        dones = np.array([t[4] for t in self.memory], dtype=bool)

        # 2. Find starts where a 'done' appears in the middle of the sequence
        # A convolution with a kernel of ones sums up 'dones' in a sliding window.
        # If the sum > 0, a 'done' was found, making it an invalid start.
        invalid_starts = np.convolve(dones, np.ones(seq_len - 1), mode='valid') > 0
        
        # 3. Get all valid start indices
        # We subtract the length of the invalid_starts array to align indices.
        possible_indices = np.arange(n_transitions - seq_len + 1)
        valid_indices = possible_indices[~invalid_starts[:len(possible_indices)]]
        
        if len(valid_indices) < batch_size:
            return None

        # 4. Sample and gather
        batch_start_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        sequences = [list(self.memory)[i : i + seq_len] for i in batch_start_indices]
        
        obs_seqs, act_seqs, rew_seqs, _, _ = zip(*[zip(*seq) for seq in sequences])

        obs_batch = torch.from_numpy(np.array(obs_seqs)).float().to(config.DEVICE)
        action_batch = torch.from_numpy(np.array(act_seqs)).float().to(config.DEVICE)
        reward_batch = torch.from_numpy(np.array(rew_seqs)).float().to(config.DEVICE).unsqueeze(-1)

        return obs_batch, action_batch, reward_batch

    def __len__(self):
        return len(self.memory)