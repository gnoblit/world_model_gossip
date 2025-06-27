### Defines teh replay buffer to store data for sampling

import random
from collections import deque
import torch
import numpy as np
from gossip_wm  import config

class ReplayBuffer:
    """A simple replay buffer for storing and sampling transitions."""

    def __init__(self, capacity):
        # Deque nicely lets us add adn remove items from both ends. 
        # `maxlen` prevents going over capacity.
        self.memory = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        """Saves a transition."""
        self.memory.append((obs, action, reward, next_obs, done))

    def sample_transition(self, batch_size):
        """
        Samples a batch of single transitions and converts them to PyTorch tensors.
        Useful for training the VAE.
        """
        if len(self.memory) < batch_size:
            return None
        
        transitions = random.sample(self.memory, batch_size)
        batch = list(zip(*transitions))
        
        obs_batch = torch.from_numpy(np.array(batch[0])).float().to(config.DEVICE)
        action_batch = torch.from_numpy(np.array(batch[1])).float().to(config.DEVICE)
        reward_batch = torch.from_numpy(np.array(batch[2])).float().to(config.DEVICE).unsqueeze(1)
        next_obs_batch = torch.from_numpy(np.array(batch[3])).float().to(config.DEVICE)
        done_batch = torch.from_numpy(np.array(batch[4])).float().to(config.DEVICE).unsqueeze(1)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
    
        def __len__(self):
            return len(self.memory)