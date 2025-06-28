import numpy as np
import cv2
from gymnasium.spaces import Box, Discrete

from .base_wrapper import BaseEnvWrapper
from gossip_wm import config

class MiniGridWrapper(BaseEnvWrapper):
    """
    Wrapper for MiniGrid environments.
    - The raw observation is a dictionary {'image': (W, H, 3), ...}.
    - We extract the 'image' key.
    - We grayscale, resize, normalize, and add a channel dimension.
    """
    def __init__(self, env):
        super().__init__(env)
        # Standardized observation space for our VAE
        self._observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(1, *config.RESIZE_DIM),
            dtype=np.float32
        )
        # MiniGrid has a discrete action space (typically 7 actions)
        self._action_space = env.action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def observation(self, obs):
        """
        Preprocesses the pixel observation from MiniGrid.
        """
        # The observation from MiniGrid is a dictionary. We need the 'image'.
        image_obs = obs['image']
        
        # The image is already small (e.g., 56x56 for a 7x7 view with 8px tiles)
        # and has uint8 values [0, 255].
        
        # Convert to grayscale
        gray_obs = cv2.cvtColor(image_obs, cv2.COLOR_RGB2GRAY)
        
        # Resize to our standard dimension
        resized_obs = cv2.resize(
            gray_obs, config.RESIZE_DIM, interpolation=cv2.INTER_NEAREST
        )
        
        # Normalize and add channel dimension
        normalized_obs = resized_obs.astype(np.float32) / 255.0
        return np.expand_dims(normalized_obs, axis=0)