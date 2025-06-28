import gymnasium as gym
import numpy as np
import cv2
from gymnasium.spaces import Box

from .base_wrapper import BaseEnvWrapper
from gossip_wm  import config

class BipedalWalkerWrapper(BaseEnvWrapper):
    """
    Wrapper for the BipedalWalker-v3 environment that preprocesses
    pixel observations.
    """
    def __init__(self, env):
        super().__init__(env)
        # The output of this wrapper is a standardized image
        self._observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(1, *config.RESIZE_DIM),
            dtype=np.float32
        )
        # BipedalWalker has 4 continuous actions
        self._action_space = Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def observation(self, obs):
        """
        Preprocesses the pixel observation from the environment.
        - The raw observation from `render_mode="rgb_array"` is (400, 600, 3).
        - We grayscale, resize, normalize, and add a channel dimension.
        """
        # Convert to grayscale
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Resize - crop the height to make it square before resizing
        # This focuses on the central action area. Raw shape is (400, 600)
        h, w = gray_obs.shape
        crop_h = min(h, w) # Crop to a 400x400 square from the center-left
        start_w = w // 2 - crop_h // 2
        cropped_obs = gray_obs[:, start_w:start_w + crop_h]

        resized_obs = cv2.resize(
            cropped_obs, config.RESIZE_DIM, interpolation=cv2.INTER_AREA
        )
        
        # Normalize and add channel dimension
        normalized_obs = resized_obs.astype(np.float32) / 255.0
        return np.expand_dims(normalized_obs, axis=0)