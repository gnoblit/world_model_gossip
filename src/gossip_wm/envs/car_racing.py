### Defines environment for CarRacing-v3

import gymnasium as gym
import numpy as np
import cv2  
from gymnasium.spaces import Box
from gossip_wm  import config
from . import BaseEnvWrapper

class CarRacingWrapper(BaseEnvWrapper):
    """
    A wrapper for the CarRacing-v3 environment that preprocesses observations.
    - Converts obs to greyscale.
    - Resizes to smaller dim (64x64).
    - Normalizes pixels to [0,1] range.
    - Changes obs shape to (C, H, W) for PyTorch
    """
    def __init__(self, env):
        super().__init__(env)
        self._observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(1, *config.RESIZE_DIM),
            dtype=np.float32
        )

    @property
    def observation_space(self):
        return self._observation_space

    def observation(self, obs):
        """
        Preprocesses the observation for CarRacing.
        - Crop, grayscale, resize, normalize, and add channel dimension.
        """
        # Crop the bottom score bar
        cropped_obs = obs[:84, :, :]
        # Convert to grayscale
        gray_obs = cv2.cvtColor(cropped_obs, cv2.COLOR_RGB2GRAY)
        # Resize
        resized_obs = cv2.resize(
            gray_obs, config.RESIZE_DIM, interpolation=cv2.INTER_AREA
        )
        # Normalize and add channel dimension
        normalized_obs = resized_obs.astype(np.float32) / 255.0
        return np.expand_dims(normalized_obs, axis=0)