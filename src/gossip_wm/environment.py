### Defines environment for CarRacing-v3

import gymnasium as gym
import numpy as np
import cv2  # Fast image processing
from gymnasium.spaces import Box
from . import config # Import the config file

class CarRacingWrapper(gym.Wrapper):
    """
    A wrapper for the CarRacing-v3 environment that preprocesses observations.
    - Converts obs to greyscale.
    - Resizes to smaller dim (64x64).
    - Normalizes pixels to [0,1] range.
    - Changes obs shape to (C, H, W) for PyTorch
    """
    def __init__(self, env, resize_dim=(64,64)):
        super(CarRacingWrapper, self).__init__(env)
        self.resize_dim = config.RESIZE_DIM

        # Update the observation space to reflect the preprocessed output.
        # Preprocessed observations are single-channel (grayscale), resized,
        # and normalized to [0, 1].
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(1, self.resize_dim[0], self.resize_dim[1]),
            dtype=np.float32
        )

    def observation(self, obs):
        """
        Preprocesses the observation.
        - Converts from RGB to Grayscale.
        - Resizes the image.
        - Normalizes pixel values to [0, 1].
        - Adds a channel dimension.
        """
        # 1. Crop the bottom part (scores/info bar) of the observation
        # The original observation is 96x96. We crop the bottom 12 pixels.
        cropped_obs = obs[:84, :, :]
        
        # 2. Convert from RGB to Grayscale
        # cv2.cvtColor expects BGR, but gym provides RGB. The result is the same.
        gray_obs = cv2.cvtColor(cropped_obs, cv2.COLOR_RGB2GRAY)
        
        # 3. Resize the image
        resized_obs = cv2.resize(
            gray_obs, self.resize_dim, interpolation=cv2.INTER_AREA
        )
        
        # 4. Normalize pixel values to [0, 1] and add channel dimension
        # The final shape should be (1, H, W) for PyTorch's CNNs.
        normalized_obs = resized_obs.astype(np.float32) / 255.0
        return np.expand_dims(normalized_obs, axis=0)

    def step(self, action):
        """
        Overrides the step method to apply observation preprocessing.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Overrides the reset method to apply observation preprocessing to the initial observation.
        """
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info