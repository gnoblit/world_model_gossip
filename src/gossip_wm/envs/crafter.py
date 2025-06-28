import numpy as np
import cv2
from gymnasium.spaces import Box, Discrete

from .base_wrapper import BaseEnvWrapper
from gossip_wm import config

class CrafterWrapper(BaseEnvWrapper):
    """
    Wrapper for the Crafter environment.
    """
    def __init__(self, env):
        super().__init__(env)
        self._observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(1, *config.RESIZE_DIM),
            dtype=np.float32
        )
        # Crafter has 17 discrete actions
        self._action_space = env.action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def observation(self, obs):
        """
        Preprocesses the pixel observation from Crafter.
        - Raw obs is (64, 64, 3) with values [0, 255].
        """
        # Grayscale, normalize, and add channel dimension
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        normalized_obs = gray_obs.astype(np.float32) / 255.0
        return np.expand_dims(normalized_obs, axis=0)