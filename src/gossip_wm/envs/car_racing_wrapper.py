import gymnasium as gym
import numpy as np
import cv2
from .base_wrapper import BaseEnvWrapper
from gossip_wm import config

class CarRacingWrapper(BaseEnvWrapper):
    """
    Wrapper for the CarRacing environment.
    - Converts observations to grayscale.
    - Resizes observations.
    - Normalizes pixel values.
    """
    def __init__(self, env):
        super().__init__(env)
        self._observation_space = gym.spaces.Box(
            low=0.0, high=1.0, 
            shape=(1, *config.RESIZE_DIM), 
            dtype=np.float32
        )
        self._action_space = self.env.action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def observation(self, obs):
        # CarRacing obs is (96, 96, 3)
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_obs = cv2.resize(
            gray_obs, 
            config.RESIZE_DIM, 
            interpolation=cv2.INTER_AREA
        )
        obs_with_channel = np.expand_dims(resized_obs, axis=0)
        return obs_with_channel.astype(np.float32) / 255.0