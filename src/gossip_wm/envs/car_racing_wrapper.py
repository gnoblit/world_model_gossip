# src/gossip_wm/envs/car_racing_wrapper.py
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
    - Overrides step and reset to handle observation processing and action types.
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
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_obs = cv2.resize(
            gray_obs, 
            config.RESIZE_DIM, 
            interpolation=cv2.INTER_AREA
        )
        obs_with_channel = np.expand_dims(resized_obs, axis=0)
        return obs_with_channel.astype(np.float32) / 255.0

    def step(self, action):
        """
        Ensures the action is of a type accepted by the Box2D environment,
        avoiding internal numpy type promotion issues.
        """
        # *** THE FINAL FIX ***
        # Convert the numpy array to a list of Python native floats.
        # This prevents the upstream environment code from being poisoned
        # by numpy's float64 type promotion when it does its internal math.
        action = [float(a) for a in action]
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Calls the environment's reset and processes the initial observation.
        """
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info