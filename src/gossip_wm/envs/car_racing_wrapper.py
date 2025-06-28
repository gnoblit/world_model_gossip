# src/gossip_wm/envs/car_racing_wrapper.py
import gymnasium as gym
import numpy as np
import cv2
from .base_wrapper import BaseEnvWrapper
from gossip_wm import config

class CarRacingWrapper(BaseEnvWrapper):
    """Wrapper for CarRacing, handles observation processing and action type."""
    def __init__(self, env):
        super().__init__(env)
        env_conf = config.get_env_config()
        self.resize_dim = env_conf['RESIZE_DIM']
        self._observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1, *self.resize_dim), dtype=np.float32
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
        resized_obs = cv2.resize(gray_obs, self.resize_dim, interpolation=cv2.INTER_AREA)
        obs_with_channel = np.expand_dims(resized_obs, axis=0)
        return obs_with_channel.astype(np.float32) / 255.0

    def step(self, action):
        action = [float(a) for a in action]
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info