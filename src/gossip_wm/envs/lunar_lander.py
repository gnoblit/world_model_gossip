import numpy as np
import cv2
from gymnasium.spaces import Box, Discrete

from .base_wrapper import BaseEnvWrapper
from gossip_wm import config

class LunarLanderWrapper(BaseEnvWrapper):
    """
    Wrapper for the LunarLander-v2 environment that preprocesses
    pixel observations.
    """
    def __init__(self, env):
        super().__init__(env)
        self._observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(1, *config.RESIZE_DIM),
            dtype=np.float32
        )
        # LunarLander has 4 discrete actions
        self._action_space = Discrete(4)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def observation(self, obs):
        """
        Preprocesses the pixel observation from the environment.
        The raw observation from `render_mode="rgb_array"` is (400, 600, 3).
        """
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Crop to a square, focusing on the lander
        h, w = gray_obs.shape
        crop_h = min(h, w)
        start_w = w // 2 - crop_h // 2
        cropped_obs = gray_obs[:, start_w:start_w + crop_h]

        resized_obs = cv2.resize(
            cropped_obs, config.RESIZE_DIM, interpolation=cv2.INTER_AREA
        )
        
        normalized_obs = resized_obs.astype(np.float32) / 255.0
        return np.expand_dims(normalized_obs, axis=0)

    def step(self, action):
        # The model will output continuous actions. We need to convert to discrete for the env.
        if isinstance(action, np.ndarray):
            action = np.argmax(action)
        return super().step(action)