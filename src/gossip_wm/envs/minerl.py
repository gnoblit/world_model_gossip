import numpy as np
import cv2
from gymnasium.spaces import Box, Discrete

from .base_wrapper import BaseEnvWrapper
from gossip_wm import config

# This is a highly simplified action mapping for demonstration.
# A real implementation would require a more sophisticated mapping.
MINERL_ACTION_MAP = {
    0: {'camera': [0, 0], 'attack': 0, 'forward': 0, 'jump': 0},      # No-op
    1: {'camera': [0, 0], 'attack': 1, 'forward': 0, 'jump': 0},      # Attack
    2: {'camera': [0, 0], 'attack': 0, 'forward': 1, 'jump': 0},      # Forward
    3: {'camera': [0, 0], 'attack': 0, 'forward': 1, 'jump': 1},      # Jump-Forward
    4: {'camera': [-10, 0], 'attack': 0, 'forward': 0, 'jump': 0},    # Turn Left
    5: {'camera': [10, 0], 'attack': 0, 'forward': 0, 'jump': 0},     # Turn Right
    6: {'camera': [0, -10], 'attack': 0, 'forward': 0, 'jump': 0},    # Look Up
    7: {'camera': [0, 10], 'attack': 0, 'forward': 0, 'jump': 0},     # Look Down
}


class MineRLWrapper(BaseEnvWrapper):
    """
    Wrapper for MineRL environments.
    """
    def __init__(self, env):
        super().__init__(env)
        self._observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(1, *config.RESIZE_DIM),
            dtype=np.float32
        )
        self._action_space = Discrete(len(MINERL_ACTION_MAP))

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def observation(self, obs):
        """
        Preprocesses the pixel observation from MineRL.
        - Extracts the 'pov' key.
        - Grayscales, resizes, and normalizes.
        """
        pov_obs = obs['pov']
        gray_obs = cv2.cvtColor(pov_obs, cv2.COLOR_RGB2GRAY)
        resized_obs = cv2.resize(
            gray_obs, config.RESIZE_DIM, interpolation=cv2.INTER_AREA
        )
        normalized_obs = resized_obs.astype(np.float32) / 255.0
        return np.expand_dims(normalized_obs, axis=0)

    def step(self, action):
        """
        Maps a discrete action index to the complex MineRL dictionary action.
        """
        if isinstance(action, np.ndarray):
            action = action.item()
        
        minerl_action = self.env.action_space.no_op()
        minerl_action.update(MINERL_ACTION_MAP[action])

        return super().step(minerl_action)