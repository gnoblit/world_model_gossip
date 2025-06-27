import pytest
import gymnasium as gym
import numpy as np

from gossip_wm.environment import CarRacingWrapper
from gossip_wm import config

@pytest.fixture(scope="module")
def wrapped_env():
    """Fixture to create and close the wrapped environment."""
    env = CarRacingWrapper(gym.make(config.ENV_NAME, continuous=True))
    yield env
    env.close()

def test_env_observation_space(wrapped_env):
    """Tests if the observation space of the wrapper is correct."""
    expected_shape = (1, *config.RESIZE_DIM)
    assert wrapped_env.observation_space.shape == expected_shape, "Observation space shape is incorrect."
    assert wrapped_env.observation_space.low.min() >= 0.0, "Observation space lower bound is incorrect."
    assert wrapped_env.observation_space.high.max() <= 1.0, "Observation space upper bound is incorrect."

def test_env_reset(wrapped_env):
    """Tests the reset method of the wrapped environment."""
    obs, info = wrapped_env.reset()
    
    assert isinstance(obs, np.ndarray), "Observation should be a numpy array."
    assert obs.shape == wrapped_env.observation_space.shape, "Reset observation shape is incorrect."
    assert obs.min() >= 0.0 and obs.max() <= 1.0, "Reset observation values are not in [0, 1] range."
    assert isinstance(info, dict), "Info should be a dictionary."

def test_env_step(wrapped_env):
    """Tests the step method of the wrapped environment."""
    wrapped_env.reset()
    action = wrapped_env.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    
    assert isinstance(obs, np.ndarray), "Step observation should be a numpy array."
    assert obs.shape == wrapped_env.observation_space.shape, "Step observation shape is incorrect."
    assert obs.min() >= 0.0 and obs.max() <= 1.0, "Step observation values are not in [0, 1] range."
    assert isinstance(reward, float), "Reward should be a float."
    assert isinstance(terminated, bool), "Terminated flag should be a boolean."
    assert isinstance(truncated, bool), "Truncated flag should be a boolean."