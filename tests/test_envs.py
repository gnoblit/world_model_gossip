# tests/test_envs.py
import pytest
import gymnasium as gym
import numpy as np

from gossip_wm import config
from gossip_wm.envs import make_env
from tests.conftest import SUPPORTED_ENVS

# Helper to check if a package is available for skipping tests
def is_pkg_installed(pkg_name):
    """Checks if a package is installed without importing it everywhere."""
    try:
        __import__(pkg_name)
        return True
    except ImportError:
        return False

@pytest.mark.parametrize("env_name", SUPPORTED_ENVS)
def test_env_creation_and_wrapper(env_name):
    """
    Tests that each supported environment can be created via the factory,
    and that the wrapper provides the correct observation and action spaces.
    Skips tests for environments whose dependencies are not installed.
    """
    if "MiniGrid" in env_name and not is_pkg_installed("minigrid"):
        pytest.skip(f"minigrid not installed, skipping {env_name}.")
    if "Vizdoom" in env_name and not is_pkg_installed("vizdoom"):
        pytest.skip(f"vizdoom not installed, skipping {env_name}.")
    if any(box2d_env in env_name for box2d_env in ["CarRacing", "BipedalWalker", "LunarLander"]):
        if not is_pkg_installed("Box2D"):
             pytest.skip(f"Box2D dependencies not installed, skipping {env_name}.")

    original_env_name = config.ENV_NAME
    config.ENV_NAME = env_name
    
    env_conf = config.get_env_config()
    env = make_env(env_name)

    assert isinstance(env, gym.Wrapper)
    
    expected_obs_shape = (1, *config.RESIZE_DIM)
    assert env.observation_space.shape == expected_obs_shape, "Observation space shape is incorrect."
    assert env.observation_space.dtype == np.float32, "Observation space dtype should be float32."

    if env_conf["IS_DISCRETE"]:
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == env_conf["ACTION_DIM"]
    else:
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (env_conf["ACTION_DIM"],)

    obs, info = env.reset(seed=42)
    assert obs.shape == expected_obs_shape
    assert obs.min() >= 0.0 and obs.max() <= 1.0, "Observation not normalized."

    action = env.action_space.sample()
        
    # No longer needed, wrapper handles it.
    # if isinstance(env.action_space, gym.spaces.Box):
    #     action = action.astype(np.float32)
        
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert next_obs.shape == expected_obs_shape
    assert next_obs.min() >= 0.0 and next_obs.max() <= 1.0, "Next observation not normalized."
    
    assert isinstance(reward, (int, float, np.number))
    
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    env.close()
    
    config.ENV_NAME = original_env_name