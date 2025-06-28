# tests/conftest.py
import pytest
import torch
import numpy as np
import gymnasium as gym

from gossip_wm import config
from gossip_wm.models import VAE, TransitionModel, WorldModel

SUPPORTED_ENVS = list(config.ENV_CONFIGS.keys())

@pytest.fixture(scope="module")
def device():
    return config.DEVICE

@pytest.fixture(scope="module")
def default_env_config():
    """Provides the config for the default environment (CarRacing-v3)."""
    return config.ENV_CONFIGS["CarRacing-v3"]

@pytest.fixture(scope="module")
def dummy_obs_batch(default_env_config):
    """Creates a dummy batch of observations (images)."""
    resize_dim = default_env_config['RESIZE_DIM']
    b, c, h, w = config.BATCH_SIZE, 1, *resize_dim
    return torch.rand(b, c, h, w)

@pytest.fixture(scope="module")
def dummy_latent_batch(default_env_config):
    """Creates a dummy batch of latent vectors."""
    latent_dim = default_env_config['LATENT_DIM']
    return torch.randn(config.BATCH_SIZE, latent_dim)

@pytest.fixture(scope="module")
def dummy_action_batch(default_env_config):
    action_dim = default_env_config["ACTION_DIM"]
    return torch.randn(config.BATCH_SIZE, action_dim)

@pytest.fixture(scope="module")
def dummy_sequence_batch(default_env_config):
    """Creates a dummy batch of sequences (obs, action)."""
    resize_dim = default_env_config['RESIZE_DIM']
    action_dim = default_env_config['ACTION_DIM']
    b, t, c, h, w = config.BATCH_SIZE, config.SEQUENCE_LENGTH, 1, *resize_dim
    obs_seq = torch.rand(b, t, c, h, w)
    act_seq = torch.rand(b, t, action_dim)
    return obs_seq, act_seq

@pytest.fixture(scope="module")
def vae_model(device, default_env_config):
    """Provides a VAE model instance on the correct device."""
    latent_dim = default_env_config['LATENT_DIM']
    model = VAE(latent_dim=latent_dim).to(device)
    model.eval()
    return model

@pytest.fixture(scope="module")
def transition_model(device, default_env_config):
    """Provides a TransitionModel instance on the correct device."""
    action_dim = default_env_config['ACTION_DIM']
    latent_dim = default_env_config['LATENT_DIM']
    hidden_dim = default_env_config['TRANSITION_HIDDEN_DIM']
    model = TransitionModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim
    ).to(device)
    model.eval()
    return model

@pytest.fixture(scope="module")
def world_model(device):
    """Provides a full WorldModel instance on the correct device (CarRacing-v3)."""
    config.ENV_NAME = "CarRacing-v3"
    model = WorldModel().to(device)
    model.eval()
    return model