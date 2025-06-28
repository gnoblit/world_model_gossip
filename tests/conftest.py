import pytest
import torch
import numpy as np
import gymnasium as gym

# Import from your source code
from gossip_wm import config
from gossip_wm.models import VAE, TransitionModel, WorldModel

# A list of all supported environments for parameterization in tests
SUPPORTED_ENVS = list(config.ENV_CONFIGS.keys())

@pytest.fixture(scope="module")
def device():
    """Provides the device from the config as a fixture."""
    return config.DEVICE

@pytest.fixture(scope="module")
def dummy_obs_batch():
    """Creates a dummy batch of observations (images)."""
    b, c, h, w = config.BATCH_SIZE, 1, *config.RESIZE_DIM
    return torch.rand(b, c, h, w)

@pytest.fixture(scope="module")
def dummy_latent_batch():
    """Creates a dummy batch of latent vectors."""
    return torch.randn(config.BATCH_SIZE, config.LATENT_DIM)

@pytest.fixture(scope="module")
def dummy_action_batch():
    """Creates a dummy batch of continuous actions, matching CarRacing default."""
    # This is kept for legacy tests that don't parameterize envs.
    # New tests should generate actions based on the specific env's action space.
    action_dim = config.ENV_CONFIGS["CarRacing-v3"]["ACTION_DIM"]
    return torch.randn(config.BATCH_SIZE, action_dim)

@pytest.fixture(scope="module")
def dummy_sequence_batch():
    """Creates a dummy batch of sequences (obs, action), matching CarRacing default."""
    b, t = config.BATCH_SIZE, config.SEQUENCE_LENGTH
    c, h, w = 1, *config.RESIZE_DIM
    action_dim = config.ENV_CONFIGS["CarRacing-v3"]["ACTION_DIM"]
    
    obs_seq = torch.rand(b, t, c, h, w)
    act_seq = torch.rand(b, t, action_dim)
    return obs_seq, act_seq

@pytest.fixture(scope="module")
def vae_model(device):
    """Provides a VAE model instance on the correct device."""
    model = VAE(latent_dim=config.LATENT_DIM).to(device)
    model.eval() # Set to eval mode for testing
    return model

@pytest.fixture(scope="module")
def transition_model(device):
    """Provides a TransitionModel instance on the correct device (CarRacing default)."""
    action_dim = config.ENV_CONFIGS["CarRacing-v3"]["ACTION_DIM"]
    model = TransitionModel(
        latent_dim=config.LATENT_DIM,
        action_dim=action_dim,
        hidden_dim=config.TRANSITION_HIDDEN_DIM
    ).to(device)
    model.eval()
    return model

@pytest.fixture(scope="module")
def world_model(device):
    """Provides a full WorldModel instance on the correct device (CarRacing default)."""
    # Set the config to the default for this fixture, in case other tests changed it
    config.ENV_NAME = "CarRacing-v3"
    model = WorldModel().to(device)
    model.eval()
    return model