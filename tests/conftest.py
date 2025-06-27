import pytest
import torch
import numpy as np

# Import from your source code
from gossip_wm import config
from gossip_wm.models import VAE, TransitionModel, WorldModel

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
    """Creates a dummy batch of actions."""
    return torch.randn(config.BATCH_SIZE, config.ACTION_DIM)

@pytest.fixture(scope="module")
def dummy_sequence_batch():
    """Creates a dummy batch of sequences (obs, action)."""
    b, t = config.BATCH_SIZE, config.SEQUENCE_LENGTH
    c, h, w = 1, *config.RESIZE_DIM
    
    obs_seq = torch.rand(b, t, c, h, w)
    act_seq = torch.rand(b, t, config.ACTION_DIM)
    return obs_seq, act_seq

@pytest.fixture(scope="module")
def vae_model(device):
    """Provides a VAE model instance on the correct device."""
    model = VAE(latent_dim=config.LATENT_DIM).to(device)
    model.eval() # Set to eval mode for testing
    return model

@pytest.fixture(scope="module")
def transition_model(device):
    """Provides a TransitionModel instance on the correct device."""
    model = TransitionModel(
        latent_dim=config.LATENT_DIM,
        action_dim=config.ACTION_DIM,
        hidden_dim=config.TRANSITION_HIDDEN_DIM
    ).to(device)
    model.eval()
    return model

@pytest.fixture(scope="module")
def world_model(device):
    """Provides a full WorldModel instance on the correct device."""
    model = WorldModel().to(device)
    model.eval()
    return model