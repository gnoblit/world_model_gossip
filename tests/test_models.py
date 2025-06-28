# tests/test_models.py
import pytest
import torch
import torch.nn.functional as F

from gossip_wm import config
from gossip_wm.models import reparameterize, vae_loss_function, WorldModel, TransitionModel
from tests.conftest import SUPPORTED_ENVS

# Note: VAE and other models are imported from conftest.py as fixtures

def test_reparameterize(dummy_latent_batch):
    """Tests the reparameterization trick."""
    mu = logvar = torch.zeros_like(dummy_latent_batch)
    z = reparameterize(mu, logvar)
    assert z.shape == mu.shape, "Reparameterized z has incorrect shape."
    # With mu=0, logvar=0 (std=1), the mean of z should be close to 0
    assert torch.allclose(z.mean(), torch.tensor(0.0), atol=0.2), "Mean of z should be near 0."

def test_encoder_shapes(vae_model, dummy_obs_batch, device):
    """Tests the output shapes of the Encoder."""
    config.ENV_NAME = "CarRacing-v3" # Ensure correct config state for this test
    dummy_obs_batch = dummy_obs_batch.to(device)
    mu, logvar = vae_model.encoder(dummy_obs_batch)
    
    assert mu.shape == (config.BATCH_SIZE, config.LATENT_DIM)
    assert logvar.shape == (config.BATCH_SIZE, config.LATENT_DIM)

def test_decoder_shapes(vae_model, dummy_latent_batch, device):
    """Tests the output shape of the Decoder."""
    config.ENV_NAME = "CarRacing-v3" # Ensure correct config state for this test
    dummy_latent_batch = dummy_latent_batch.to(device)
    recon = vae_model.decoder(dummy_latent_batch)
    
    expected_shape = (config.BATCH_SIZE, 1, *config.RESIZE_DIM)
    assert recon.shape == expected_shape

def test_vae_forward_pass(vae_model, dummy_obs_batch, device):
    """Tests the full forward pass of the VAE."""
    config.ENV_NAME = "CarRacing-v3" # Ensure correct config state for this test
    dummy_obs_batch = dummy_obs_batch.to(device)
    recon, mu, logvar = vae_model(dummy_obs_batch)
    
    expected_img_shape = (config.BATCH_SIZE, 1, *config.RESIZE_DIM)
    expected_latent_shape = (config.BATCH_SIZE, config.LATENT_DIM)
    
    assert recon.shape == expected_img_shape
    assert mu.shape == expected_latent_shape
    assert logvar.shape == expected_latent_shape
    
    # Check that output values are in [0, 1] due to sigmoid
    assert recon.min() >= 0.0 and recon.max() <= 1.0

def test_vae_loss_function(vae_model, dummy_obs_batch, device):
    """Tests that the VAE loss function returns a scalar tensor."""
    dummy_obs_batch = dummy_obs_batch.to(device)
    recon, mu, logvar = vae_model(dummy_obs_batch)
    loss = vae_loss_function(recon, dummy_obs_batch, mu, logvar, beta=1.0)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([]) # A scalar tensor has an empty shape
    assert loss.item() >= 0.0, "Loss must be non-negative."

@pytest.mark.parametrize("env_name", SUPPORTED_ENVS)
def test_transition_model_shapes_all_envs(env_name, device):
    """Tests the output shapes of the TransitionModel for all envs."""
    original_env_name = config.ENV_NAME
    config.ENV_NAME = env_name
    env_specific_config = config.get_env_config()

    # Get model dimensions from the dynamic config
    action_dim = env_specific_config['ACTION_DIM']
    is_discrete = env_specific_config['IS_DISCRETE']
    latent_dim = env_specific_config['LATENT_DIM']
    hidden_dim = env_specific_config['TRANSITION_HIDDEN_DIM']

    model = TransitionModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim
    ).to(device)
    model.eval()

    z_seq = torch.randn(config.BATCH_SIZE, config.SEQUENCE_LENGTH, latent_dim).to(device)
    
    if is_discrete:
        act_indices = torch.randint(0, action_dim, (config.BATCH_SIZE, config.SEQUENCE_LENGTH)).to(device)
        act_seq = F.one_hot(act_indices, num_classes=action_dim).float()
    else:
        act_seq = torch.rand(config.BATCH_SIZE, config.SEQUENCE_LENGTH, action_dim).to(device)
    
    pred_mu, pred_logvar, pred_done, next_hidden = model(z_seq[:, :-1, :], act_seq[:, :-1, :])
    
    expected_latent_shape = (config.BATCH_SIZE, config.SEQUENCE_LENGTH - 1, latent_dim)
    assert pred_mu.shape == expected_latent_shape
    assert pred_logvar.shape == expected_latent_shape
    
    expected_done_shape = (config.BATCH_SIZE, config.SEQUENCE_LENGTH - 1, 1)
    assert pred_done.shape == expected_done_shape
    
    assert next_hidden.shape == (1, config.BATCH_SIZE, hidden_dim)
    
    config.ENV_NAME = original_env_name
    
@pytest.mark.parametrize("env_name", SUPPORTED_ENVS)
def test_world_model_creation_all_envs(env_name, device):
    """Tests that the full WorldModel can be instantiated for all envs."""
    original_env_name = config.ENV_NAME
    config.ENV_NAME = env_name
    
    model = WorldModel().to(device)
    env_specific_config = config.get_env_config()

    assert model.vae is not None
    assert model.transition is not None
    assert model.transition.action_dim == env_specific_config['ACTION_DIM']

    config.ENV_NAME = original_env_name

def test_world_model_load_vae_weights(world_model, vae_model, tmp_path):
    """Tests the utility function to load pre-trained VAE weights."""
    vae_path = tmp_path / "vae.pth"
    torch.save(vae_model.state_dict(), vae_path)
    
    # Create a new world model with random VAE weights
    config.ENV_NAME = "CarRacing-v3" # Ensure config is set
    fresh_world_model = WorldModel().to(config.DEVICE)
    
    # Check that weights are different before loading
    assert not torch.equal(
        fresh_world_model.vae.encoder.conv1.weight,
        vae_model.encoder.conv1.weight
    )
    
    # Load weights
    fresh_world_model.load_vae_weights(path=vae_path)
    
    # Check that weights are the same after loading
    assert torch.equal(
        fresh_world_model.vae.encoder.conv1.weight,
        vae_model.encoder.conv1.weight
    )