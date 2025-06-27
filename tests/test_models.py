import torch

from gossip_wm import config
from gossip_wm.models import reparameterize, vae_loss_function

# Note: Models are imported from conftest.py as fixtures

def test_reparameterize(dummy_latent_batch):
    """Tests the reparameterization trick."""
    mu = logvar = torch.zeros_like(dummy_latent_batch)
    z = reparameterize(mu, logvar)
    assert z.shape == mu.shape, "Reparameterized z has incorrect shape."
    # With mu=0, logvar=0 (std=1), the mean of z should be close to 0
    assert torch.allclose(z.mean(), torch.tensor(0.0), atol=0.2), "Mean of z should be near 0."

def test_encoder_shapes(vae_model, dummy_obs_batch, device):
    """Tests the output shapes of the Encoder."""
    dummy_obs_batch = dummy_obs_batch.to(device)
    mu, logvar = vae_model.encoder(dummy_obs_batch)
    
    assert mu.shape == (config.BATCH_SIZE, config.LATENT_DIM)
    assert logvar.shape == (config.BATCH_SIZE, config.LATENT_DIM)

def test_decoder_shapes(vae_model, dummy_latent_batch, device):
    """Tests the output shape of the Decoder."""
    dummy_latent_batch = dummy_latent_batch.to(device)
    recon = vae_model.decoder(dummy_latent_batch)
    
    expected_shape = (config.BATCH_SIZE, 1, *config.RESIZE_DIM)
    assert recon.shape == expected_shape

def test_vae_forward_pass(vae_model, dummy_obs_batch, device):
    """Tests the full forward pass of the VAE."""
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

def test_transition_model_shapes(transition_model, dummy_sequence_batch, device):
    """Tests the output shapes of the TransitionModel."""
    _, act_seq = dummy_sequence_batch
    z_seq = torch.randn(
        config.BATCH_SIZE, config.SEQUENCE_LENGTH, config.LATENT_DIM
    ).to(device)
    act_seq = act_seq.to(device)
    
    # We predict the next T-1 states from the first T-1 states and actions
    pred_mu, pred_logvar, hidden = transition_model(z_seq[:, :-1, :], act_seq[:, :-1, :])
    
    # Output should have T-1 sequence length
    expected_shape = (config.BATCH_SIZE, config.SEQUENCE_LENGTH - 1, config.LATENT_DIM)
    assert pred_mu.shape == expected_shape
    assert pred_logvar.shape == expected_shape
    assert hidden.shape == (1, config.BATCH_SIZE, config.TRANSITION_HIDDEN_DIM)

def test_world_model_creation(world_model):
    """Tests that the full WorldModel can be instantiated."""
    assert world_model.vae is not None
    assert world_model.transition is not None