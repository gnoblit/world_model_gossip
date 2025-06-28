import pytest
import torch
import numpy as np

from gossip_wm import config
from gossip_wm.training import calculate_beta, plot_loss_curves, save_reconstruction_images, visualize_dreams
from gossip_wm.models import WorldModel

def test_calculate_beta():
    """Tests the KL annealing beta calculation."""
    anneal_steps = config.BETA_ANNEAL_STEPS
    max_beta = config.BETA_KL_MAX
    
    # Before annealing starts
    assert calculate_beta(0, max_beta, anneal_steps) == 0.0
    
    # Halfway through annealing
    assert calculate_beta(anneal_steps / 2, max_beta, anneal_steps) == pytest.approx(max_beta / 2)
    
    # At the end of annealing
    assert calculate_beta(anneal_steps, max_beta, anneal_steps) == max_beta
    
    # After annealing
    assert calculate_beta(anneal_steps + 100, max_beta, anneal_steps) == max_beta

def test_plot_loss_curves(mocker, tmp_path):
    """Tests that the plotting function saves a file."""
    mock_savefig = mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("matplotlib.pyplot.figure")
    mocker.patch("matplotlib.pyplot.close")

    losses = {"test_loss": [1.0, 0.9, 0.8]}
    plot_loss_curves(losses, "Test Plot", "test.png", str(tmp_path))
    
    expected_path = tmp_path / "test.png"
    mock_savefig.assert_called_once_with(str(expected_path))

def test_save_reconstruction_images(mocker, tmp_path):
    """Tests that the reconstruction saving function calls the saver."""
    # CORRECTED: Mock the function where it is used.
    mock_save_image = mocker.patch("gossip_wm.training.save_image")
    
    # Create dummy tensors
    original = torch.rand(16, 1, 64, 64)
    reconstruction = torch.rand(16, 1, 64, 64)
    
    run_dir = tmp_path
    recons_dir = run_dir / "reconstructions"
    recons_dir.mkdir()

    save_reconstruction_images(original, reconstruction, 100, str(run_dir))

    expected_path = recons_dir / "reconstruction_100.png"
    mock_save_image.assert_called_once()
    # Check the path argument of the call
    call_args, _ = mock_save_image.call_args
    assert call_args[1] == str(expected_path)


def test_visualize_dreams(mocker, tmp_path, world_model):
    """Tests that the dream visualization function calls the saver."""
    mock_save_image = mocker.patch("gossip_wm.training.save_image")
    
    # Get latent_dim from the model fixture itself, which is robust
    latent_dim = world_model.transition.latent_dim
    start_z = torch.randn(latent_dim)
    
    agent_id_str = "0"
    step = 500
    
    run_dir = tmp_path
    dream_dir = run_dir / f"dreams_agent_{agent_id_str}"
    dream_dir.mkdir()

    visualize_dreams(world_model, start_z, step, agent_id_str, str(run_dir), dream_len=10)
    
    expected_path = dream_dir / f"dream_{step}.png"
    mock_save_image.assert_called_once()
    call_args, _ = mock_save_image.call_args
    assert call_args[1] == str(expected_path)