import pytest
import torch
import numpy as np
import os

from gossip_wm import config
from gossip_wm.training import (
    calculate_beta, 
    plot_loss_curves, 
    save_reconstruction_images, 
    visualize_dreams,
    generate_and_save_buffer,
    load_buffer_from_file
)
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

def test_buffer_generation_and_loading_respects_capacity(tmp_path):
    """
    Tests that generating a buffer to disk and loading it with Reservoir Sampling
    correctly respects the specified memory capacity.
    """
    # 1. Setup
    original_env_name = config.ENV_NAME
    config.ENV_NAME = "CarRacing-v3" # Use a known env for data generation
    
    total_steps_on_disk = 500
    buffer_capacity_in_ram = 100
    batch_size = 32

    buffer_path = tmp_path / "test_buffer.pkl"

    # 2. Generate a buffer file with more steps than the target capacity
    generate_and_save_buffer(num_steps=total_steps_on_disk, save_path=str(buffer_path))
    assert os.path.exists(buffer_path)

    # 3. Load the buffer from the file using the new function
    # This should use reservoir sampling to only load `buffer_capacity_in_ram` samples.
    buffer = load_buffer_from_file(str(buffer_path), capacity=buffer_capacity_in_ram)

    # 4. Assertions
    assert buffer is not None, "Buffer loading failed."
    
    # --- The most important check ---
    # The buffer in memory should have a length equal to the capacity,
    # NOT the total number of steps on disk.
    assert len(buffer) == buffer_capacity_in_ram
    assert len(buffer) != total_steps_on_disk

    # Check that we can sample from the loaded buffer
    batch = buffer.sample_transitions(batch_size)
    assert batch is not None
    assert len(batch) == 5 # obs, action, reward, next_obs, done
    assert batch[0].shape[0] == batch_size # Check batch size of obs

    # 5. Teardown
    config.ENV_NAME