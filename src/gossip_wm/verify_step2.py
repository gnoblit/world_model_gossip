# verify_step2.py

import gymnasium
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import save_image
from tqdm import tqdm
import os

from gossip_wm.environment import CarRacingWrapper
from gossip_wm.replay_buffer import ReplayBuffer
from gossip_wm.models import VAE, vae_loss_function
from gossip_wm import config

def calculate_beta(step, max_beta, anneal_steps):
    """
    Calculates the value of beta for KL annealing.
    Linearly increases from 0 to max_beta over anneal_steps.
    """
    if step > anneal_steps:
        return max_beta
    return max_beta * (step / anneal_steps)

def train_vae(num_steps=10000): # Increased steps for better convergence
    print("--- Phase 1, Step 1.2 (Improved): VAE Training with KL Annealing ---")
    
    # Setup
    print(f"Using device: {config.DEVICE}")
    os.makedirs("results", exist_ok=True)

    env = CarRacingWrapper(gymnasium.make(config.ENV_NAME, continuous=True))
    buffer = ReplayBuffer(capacity=config.BUFFER_CAPACITY)
    
    model = VAE(latent_dim=config.LATENT_DIM).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 1. Populate buffer
    print(f"Populating buffer with {config.SEED_STEPS} random steps...")
    obs, _ = env.reset()
    for _ in tqdm(range(config.SEED_STEPS)):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs if not done else env.reset()[0]
    
    # 2. VAE Training Loop
    print(f"\nStarting VAE training for {num_steps} steps...")
    model.train()
    pbar = tqdm(range(num_steps))
    for step in pbar:
        batch = buffer.sample_transitions(config.BATCH_SIZE)
        if batch is None:
            continue
        
        obs_batch, _, _, _, _ = batch
        
        # Calculate current beta for KL annealing
        beta = calculate_beta(step, config.BETA_KL_MAX, config.BETA_ANNEAL_STEPS)
        
        # Forward pass
        recon_batch, mu, logvar = model(obs_batch)
        
        # Calculate loss
        loss = vae_loss_function(recon_batch, obs_batch, mu, logvar, beta)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0) # Added gradient clipping
        optimizer.step()
        
        pbar.set_description(f"Loss: {loss.item() / config.BATCH_SIZE:.2f}, Beta: {beta:.2f}")
        
        if step % config.LOG_INTERVAL == 0:
            save_reconstruction_images(obs_batch, recon_batch, step)

    print("VAE training complete.")
    # Save the final model
    torch.save(model.state_dict(), "vae_model.pth")
    print("Saved final VAE model to vae_model.pth")
    env.close()

def save_reconstruction_images(original, reconstruction, step):
    """Saves a comparison of original and reconstructed images."""
    with torch.no_grad():
        comparison = torch.cat([original[:8], reconstruction[:8]])
        save_image(comparison.cpu(), f"results/reconstruction_{step}.png", nrow=8)

if __name__ == "__main__":
    train_vae()