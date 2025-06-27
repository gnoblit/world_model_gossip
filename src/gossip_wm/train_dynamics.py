# train_dynamics.py

import gymnasium
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import save_image
from tqdm import tqdm
import os
import numpy as np

# Import necessary components
from gossip_wm.environment import CarRacingWrapper
from gossip_wm.replay_buffer import ReplayBuffer 
from gossip_wm.models import WorldModel, reparameterize
from gossip_wm import config

def pre_encode_buffer(model, buffer):
    """
    Encodes all observations in the buffer to latent states.
    Returns a new buffer with latent states instead of observations.
    """
    print("Pre-encoding replay buffer...")
    model.eval() # Set model to evaluation mode for encoding
    
    latent_buffer = []
    with torch.no_grad():
        for i in tqdm(range(len(buffer.memory))):
            obs, action, reward, next_obs, done = buffer.memory[i]
            
            # Convert obs to tensor and encode
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(config.DEVICE)
            mu, logvar = model.vae.encoder(obs_tensor)
            z = reparameterize(mu, logvar).squeeze(0).cpu().numpy() # Store as numpy array on CPU
            
            latent_buffer.append((z, action, reward, done))
            
    model.train() # Set model back to training mode
    return latent_buffer

def train_dynamics_model(num_steps=10000): # Can use fewer steps now
    print("--- Optimized Dynamics Model Training ---")
    
    # --- Setup ---
    os.makedirs("results", exist_ok=True)
    model = WorldModel().to(config.DEVICE)
    try:
        model.load_vae_weights()
    except FileNotFoundError:
        print("ERROR: Pre-trained VAE weights (vae_model.pth) not found. Please run verify_step2.py first.")
        return
    
    # --- Data Collection (same as before) ---
    env = CarRacingWrapper(gymnasium.make(config.ENV_NAME, continuous=True))
    pixel_buffer = ReplayBuffer(capacity=config.SEED_STEPS) # Temporary buffer for pixels
    print(f"Collecting {config.SEED_STEPS} steps of random data...")
    obs, _ = env.reset()
    for _ in tqdm(range(config.SEED_STEPS)):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        pixel_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs if not done else env.reset()[0]
    
    # --- OPTIMIZATION: Pre-encode the entire buffer ---
    latent_buffer_list = pre_encode_buffer(model, pixel_buffer)
    del pixel_buffer # Free up memory
    
    # --- Training Loop Setup ---
    # We only optimize the transition model parameters
    optimizer = optim.Adam(model.transition.parameters(), lr=config.LEARNING_RATE)
    # OPTIMIZATION: Automatic Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    print(f"\nStarting dynamics model training for {num_steps} steps...")
    model.transition.train()
    model.vae.eval() # VAE is frozen
    
    pbar = tqdm(range(num_steps))
    for step in pbar:
        # Sample sequences directly from our list of latent tuples
        batch_indices = np.random.choice(len(latent_buffer_list) - config.SEQUENCE_LENGTH, size=config.BATCH_SIZE, replace=False)
        sequences = [latent_buffer_list[i : i + config.SEQUENCE_LENGTH] for i in batch_indices]
        
        z_seq, act_seq, _, _ = zip(*[zip(*seq) for seq in sequences])
        
        z_batch = torch.from_numpy(np.array(z_seq)).to(config.DEVICE)
        action_batch = torch.from_numpy(np.array(act_seq)).to(config.DEVICE)
        
        # --- OPTIMIZATION: Use Mixed Precision ---
        with torch.amp.autocast(device_type=config.DEVICE.type, enabled=torch.cuda.is_available()):
            # --- Prediction Loss (Simplified Loop) ---
            pred_mu_seq, pred_logvar_seq, _ = model.transition(z_batch[:, :-1, :], action_batch[:, :-1, :])
            target_z_seq = z_batch[:, 1:, :].detach()
            prediction_loss = F.mse_loss(pred_mu_seq, target_z_seq) # Use mean reduction
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(prediction_loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.transition.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        pbar.set_description(f"Prediction Loss: {prediction_loss.item():.4f}")
        
        if step % config.LOG_INTERVAL == 0:
            visualize_dreams(model, z_batch[0, 0, :], step)

    
    print("Dynamics Model training complete.")
    env.close()

def visualize_dreams(model, start_z, step, dream_len=50):
    """Generates and saves a video of a dream sequence, starting from a latent vector."""
    model.eval()
    with torch.no_grad():
        z = start_z.unsqueeze(0).to(config.DEVICE)
        dream_frames = [model.vae.decoder(z)]
        hidden = None
        action = torch.tensor([0.0, 0.5, 0.0]).unsqueeze(0).to(config.DEVICE)
        
        for _ in range(dream_len - 1):
            mu, _, hidden = model.transition(z, action, hidden)
            # For dreaming, we just use the mean of the prediction
            z = mu.squeeze(1)
            dream_frames.append(model.vae.decoder(z))
            
    dream_video = torch.cat(dream_frames, dim=0)
    save_image(dream_video.cpu(), f"results/dream_{step}.png", nrow=10)
    model.train()
    model.vae.eval() # Keep VAE in eval mode

if __name__ == "__main__":
    # For a quick verification, you can reduce the number of steps
    train_dynamics_model(num_steps=5000)