import gymnasium
import torch
import torch.optim as optim
import torch.nn.functional as F  # Corrected import
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import save_image
from tqdm import tqdm
import os

from gossip_wm.environment import CarRacingWrapper
from gossip_wm.replay_buffer import ReplayBuffer
from gossip_wm.models import WorldModel, vae_loss_function, reparameterize
from gossip_wm import config

def calculate_beta(step, max_beta, anneal_steps):
    """Calculates the value of beta for KL annealing."""
    if step > anneal_steps:
        return max_beta
    return max_beta * (step / anneal_steps)

def train_world_model(num_steps=20000):
    print("--- Phase 1, Step 1.3: Full World Model Training ---")
    
    # Setup
    os.makedirs("results", exist_ok=True)
    env = CarRacingWrapper(gymnasium.make(config.ENV_NAME, continuous=True))
    buffer = ReplayBuffer(capacity=config.BUFFER_CAPACITY)
    
    model = WorldModel().to(config.DEVICE)
    try:
        model.load_vae_weights()
    except FileNotFoundError:
        print("Pre-trained VAE weights not found. Training from scratch.")

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
    
    # 2. Training Loop
    print(f"\nStarting World Model training for {num_steps} steps...")
    model.train()
    pbar = tqdm(range(num_steps))
    for step in pbar:
        # Sample a batch of sequences
        batch = buffer.sample_sequences(config.BATCH_SIZE, config.SEQUENCE_LENGTH)
        if batch is None:
            continue
        
        obs_seq, act_seq, _ = batch
        
        # Reshape obs for VAE: (batch * seq_len, 1, 64, 64)
        b, t, c, h, w = obs_seq.shape
        obs_flat = obs_seq.view(b * t, c, h, w)
        
        # VAE forward pass on all observations in the sequences
        recon_flat, mu_flat, logvar_flat = model.vae(obs_flat)
        
        # --- Reconstruction Loss (like before) ---
        beta = calculate_beta(step, config.BETA_KL_MAX, config.BETA_ANNEAL_STEPS)
        recon_loss = vae_loss_function(recon_flat, obs_flat, mu_flat, logvar_flat, beta)
        
        # --- Prediction Loss ---
        # Reshape latent variables back to sequence format
        mu_seq = mu_flat.view(b, t, -1)
        logvar_seq = logvar_flat.view(b, t, -1)
        z_seq = reparameterize(mu_seq, logvar_seq)
        
        # Use the transition model to predict the next latent state for each step in the sequence
        # We predict z_{t+1} from (z_t, a_t), so we feed states 0-to-48 and actions 0-to-48
        pred_mu_seq, pred_logvar_seq, _ = model.transition(z_seq[:, :-1, :], act_seq[:, :-1, :])
        
        # The target is the actual next latent state, z_{t+1}
        target_z_seq = z_seq[:, 1:, :].detach() # Use .detach() as we don't want to backprop through the target
        
        # Prediction loss is the MSE between predicted and actual next latent state
        # We can also model this as a probability distribution, but MSE is simpler.
        prediction_loss = F.mse_loss(pred_mu_seq, target_z_seq, reduction='sum')
        
        # --- Total Loss ---
        total_loss = recon_loss + prediction_loss
        
        # Backward pass and optimize
        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        pbar.set_description(
            f"Total: {total_loss.item()/b:.2f} | "
            f"Recon: {recon_loss.item()/b:.2f} | "
            f"Pred: {prediction_loss.item()/b:.2f}"
        )
        
        # 3. Logging
        if step % config.LOG_INTERVAL == 0:
            visualize_dreams(model, obs_seq[0, 0, :, :, :], step)

    print("World Model training complete.")
    env.close()

def visualize_dreams(model, start_obs, step, dream_len=50):
    """Generates and saves a video of a dream sequence."""
    model.eval()
    with torch.no_grad():
        # Encode the starting observation
        start_obs = start_obs.unsqueeze(0).to(config.DEVICE)
        mu, logvar = model.vae.encoder(start_obs)
        z = reparameterize(mu, logvar)
        
        dream_frames = [model.vae.decoder(z)]
        hidden = None
        
        # For dreaming, we'll just repeat a simple action (go straight)
        action = torch.tensor([0.0, 0.5, 0.0]).unsqueeze(0).to(config.DEVICE) 
        
        for _ in range(dream_len - 1):
            mu, logvar, hidden = model.transition(z, action, hidden)
            z = reparameterize(mu, logvar)
            dream_frames.append(model.vae.decoder(z.squeeze(1)))
            
    dream_video = torch.cat(dream_frames, dim=0)
    save_image(dream_video.cpu(), f"results/dream_{step}.png", nrow=10)
    model.train()


if __name__ == "__main__":
    train_world_model()