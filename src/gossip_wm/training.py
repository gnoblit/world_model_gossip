# src/gossip_wm/training.py

import gymnasium
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import save_image
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle 

# Import from our own library
from .envs import make_env 
from .replay_buffer import ReplayBuffer
from .models import VAE, WorldModel, vae_loss_function, reparameterize
from . import config


### =================================================================
###                   HELPER & PLOTTING FUNCTIONS
### =================================================================

def calculate_beta(step, max_beta, anneal_steps):
    """Calculates the value of beta for KL annealing."""
    if step > anneal_steps:
        return max_beta
    return max_beta * (step / anneal_steps)

def save_reconstruction_images(original, reconstruction, step, run_dir):
    """Saves a comparison of original and reconstructed images to the run-specific folder."""
    with torch.no_grad():
        comparison = torch.cat([original[:8], reconstruction[:8]])
        save_path = os.path.join(run_dir, "reconstructions", f"reconstruction_{step}.png")
        save_image(comparison.cpu(), save_path, nrow=8)

def visualize_dreams(model, start_z, step, agent_id_str, run_dir, dream_len=50):
    """Generates and saves a video of a dream sequence for a specific agent."""
    model.eval()
    with torch.no_grad():
        z = start_z.unsqueeze(0).to(config.DEVICE)
        dream_frames = [model.vae.decoder(z)]
        hidden = None
        action = torch.tensor([0.0, 0.5, 0.0]).unsqueeze(0).to(config.DEVICE) 
        
        for _ in range(dream_len - 1):
            mu, _, hidden = model.transition(z, action, hidden)
            z = mu.squeeze(1) # Dream deterministically using the mean
            dream_frames.append(model.vae.decoder(z))
            
    dream_video = torch.cat(dream_frames, dim=0)
    
    # Construct the correct save path inside the agent-specific dream folder
    dream_dir = os.path.join(run_dir, f"dreams_agent_{agent_id_str}")
    save_path = os.path.join(dream_dir, f"dream_{step}.png")
    
    save_image(dream_video.cpu(), save_path, nrow=10)
    
    # Restore model to training mode
    model.train()
    if hasattr(model, 'vae'):
        model.vae.eval() # Keep VAE frozen if it's a WorldModel

def pre_encode_buffer(model, pixel_buffer):
    """Encodes all observations in the buffer to latent states."""
    print("Pre-encoding replay buffer...")
    model.eval()
    
    latent_buffer = []
    with torch.no_grad():
        for i in tqdm(range(len(pixel_buffer.memory)), desc="Encoding Buffer"):
            obs, action, reward, _, done = pixel_buffer.memory[i]
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(config.DEVICE)
            mu, logvar = model.vae.encoder(obs_tensor)
            z = reparameterize(mu, logvar).squeeze(0).cpu().numpy()
            latent_buffer.append((z, action, reward, done))
            
    model.train()
    return latent_buffer

def plot_loss_curves(losses, title, filename, run_dir):
    """Plots and saves loss curves to the run-specific folder."""
    plt.figure(figsize=(12, 6))
    for name, values in losses.items():
        plt.plot(values, label=name, alpha=0.8)
    plt.title(title)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(run_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss plot to {save_path}")

def generate_and_save_buffer(num_steps, save_path):
    """Generates experience by running a random agent and saves the buffer to disk."""
    print(f"--- Generating {num_steps} steps of experience for {config.ENV_NAME} ---")
    
    env = make_env(config.ENV_NAME)
    buffer = ReplayBuffer(capacity=num_steps) # Capacity is the number of steps to generate
    
    obs, _ = env.reset()
    for _ in tqdm(range(num_steps), desc="Generating Data"):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs if not done else env.reset()[0]
        
    env.close()
    
    # Save the populated buffer to a file
    with open(save_path, 'wb') as f:
        pickle.dump(buffer, f)
    print(f"\nSuccessfully saved replay buffer with {len(buffer)} transitions to {save_path}")
### =================================================================
###                   TRAINING FUNCTION: VAE
### =================================================================

def train_vae_only(run_dir, buffer_path, num_steps=10000):
    """Trains only the VAE component of the World Model."""
    print("--- Training Mode: VAE Only ---")
    
    try:
        with open(buffer_path, 'rb') as f:
            buffer = pickle.load(f)
        print(f"Successfully loaded replay buffer from {buffer_path}")
    except FileNotFoundError:
        print(f"ERROR: Replay buffer not found at '{buffer_path}'.")
        print("Please run --mode data first to generate a buffer.")
        return
    
    model = VAE(latent_dim=config.LATENT_DIM).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    env = make_env(gymnasium.make(config.ENV_NAME, continuous=True))
    
    print(f"Populating buffer with {config.SEED_STEPS} random steps...")
    obs, _ = env.reset()
    for _ in tqdm(range(config.SEED_STEPS), desc="Seeding Buffer"):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs if not done else env.reset()[0]
    
    print(f"\nStarting VAE training for {num_steps} steps...")
    model.train()
    loss_history = []
    
    pbar = tqdm(range(num_steps))
    for step in pbar:
        batch = buffer.sample_transitions(config.BATCH_SIZE)
        if batch is None: continue
        
        obs_batch, _, _, _, _ = batch
        beta = calculate_beta(step, config.BETA_KL_MAX, config.BETA_ANNEAL_STEPS)
        recon_batch, mu, logvar = model(obs_batch)
        loss = vae_loss_function(recon_batch, obs_batch, mu, logvar, beta)
        
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        loss_history.append(loss.item() / config.BATCH_SIZE)
        pbar.set_description(f"Loss: {loss_history[-1]:.2f}, Beta: {beta:.2f}")
        
        if step > 0 and step % config.LOG_INTERVAL == 0:
            save_reconstruction_images(obs_batch, recon_batch, step, run_dir)

    print("VAE training complete.")
    model_save_path = os.path.join(run_dir, "models", "vae_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved final VAE model to {model_save_path}")
    
    plot_loss_curves({"VAE Loss": loss_history}, "VAE Training Loss", "vae_loss_curve.png", run_dir)
    env.close()

### =================================================================
###                TRAINING FUNCTION: DYNAMICS (BASELINE)
### =================================================================

def train_dynamics_baseline(run_dir, vae_run_id, buffer_path, num_steps=10000):
    """Trains the dynamics model (TransitionModel) only, assuming a pre-trained VAE."""
    print("--- Training Mode: Dynamics Baseline ---")
    
    model = WorldModel().to(config.DEVICE)
    vae_path = os.path.join("runs", "vae", vae_run_id, "models", "vae_model.pth")
    try:
        model.load_vae_weights(path=vae_path)
    except FileNotFoundError:
        print(f"ERROR: Pre-trained VAE weights not found at '{vae_path}'.")
        return

    env = make_env(gymnasium.make(config.ENV_NAME, continuous=True))
        # --- UPDATED: Load buffer from file ---
    try:
        with open(buffer_path, 'rb') as f:
            pixel_buffer = pickle.load(f)
        print(f"Successfully loaded replay buffer from {buffer_path}")
    except FileNotFoundError:
        print(f"ERROR: Replay buffer not found at '{buffer_path}'.")
        return
    # ... (data collection)
    
    latent_buffer_list = pre_encode_buffer(model, pixel_buffer)
    del pixel_buffer
    
    optimizer = optim.Adam(model.transition.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=(config.DEVICE.type == 'cuda'))
    
    print(f"\nStarting dynamics model training for {num_steps} steps...")
    model.transition.train()
    model.vae.eval()
    
    loss_history = []
    pbar = tqdm(range(num_steps))
    for step in pbar:
        # ... (training step logic)
        batch_indices = np.random.choice(len(latent_buffer_list) - config.SEQUENCE_LENGTH, size=config.BATCH_SIZE, replace=False)
        sequences = [latent_buffer_list[i : i + config.SEQUENCE_LENGTH] for i in batch_indices]
        z_seq, act_seq, _, _ = zip(*[zip(*seq) for seq in sequences])
        z_batch = torch.from_numpy(np.array(z_seq)).to(config.DEVICE)
        action_batch = torch.from_numpy(np.array(act_seq)).to(config.DEVICE)
        
        with torch.amp.autocast(device_type=config.DEVICE.type, enabled=(config.DEVICE.type == 'cuda')):
            pred_mu_seq, _, _ = model.transition(z_batch[:, :-1, :], action_batch[:, :-1, :])
            target_z_seq = z_batch[:, 1:, :].detach()
            prediction_loss = F.mse_loss(pred_mu_seq, target_z_seq)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(prediction_loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.transition.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        loss_history.append(prediction_loss.item())
        pbar.set_description(f"Prediction Loss: {loss_history[-1]:.4f}")
        
        if step > 0 and step % config.LOG_INTERVAL == 0:
            # For baseline, agent_id_str is just '0'
            visualize_dreams(model, z_batch[0, 0, :], step, '0', run_dir)
            
    print("Dynamics Model training complete.")
    
    model_save_path = os.path.join(run_dir, "models", "dynamics_model.pth")
    torch.save(model.transition.state_dict(), model_save_path)
    print(f"Saved final dynamics model to {model_save_path}")
    
    plot_loss_curves({"Prediction Loss": loss_history}, "Dynamics Model Training Loss", "dynamics_loss_curve.png", run_dir)
    env.close()

### =================================================================
###                   TRAINING FUNCTION: GOSSIP
### =================================================================
def train_gossip(run_dir, vae_run_id, buffer_path, num_agents, num_steps=15000):
    """Trains a 'society' of world models using the gossip protocol."""
    print(f"--- Training Mode: Gossip with {num_agents} agents ---")
    
    print("Initializing model society...")
    models = [WorldModel().to(config.DEVICE) for _ in range(num_agents)]
    optimizers = [optim.Adam(m.transition.parameters(), lr=config.LEARNING_RATE) for m in models]
    
    vae_path = os.path.join("runs", "vae", vae_run_id, "models", "vae_model.pth")
    try:
        for model in models:
            model.load_vae_weights(path=vae_path)
            model.vae.eval() # VAE is pre-trained and frozen
            model.transition.train()
    except FileNotFoundError:
        print(f"ERROR: Pre-trained VAE weights not found at '{vae_path}'.")
        return

    env = make_env(gymnasium.make(config.ENV_NAME, continuous=True))
    try:
        with open(buffer_path, 'rb') as f:
            pixel_buffer = pickle.load(f)
        print(f"Successfully loaded replay buffer from {buffer_path}")
    except FileNotFoundError:
        print(f"ERROR: Replay buffer not found at '{buffer_path}'.")
        return
        
    latent_buffer_list = pre_encode_buffer(models[0], pixel_buffer)
    del pixel_buffer
    
    obs, _ = env.reset()
    for _ in tqdm(range(config.SEED_STEPS), desc="Seeding Buffer"):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        pixel_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs if not done else env.reset()[0]
        
    latent_buffer_list = pre_encode_buffer(models[0], pixel_buffer)
    del pixel_buffer
    
    # FIX 1: Update GradScaler instantiation to the modern API
    scaler = torch.amp.GradScaler( enabled=(config.DEVICE.type == 'cuda'))
    
    print(f"\nStarting gossip training for {num_steps} steps...")
    
    loss_histories = {f"loss_agent_{i}": [] for i in range(num_agents)}

    pbar = tqdm(range(num_steps))
    for step in pbar:
        agent_indices = random.sample(range(num_agents), 2)
        idx_i, idx_j = agent_indices[0], agent_indices[1]
        agent_i, agent_j = models[idx_i], models[idx_j]
        optim_i, optim_j = optimizers[idx_i], optimizers[idx_j]

        start_idx = np.random.randint(0, len(latent_buffer_list) - config.GOSSIP_DREAM_STEPS)
        start_z_numpy, _, _, _ = latent_buffer_list[start_idx]
        start_z = torch.from_numpy(start_z_numpy).unsqueeze(0).to(config.DEVICE)
        
        # Zero gradients for BOTH optimizers before the forward pass
        optim_i.zero_grad(set_to_none=True)
        optim_j.zero_grad(set_to_none=True)

        # Autocast context for the forward pass
        with torch.amp.autocast(device_type=config.DEVICE.type, enabled=(config.DEVICE.type == 'cuda')):
            # FIX 2: THIS IS THE MAIN FIX.
            # The dreaming process MUST have gradients enabled to train the transition models.
            # We remove the `with torch.no_grad():` block that was here before.
            
            # Dream forward with agent i to get its final latent state
            z_i, hidden_i = start_z, None
            action = torch.tensor([0.0, 0.5, 0.0]).unsqueeze(0).to(config.DEVICE) 
            for _ in range(config.GOSSIP_DREAM_STEPS):
                mu_i, _, hidden_i = agent_i.transition(z_i, action, hidden_i)
                z_i = mu_i
            final_z_i = z_i

            # Dream forward with agent j
            z_j, hidden_j = start_z, None
            for _ in range(config.GOSSIP_DREAM_STEPS):
                mu_j, _, hidden_j = agent_j.transition(z_j, action, hidden_j)
                z_j = mu_j
            final_z_j = z_j

            # The targets for the loss are computed from the final dream states.
            # These should be detached so we don't backprop through the VAEs or the other agent's model.
            with torch.no_grad():
                img_i = agent_i.vae.decoder(final_z_i)
                img_j = agent_j.vae.decoder(final_z_j)
                
                mu_j_from_i, _ = agent_j.vae.encoder(img_i)
                mu_i_from_j, _ = agent_i.vae.encoder(img_j)
            
            # Calculate losses based on the gossip protocol
            final_z_i_squeezed = final_z_i.squeeze(1)
            final_z_j_squeezed = final_z_j.squeeze(1)
            
            loss_i = F.mse_loss(final_z_i_squeezed, mu_i_from_j)
            loss_j = F.mse_loss(final_z_j_squeezed, mu_j_from_i)
            
            # Combine the losses before the backward pass
            total_gossip_loss = config.GOSSIP_WEIGHT * (loss_i + loss_j)

        # Scale and backpropagate the combined loss.
        # This will now correctly populate .grad for parameters in both transition models.
        scaler.scale(total_gossip_loss).backward()
        
        # Unscale and step each optimizer individually
        scaler.unscale_(optim_i)
        clip_grad_norm_(agent_i.transition.parameters(), max_norm=1.0)
        scaler.step(optim_i)

        scaler.unscale_(optim_j)
        clip_grad_norm_(agent_j.transition.parameters(), max_norm=1.0)
        scaler.step(optim_j)
        
        # Update the scaler once at the end
        scaler.update()
                
        loss_histories[f"loss_agent_{idx_i}"].append(loss_i.item())
        loss_histories[f"loss_agent_{idx_j}"].append(loss_j.item())
        avg_loss = (loss_i.item() + loss_j.item()) / 2
        pbar.set_description(f"Gossip Loss (avg): {avg_loss:.4f} between agents {idx_i} & {idx_j}")

        if step > 0 and step % (config.LOG_INTERVAL * 5) == 0:
            visualize_dreams(agent_i, start_z.squeeze(0), step, str(idx_i), run_dir)
            visualize_dreams(agent_j, start_z.squeeze(0), step, str(idx_j), run_dir)
            
    print("Gossip training complete.")
    for i, model in enumerate(models):
        model_save_path = os.path.join(run_dir, "models", f"gossip_model_{i}.pth")
        torch.save(model.state_dict(), model_save_path)
    print(f"Saved {num_agents} models to {os.path.join(run_dir, 'models')}")

    plot_loss_curves(loss_histories, "Gossip Training Loss per Agent", "gossip_loss_curve.png", run_dir)
    env.close()