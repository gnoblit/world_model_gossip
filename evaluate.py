# src/gossip_wm/evaluate.py

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import imageio
from tqdm import tqdm
import gymnasium as gym
import re
import pathlib
import cv2

# Import from our library
from gossip_wm.models import WorldModel, reparameterize
from gossip_wm import config
from src.gossip_wm.envs import make_env

def load_world_model(model_path, device):
    """
    Loads a trained WorldModel from a state dictionary.
    This function intelligently handles different saved formats.
    """
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
        
    state_dict = torch.load(model_path, map_location=device)
    
    model = WorldModel().to(device)
    
    if 'vae.encoder.conv1.weight' in state_dict:
        model.load_state_dict(state_dict)
    elif 'encoder.conv1.weight' in state_dict:
        # This case is for loading a standalone VAE model, if needed.
        model.vae.load_state_dict(state_dict)
    elif 'gru.weight_ih_l0' in state_dict:
        # This is a dynamics-only model. The VAE will be loaded separately.
        model.transition.load_state_dict(state_dict)
    else:
        raise ValueError("Unknown model state_dict format.")
        
    model.eval()
    print("Model loaded successfully.")
    return model

def generate_dream_sequence(model, start_obs, num_frames=500, stochastic=False, dynamic_action=False):
    """
    Generates a sequence of dream frames from a starting observation.
    """
    print(f"Generating a dream of {num_frames} frames (stochastic={stochastic}, dynamic_action={dynamic_action})...")
    
    dream_frames = []
    upscale_dim = (256, 256)

    env_conf = config.get_env_config()
    is_discrete = env_conf['IS_DISCRETE']
    action_dim = env_conf['ACTION_DIM']

    with torch.no_grad():
        start_obs_tensor = torch.from_numpy(start_obs).unsqueeze(0).to(config.DEVICE)
        mu_start, _ = model.vae.encoder(start_obs_tensor)
        z = mu_start
        hidden = None
        
        for i in tqdm(range(num_frames), desc="Dreaming..."):
            frame_tensor = model.vae.decoder(z)
            frame_np = frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            rgb_frame = np.repeat(frame_np, 3, axis=2)
            upscaled_frame = cv2.resize(rgb_frame, upscale_dim, interpolation=cv2.INTER_NEAREST)
            dream_frames.append((upscaled_frame * 255).astype(np.uint8))

            if dynamic_action:
                if is_discrete:
                    # Cycle through first 5 movement actions (nop, fwd, back, left, right)
                    action_idx = (i // 10) % 5
                    action = F.one_hot(torch.tensor(action_idx), num_classes=action_dim).float().unsqueeze(0).to(config.DEVICE)
                else:
                    steer = np.sin(i * 0.1) * 0.6
                    action = torch.tensor([steer, 0.3, 0.0], dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
            else:
                if is_discrete:
                    action_idx = 1 # Use "forward" as a default action
                    action = F.one_hot(torch.tensor(action_idx), num_classes=action_dim).float().unsqueeze(0).to(config.DEVICE)
                else:
                    action = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float32).unsqueeze(0).to(config.DEVICE)

            mu, logvar, hidden = model.transition(z, action, hidden)
            
            if stochastic:
                z = reparameterize(mu.squeeze(1), logvar.squeeze(1))
            else:
                z = mu.squeeze(1)

    return dream_frames

def get_run_id_from_path(path):
    """
    Extracts the run_id (e.g., '2024-06-28_18-00_a1b2c3') from a model path.
    Returns the hex part of the ID.
    """
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_[a-f0-9]{6})', path)
    if match:
        run_id = match.group(1)
        hex_part = run_id.split('_')[-1]
        return hex_part
    return "unknown"

def calculate_and_print_variance(frames, model_name):
    """
    Calculates the variance of pixel values across all frames in a sequence.
    """
    if not frames:
        print(f"No frames generated for Model {model_name}. Cannot calculate variance.")
        return
    frames_array = np.array(frames, dtype=np.float32) / 255.0
    mean_variance = np.mean(np.var(frames_array, axis=0))
    print(f"\n--- Diagnostic: Variance for Model {model_name} ---")
    print(f"Mean variance across all frames: {mean_variance:.8f}")
    if mean_variance < 1e-6:
        print("!! WARNING: Variance is zero or near-zero. The generated frames are static. !!")
    else:
        print("-> OK: The dream is evolving.")
    print("------------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare World Models by generating long dreams.")
    parser.add_argument("--model_path_a", required=True, type=str, help="Path to the first model's state_dict.")
    parser.add_argument("--model_path_b", required=True, type=str, help="Path to the second model's state_dict.")
    # NEW: Add a required argument for the VAE run ID.
    parser.add_argument("--vae_run_id", required=True, type=str, help="Run ID of the pre-trained VAE used for training dynamics models.")
    parser.add_argument("--num_frames", type=int, default=500, help="Number of frames to dream.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the environment's starting state.")
    parser.add_argument("--stochastic", action='store_true', help="Enable stochastic dreaming.")
    parser.add_argument("--dynamic_action", action='store_true', help="Use a dynamic, sinusoidal steering action.")
    args = parser.parse_args()

    device = config.DEVICE
    print(f"Using device: {device}")
    
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
    EVAL_DIR = PROJECT_ROOT / "evaluations"
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    env = make_env(config.ENV_NAME) 
    start_obs, _ = env.reset(seed=args.seed)
    env.close()

    # --- Load Models and VAE ---
    model_a = load_world_model(args.model_path_a, device)
    model_b = load_world_model(args.model_path_b, device)

    # NEW: Logic to load the pre-trained VAE for any dynamics-only models.
    vae_path = PROJECT_ROOT / "runs" / "vae" / args.vae_run_id / "models" / "vae_model.pth"
    if not vae_path.exists():
        raise FileNotFoundError(f"Required VAE model not found at {vae_path}")
    
    print(f"\nLoading shared VAE from: {vae_path}")
    vae_state_dict = torch.load(vae_path, map_location=device)

    # Check Model A
    state_dict_a = torch.load(args.model_path_a, map_location=device)
    if 'gru.weight_ih_l0' in state_dict_a and 'vae.encoder.conv1.weight' not in state_dict_a:
        print("Model A is a dynamics-only model. Loading shared VAE weights...")
        model_a.vae.load_state_dict(vae_state_dict)

    # Check Model B
    state_dict_b = torch.load(args.model_path_b, map_location=device)
    if 'gru.weight_ih_l0' in state_dict_b and 'vae.encoder.conv1.weight' not in state_dict_b:
        print("Model B is a dynamics-only model. Loading shared VAE weights...")
        model_b.vae.load_state_dict(vae_state_dict)
    
    # --- Generate Dreams ---
    dream_a = generate_dream_sequence(model_a, start_obs, args.num_frames, args.stochastic, args.dynamic_action)
    dream_b = generate_dream_sequence(model_b, start_obs, args.num_frames, args.stochastic, args.dynamic_action)
    
    calculate_and_print_variance(dream_a, "A")
    calculate_and_print_variance(dream_b, "B")
    
    print("Stitching dreams into a GIF...")
    combined_frames = []
    for frame_a, frame_b in zip(dream_a, dream_b):
        separator = np.zeros((frame_a.shape[0], 10, 3), dtype=np.uint8)
        combined_frame = np.concatenate([frame_a, separator, frame_b], axis=1)
        combined_frames.append(combined_frame)
        
    hex_a = get_run_id_from_path(args.model_path_a)
    hex_b = get_run_id_from_path(args.model_path_b)
    stochastic_tag = "_stochastic" if args.stochastic else ""
    action_tag = "_dynamic" if args.dynamic_action else ""
    output_filename = f"{hex_a}_vs_{hex_b}_seed{args.seed}{stochastic_tag}{action_tag}.gif"
    output_path = EVAL_DIR / output_filename
    
    imageio.mimsave(str(output_path), combined_frames, plugin="pillow", fps=20, loop=0)
    print(f"Successfully saved comparison GIF to {output_path}")

if __name__ == "__main__":
    main()