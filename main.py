# main.py

import argparse
import os
from datetime import datetime
import secrets
import json
import torch # Import torch to check for device type

from gossip_wm import config
from gossip_wm.training import train_vae_only, train_dynamics_baseline, train_gossip

def save_config(run_dir, args):
    """Saves the configuration of a run to a JSON file."""
    config_dict = {}
    
    # 1. Get all hyperparameters from the config module
    for key in dir(config):
        if key.isupper():
            value = getattr(config, key)
            # Convert non-serializable types to strings
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
            
    # 2. Add command-line arguments to the dictionary
    config_dict['args'] = vars(args)

    # 3. Save to a file
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4, sort_keys=True)
    print(f"Saved run configuration to {config_path}")


def setup_run_directories(mode, run_id, num_agents):
    # ... (this function is correct and remains the same)
    base_run_dir = os.path.join("runs", mode, run_id)
    os.makedirs(os.path.join(base_run_dir, "models"), exist_ok=True)
    if mode == "vae":
        os.makedirs(os.path.join(base_run_dir, "reconstructions"), exist_ok=True)
    elif mode == "dynamics":
        os.makedirs(os.path.join(base_run_dir, "dreams"), exist_ok=True)
    elif mode == "gossip":
        for i in range(num_agents):
            os.makedirs(os.path.join(base_run_dir, f"dreams_agent_{i}"), exist_ok=True)
    print(f"Created run directory structure in: {base_run_dir}")
    return base_run_dir


def main():
    # ... (the main function remains exactly the same as the previous version)
    parser = argparse.ArgumentParser(description="Train a Gossip World Model.")
    parser.add_argument("--mode", type=str, required=True, choices=["vae", "dynamics", "gossip"])
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps.")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents in the gossip society.")
    parser.add_argument("--vae_run_id", type=str, help="The run_id of the pre-trained VAE to use.")
    
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    unique_hex = secrets.token_hex(3)
    run_id = f"{timestamp}_{unique_hex}"
    
    run_dir = setup_run_directories(args.mode, run_id, args.num_agents)
    
    save_config(run_dir, args)

    if args.mode == "vae":
        train_vae_only(run_dir=run_dir, num_steps=args.steps)
    elif args.mode == "dynamics":
        if not args.vae_run_id: parser.error("--vae_run_id is required for 'dynamics' mode.")
        train_dynamics_baseline(run_dir=run_dir, vae_run_id=args.vae_run_id, num_steps=args.steps)
    elif args.mode == "gossip":
        if not args.vae_run_id: parser.error("--vae_run_id is required for 'gossip' mode.")
        train_gossip(run_dir=run_dir, vae_run_id=args.vae_run_id, num_agents=args.num_agents, num_steps=args.steps)
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()