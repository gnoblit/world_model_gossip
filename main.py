# main.py

import argparse
import os
from datetime import datetime
import secrets
import json
import torch # Import torch to check for device type

from gossip_wm import config
from gossip_wm.training import train_vae_only, train_dynamics_baseline, train_gossip, generate_and_save_buffer

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
    base_run_dir = os.path.join("runs", mode, run_id)
    os.makedirs(os.path.join(base_run_dir, "models"), exist_ok=True)
    if mode == "vae":
        os.makedirs(os.path.join(base_run_dir, "reconstructions"), exist_ok=True)
    elif mode == "dynamics":
        # The training function creates the agent-specific dream folders
        pass
    elif mode == "gossip":
        # The training function creates the agent-specific dream folders
        pass
    print(f"Created run directory structure in: {base_run_dir}")
    return base_run_dir


def main():
    parser = argparse.ArgumentParser(description="Generate data or train a Gossip World Model.")
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=["data", "vae", "dynamics", "gossip"],
        help="The action to perform."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20000,
        help="Number of steps for data generation or training."
    )
    # UPDATED: No more default buffer path, as it's now dynamic
    parser.add_argument(
        "--buffer_path",
        type=str,
        default=None,
        help="Path to a specific replay buffer. If not set, a default path will be used based on the environment."
    )
    parser.add_argument(
        "--env_name", 
        type=str, 
        default=None,
        help="Specify a single environment. Overrides the config default. Required for training modes."
    )
    parser.add_argument("--num_agents", type=int, default=config.GOSSIP_NUM_AGENTS)
    parser.add_argument("--vae_run_id", type=str, help="The run_id of the pre-trained VAE to use for dynamics/gossip training.")
    
    args = parser.parse_args()

    # Handle data generation mode first
    if args.mode == "data":
        os.makedirs("data", exist_ok=True)
        
        if args.env_name:
            # Generate for a single specified environment
            if args.env_name not in config.ENV_CONFIGS:
                parser.error(f"--env_name '{args.env_name}' is not a valid environment.")
            envs_to_generate = [args.env_name]
        else:
            # Generate for all environments
            envs_to_generate = list(config.ENV_CONFIGS.keys())
            print("--- Generating data for ALL supported environments. ---")

        for env_name in envs_to_generate:
            config.ENV_NAME = env_name # Set the global config for this iteration
            buffer_path = f"data/buffer_{env_name}.pkl"
            generate_and_save_buffer(num_steps=args.steps, save_path=buffer_path)
            
        return # Exit after generating data

    # --- The rest of the logic is for training modes ---
    
    # For training modes, an environment must be specified.
    if not args.env_name:
        parser.error("--env_name is required for training modes (vae, dynamics, gossip).")
    if args.env_name not in config.ENV_CONFIGS:
        parser.error(f"Specified --env_name '{args.env_name}' is not a valid environment.")
    
    # Set the global environment name for the entire training run
    config.ENV_NAME = args.env_name
    print(f"--- Running in mode '{args.mode}' for environment '{config.ENV_NAME}' ---")

    # Determine the buffer path to use for training
    buffer_path = args.buffer_path or f"data/buffer_{config.ENV_NAME}.pkl"
    if not os.path.exists(buffer_path):
        parser.error(f"Replay buffer not found at '{buffer_path}'. Please generate it first with --mode data.")
    print(f"Using replay buffer: {buffer_path}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    unique_hex = secrets.token_hex(3)
    run_id = f"{config.ENV_NAME}_{timestamp}_{unique_hex}"
    
    run_dir = setup_run_directories(args.mode, run_id, args.num_agents)
    save_config(run_dir, args)

    if args.mode == "vae":
        train_vae_only(run_dir=run_dir, buffer_path=buffer_path, num_steps=args.steps)
        
    elif args.mode == "dynamics":
        if not args.vae_run_id: parser.error("--vae_run_id is required for 'dynamics' mode.")
        train_dynamics_baseline(run_dir=run_dir, vae_run_id=args.vae_run_id, buffer_path=buffer_path, num_steps=args.steps)
        
    elif args.mode == "gossip":
        if not args.vae_run_id: parser.error("--vae_run_id is required for 'gossip' mode.")
        train_gossip(
            run_dir=run_dir, 
            vae_run_id=args.vae_run_id, 
            buffer_path=buffer_path,
            num_agents=args.num_agents, 
            num_steps=args.steps
        )
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()