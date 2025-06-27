# main.py

import argparse
import os
from datetime import datetime
import secrets

from gossip_wm.training import train_vae_only, train_dynamics_baseline

def setup_run_directories(mode, run_id):
    """
    Creates a tailored directory structure for a specific run based on its mode.
    """
    base_run_dir = os.path.join("runs", mode, run_id)
    
    # Always create a directory for the models
    os.makedirs(os.path.join(base_run_dir, "models"), exist_ok=True)
    
    # Create specific output folders based on the training mode
    if mode == "vae":
        os.makedirs(os.path.join(base_run_dir, "reconstructions"), exist_ok=True)
    elif mode == "dynamics":
        os.makedirs(os.path.join(base_run_dir, "dreams"), exist_ok=True)
    elif mode == "gossip":
        # The gossip run will produce reconstructions and dreams for both models
        os.makedirs(os.path.join(base_run_dir, "reconstructions"), exist_ok=True)
        os.makedirs(os.path.join(base_run_dir, "dreams_A"), exist_ok=True)
        os.makedirs(os.path.join(base_run_dir, "dreams_B"), exist_ok=True)
        
    print(f"Created run directory structure in: {base_run_dir}")
    return base_run_dir

def main():
    parser = argparse.ArgumentParser(description="Train a Gossip World Model.")
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=["vae", "dynamics", "gossip"],
        help="The training mode to run."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Number of training steps to run."
    )
    parser.add_argument(
        "--vae_run_id",
        type=str,
        help="The run_id of the pre-trained VAE to use for 'dynamics' or 'gossip' mode."
    )
    
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    unique_hex = secrets.token_hex(3)
    run_id = f"{timestamp}_{unique_hex}"
    
    # Create the tailored directories for this run
    run_dir = setup_run_directories(args.mode, run_id)

    if args.mode == "vae":
        train_vae_only(run_dir=run_dir, num_steps=args.steps)
        
    elif args.mode == "dynamics":
        if not args.vae_run_id:
            parser.error("--vae_run_id is required for 'dynamics' mode.")
        train_dynamics_baseline(run_dir=run_dir, vae_run_id=args.vae_run_id, num_steps=args.steps)
        
    elif args.mode == "gossip":
        if not args.vae_run_id:
            parser.error("--vae_run_id is required for 'gossip' mode.")
        print("Gossip training mode not yet implemented.")
        
    else:
        # This case is technically unreachable due to `choices` in argparse, but good practice
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()