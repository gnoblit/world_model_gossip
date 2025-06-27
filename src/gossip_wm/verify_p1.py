import gymnasium
import torch
from torchvision.utils import save_image
from tqdm import tqdm

# Import our custom modules
from src.gossip_wm.environment import CarRacingWrapper
from src.gossip_wm.replay_buffer import ReplayBuffer
from src.gossip_wm import config # Import the config

def main():
    print("--- Phase 1 Verification (with Config) ---")
    print(f"Using device: {config.DEVICE}")

    # 1. Initialize Environment using config
    base_env = gymnasium.make(config.ENV_NAME, continuous=True)
    env = CarRacingWrapper(base_env)
    
    print(f"Wrapped Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    # 2. Initialize Replay Buffer using config
    buffer = ReplayBuffer(capacity=config.BUFFER_CAPACITY)
    print(f"Replay Buffer initialized with capacity {config.BUFFER_CAPACITY}.")
    
    # 3. Collect some random data using config
    print(f"\nCollecting {config.SEED_STEPS} steps of random data...")
    
    obs, _ = env.reset()
    for _ in tqdm(range(config.SEED_STEPS)):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, done)
        
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
            
    print(f"Data collection complete. Buffer size: {len(buffer)}")
    
    # 4. Sample from the buffer and verify shapes using config
    print(f"\nSampling a batch of size {config.BATCH_SIZE} to verify...")
    
    sample_batch = buffer.sample_transitions(config.BATCH_SIZE)
    
    if sample_batch:
        obs_b, act_b, rew_b, next_obs_b, done_b = sample_batch
        
        print(f"Sampled Observation Batch Shape: {obs_b.shape}")
        
        # --- Expected Shapes ---
        # Obs: (BATCH_SIZE, 1, 64, 64)
        assert obs_b.shape == (config.BATCH_SIZE, 1, *config.RESIZE_DIM), "Obs shape mismatch!"
        
        save_image(obs_b[:8], "verification_image.png", nrow=4)
        print("\nSaved a sample of preprocessed observations to 'verification_image.png'.")
        
    else:
        print("Failed to sample from buffer.")
        
    env.close()
    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    main()