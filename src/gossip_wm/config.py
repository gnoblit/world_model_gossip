### Stores parameters

import torch

# --- Global Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Environment ---
ENV_NAME = "CarRacing-v3"
RESIZE_DIM = (64, 64)

# --- Replay Buffer ---
BUFFER_CAPACITY = 200000  # More realistic capacity for a full run
BATCH_SIZE = 64
SEQUENCE_LENGTH = 50      # For training the transition model

# --- Model Architecture ---
LATENT_DIM = 32           # Size of the VAE's latent vector z
ACTION_DIM = 3            # Size of the action space for CarRacing-v3 continuous
TRANSITION_HIDDEN_DIM = 256 # Hidden state size of the GRU/LSTM in the transition model

# --- Training ---
LEARNING_RATE = 1e-4
# BETA_KL is now the *maximum* value. We will anneal towards it.
BETA_KL_MAX = 1.0
BETA_ANNEAL_STEPS = 2000 # Number of steps to go from beta=0 to beta=BETA_KL_MAX
# Weight for our novel gossip/consistency loss.
# Start with a value to make it influential but not overpowering.
GOSSIP_WEIGHT = 0.1
# The number of steps to "dream" forward for the gossip protocol.
GOSSIP_DREAM_STEPS = 25

# --- Data Collection ---
# Number of random steps to seed the buffer before training starts.
SEED_STEPS = 5000

# --- Verification & Logging ---
# How often (in training steps) to save model checkpoints
SAVE_MODEL_INTERVAL = 1000
# How often (in training steps) to log images and other metrics
LOG_INTERVAL = 200