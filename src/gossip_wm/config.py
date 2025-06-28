### Stores parameters

import torch

# --- Global Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Environment ---
ENV_NAME = "CarRacing-v3" # Variable to change to determine environment
RESIZE_DIM = (64, 64)

# --- Training Hyperparameters (mostly shared across envs) ---
BUFFER_CAPACITY = 200000
BATCH_SIZE = 64
SEQUENCE_LENGTH = 50
LEARNING_RATE = 1e-4
BETA_KL_MAX = 1.0
BETA_ANNEAL_STEPS = 2000
GOSSIP_WEIGHT = 0.1
GOSSIP_DREAM_STEPS = 25
GOSSIP_NUM_AGENTS = 2
TRANSITION_HIDDEN_DIM = 256
LATENT_DIM = 32

# --- Data Collection ---
# Number of random steps to seed the buffer before training starts.
SEED_STEPS = 5000

# --- Verification & Logging ---
SAVE_MODEL_INTERVAL = 1000
LOG_INTERVAL = 200

# ==============================================================================
#                      HIERARCHICAL ENVIRONMENT CONFIGS
# ==============================================================================
# This dictionary holds all environment-specific parameters.
# The main code will dynamically pull from this based on ENV_NAME.
# ==============================================================================

ENV_CONFIGS = {
    "CarRacing-v3": {
        "ACTION_DIM": 3,
        "IS_DISCRETE": False,
        "GYM_KWARGS": {"continuous": True}
    },
    "BipedalWalker-v3": {
        "ACTION_DIM": 4,
        "IS_DISCRETE": False,
        "GYM_KWARGS": {"render_mode": "rgb_array"}
    },
    # --- Placeholders for our future environments ---
    "LunarLander-v3": { # CORRECTED: Changed v2 to v3
        "ACTION_DIM": 4, # 0:noop, 1:fire-left, 2:fire-main, 3:fire-right
        "IS_DISCRETE": True,
        "GYM_KWARGS": {"render_mode": "rgb_array", "continuous": False}
    },
    "MiniGrid-DoorKey-8x8-v0": {
        "ACTION_DIM": 7, # turn-L, turn-R, fwd, pickup, drop, toggle, done
        "IS_DISCRETE": True,
        "GYM_KWARGS": {} # No special kwargs needed
    },
    "VizdoomBasic-v0": {
        "ACTION_DIM": 3, # 0: ATTACK, 1: MOVE_RIGHT, 2: MOVE_LEFT
        "IS_DISCRETE": True,
        "GYM_KWARGS": {} # No special kwargs needed
    }
}

# --- Helper function to get the current environment's config ---
def get_env_config():
    """Returns the configuration dictionary for the currently selected ENV_NAME."""
    return ENV_CONFIGS[ENV_NAME]