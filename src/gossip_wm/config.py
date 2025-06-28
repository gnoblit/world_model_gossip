# src/gossip_wm/config.py
### Stores parameters

import torch

# ==============================================================================
#                      PRIMARY CONTROL KNOB
# ==============================================================================
# Change this variable to switch between environments.
ENV_NAME = "CarRacing-v3" 
# ==============================================================================

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Global Training Hyperparameters (mostly shared across envs) ---
BUFFER_CAPACITY = 200000
BATCH_SIZE = 64
SEQUENCE_LENGTH = 50
LEARNING_RATE = 1e-4
BETA_KL_MAX = 1.0
BETA_ANNEAL_STEPS = 2000
GOSSIP_WEIGHT = 0.1
GOSSIP_DREAM_STEPS = 25
GOSSIP_NUM_AGENTS = 2

# --- Data Collection ---
SEED_STEPS = 5000

# --- Verification & Logging ---
SAVE_MODEL_INTERVAL = 1000
LOG_INTERVAL = 200


# ==============================================================================
#               HIERARCHICAL ENVIRONMENT & MODEL CONFIGS
# ==============================================================================
# This dictionary holds all environment-specific parameters.
# ==============================================================================

ENV_CONFIGS = {
    "CarRacing-v3": {
        "ACTION_DIM": 3,
        "IS_DISCRETE": False,
        "LATENT_DIM": 32,
        "TRANSITION_HIDDEN_DIM": 256,
        "RESIZE_DIM": (64, 64),
        "GYM_KWARGS": {"continuous": True}
    },
    "BipedalWalker-v3": {
        "ACTION_DIM": 4,
        "IS_DISCRETE": False,
        "LATENT_DIM": 32,
        "TRANSITION_HIDDEN_DIM": 256,
        "RESIZE_DIM": (64, 64),
        "GYM_KWARGS": {"render_mode": "rgb_array"}
    },
    "LunarLander-v3": {
        "ACTION_DIM": 4, 
        "IS_DISCRETE": True,
        "LATENT_DIM": 32,
        "TRANSITION_HIDDEN_DIM": 256,
        "RESIZE_DIM": (64, 64),
        "GYM_KWARGS": {"render_mode": "rgb_array", "continuous": False}
    },
    "MiniGrid-DoorKey-8x8-v0": {
        "ACTION_DIM": 7,
        "IS_DISCRETE": True,
        "LATENT_DIM": 32,
        "TRANSITION_HIDDEN_DIM": 128,
        "RESIZE_DIM": (64, 64),
        "GYM_KWARGS": {} 
    },
    "VizdoomBasic-v0": {
        "ACTION_DIM": 4, # CORRECTED: Changed from 3 to 4 to match the wrapper's action space.
        "IS_DISCRETE": True,
        "LATENT_DIM": 64,
        "TRANSITION_HIDDEN_DIM": 512,
        "RESIZE_DIM": (64, 64),
        "GYM_KWARGS": {}
    }
}

# --- Helper function to get the current environment's config ---
def get_env_config():
    """Returns the configuration dictionary for the currently selected ENV_NAME."""
    if ENV_NAME not in ENV_CONFIGS:
        raise ValueError(f"Configuration for environment '{ENV_NAME}' not found.")
    return ENV_CONFIGS[ENV_NAME]

# --- DYNAMIC CONFIG ACCESS ---
# This allows accessing environment-specific configs like global variables.
# For example, you can use `config.LATENT_DIM` instead of `config.get_env_config()['LATENT_DIM']`.
# This is a bit of Python magic that makes the config system more convenient.
def __getattr__(name: str):
    """
    Dynamically retrieves an attribute from the current environment's config
    dictionary if it's not found as a global variable in this module.
    """
    # This prevents infinite recursion issues with some internal Python mechanics
    if name.startswith('__'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
        
    try:
        # Get the config dictionary for the currently set ENV_NAME
        env_config = get_env_config()
        if name in env_config:
            return env_config[name]
    except (KeyError, ValueError):
        # This can happen if ENV_NAME is not set or invalid.
        # Let the AttributeError bubble up naturally.
        pass

    # If the attribute is not in the environment config, raise the standard error.
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")