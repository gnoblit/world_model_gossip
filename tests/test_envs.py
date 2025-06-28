# src/gossip_wm/envs/__init__.py

# No top-level gym import needed here anymore for make
from gossip_wm import config

# Import all your wrappers
from gossip_wm.envs.car_racing_wrapper import CarRacingWrapper
from gossip_wm.envs.bipedal_walker_wrapper import BipedalWalkerWrapper
from gossip_wm.envs.lunar_lander_wrapper import LunarLanderWrapper
from gossip_wm.envs.minigrid_wrapper import MiniGridWrapper
from gossip_wm.envs.vizdoom_wrapper import ViZDoomWrapper 

def make_env(env_name):
    """
    Environment Factory that directly instantiates envs to bypass entry_point conflicts.
    """
    print(f"Creating environment via direct instantiation: {env_name}")
    
    env_config = config.ENV_CONFIGS[env_name]
    kwargs = env_config.get("GYM_KWARGS", {})
    
    # --- Dispatcher to create the base environment ---
    
    if env_name == "CarRacing-v3":
        # Import the specific class
        from gymnasium.envs.box2d.car_racing import CarRacing
        base_env = CarRacing(**kwargs)
        return CarRacingWrapper(base_env)
        
    elif env_name == "BipedalWalker-v3":
        from gymnasium.envs.box2d.bipedal_walker import BipedalWalker
        base_env = BipedalWalker(**kwargs)
        return BipedalWalkerWrapper(base_env)

    elif env_name == "LunarLander-v3":
        # Note: The class name might not match the env ID exactly.
        # It's LunarLander for the ID LunarLander-v2/v3
        from gymnasium.envs.box2d.lunar_lander import LunarLander
        base_env = LunarLander(**kwargs)
        return LunarLanderWrapper(base_env)

    elif env_name.startswith("MiniGrid"):
        # MiniGrid requires its own import
        import minigrid.envs
        # We can still use gym.make here if we are confident the conflict
        # is with another package, but to be safe, we can be explicit.
        # For simplicity and robustness, let's stick to gym.make JUST for minigrid/craftium
        # if their class mappings are complex. A better way is direct import.
        from minigrid.envs.doorkey import DoorKeyEnv
        base_env = DoorKeyEnv(**kwargs) # Assumes DoorKey-8x8
        return MiniGridWrapper(base_env)
        
    elif env_name.startswith("Vizdoom"):
        try:
            # We still need to import craftium to register it, but we can
            # then call make on it specifically.
            import vizdoom
            import gymnasium as gym # Local import to be safe
            base_env = gym.make(env_name, **kwargs)
            return ViZDoomWrapper(base_env)
        except ImportError:
            print("ViZDoom not installed, cannot create environment.")
            raise
            
    else:
        raise ValueError(f"Environment '{env_name}' not supported by direct instantiation.")

__all__ = ["make_env"]