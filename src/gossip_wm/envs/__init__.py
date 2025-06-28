from gossip_wm import config

# Import all your wrappers
from .car_racing_wrapper import CarRacingWrapper
from .bipedal_walker_wrapper import BipedalWalkerWrapper
from .lunar_lander_wrapper import LunarLanderWrapper
from .minigrid_wrapper import MiniGridWrapper
from .vizdoom_wrapper import ViZDoomWrapper  # <-- ADDED

def make_env(env_name):
    """
    Environment Factory that directly instantiates envs to bypass entry_point conflicts.
    """
    print(f"Creating environment via direct instantiation: {env_name}")
    
    env_config = config.ENV_CONFIGS[env_name]
    kwargs = env_config.get("GYM_KWARGS", {})
    
    # --- Dispatcher to create the base environment ---
    if env_name == "CarRacing-v3":
        from gymnasium.envs.box2d.car_racing import CarRacing
        base_env = CarRacing(**kwargs)
        return CarRacingWrapper(base_env)
        
    elif env_name == "BipedalWalker-v3":
        from gymnasium.envs.box2d.bipedal_walker import BipedalWalker
        base_env = BipedalWalker(**kwargs)
        return BipedalWalkerWrapper(base_env)

    elif env_name == "LunarLander-v3":
        from gymnasium.envs.box2d.lunar_lander import LunarLander
        base_env = LunarLander(**kwargs)
        return LunarLanderWrapper(base_env)

    elif env_name.startswith("MiniGrid"):
        # For simplicity, we can trust gym.make for a single, well-behaved package
        import gymnasium as gym
        import minigrid
        base_env = gym.make(env_name, **kwargs)
        return MiniGridWrapper(base_env)
        
    elif env_name.startswith("Vizdoom"):  # <-- ADDED VIZDOOM
        import gymnasium as gym
        import vizdoom.gymnasium_wrapper # This import registers the envs
        base_env = gym.make(env_name, **kwargs)
        return ViZDoomWrapper(base_env)
            
    else:
        raise ValueError(f"Environment '{env_name}' not supported by direct instantiation.")

__all__ = ["make_env"]