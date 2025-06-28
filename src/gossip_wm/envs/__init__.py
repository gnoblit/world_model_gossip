import gymnasium as gym
from gossip_wm import config

def make_env(env_name):
    """
    Environment Factory.
    """
    print(f"Creating environment: {env_name}")
    
    env_config = config.ENV_CONFIGS[env_name]
    kwargs = env_config.get("GYM_KWARGS", {})

    # Handle special cases for environment creation
    if env_name.startswith("MiniGrid"):
        import minigrid.wrappers
        env = gym.make(env_name, **kwargs)
    elif env_name.startswith("Crafter"):
        import crafter
        env = gym.make(env_name, **kwargs)
    elif env_name.startswith("MineRL"):
        import minerl
        env = gym.make(env_name, **kwargs)
    else:
        env = gym.make(env_name, **kwargs)

    # Apply the correct wrapper
    if env_name == "CarRacing-v3":
        from .car_racing import CarRacingWrapper
        return CarRacingWrapper(env)
    elif env_name == "BipedalWalker-v3":
        from .bipedal_walker import BipedalWalkerWrapper
        return BipedalWalkerWrapper(env)
    elif env_name == "LunarLander-v2":
        from .lunar_lander import LunarLanderWrapper
        return LunarLanderWrapper(env)
    elif env_name.startswith("MiniGrid"):
        from .minigrid import MiniGridWrapper
        return MiniGridWrapper(env)
    elif env_name.startswith("Crafter"):
        from .crafter import CrafterWrapper
        return CrafterWrapper(env)
    elif env_name.startswith("MineRL"):
        from .minerl import MineRLWrapper
        return MineRLWrapper(env)
    else:
        raise ValueError(f"Environment '{env_name}' not supported or wrapper not defined.")

__all__ = ["make_env"]