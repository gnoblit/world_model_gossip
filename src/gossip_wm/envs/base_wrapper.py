import gymnasium as gym
from abc import ABC, abstractmethod # Import the ABC tools

class BaseEnvWrapper(gym.Wrapper, ABC):
    """
    Abstract base class for environment wrappers.
    
    This class uses Python's `abc` module to enforce that any subclass
    MUST implement its own `observation_space` and `action_space` properties,
    as well as the `observation` method.
    
    It relies on the parent `gym.Wrapper` to automatically handle the `step`
    and `reset` logic.
    """
    def __init__(self, env):
        super().__init__(env)

    @property
    @abstractmethod
    def observation_space(self):
        """The observation space of the wrapped environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self):
        """The action space of the wrapped environment."""
        raise NotImplementedError

    @abstractmethod
    def observation(self, obs):
        """The method for preprocessing observations."""
        raise NotImplementedError
    