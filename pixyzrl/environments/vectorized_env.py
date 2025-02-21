from typing import Callable

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv


class VectorizedEnv:
    """
    Vectorized environment wrapper using Gymnasium's SyncVectorEnv.

    This class allows running multiple environments in parallel to speed up training.
    """

    def __init__(self, env_name: str, num_envs: int, seed: int = 42) -> None:
        """
        Initialize the vectorized environments.

        :param env_name: The name of the gym environment.
        :param num_envs: Number of environments to run in parallel.
        :param seed: Random seed for environment initialization.
        """
        self.env_name = env_name
        self.num_envs = num_envs
        self.seed = seed

        # Create vectorized environments
        self.envs = SyncVectorEnv([self.make_env(env_name, seed + i) for i in range(num_envs)])

    def make_env(self, env_name: str, seed: int) -> Callable[[], gym.Env]:  # type: ignore
        """
        Create an environment with a specific seed.
        """

        def _init():
            env = gym.make(env_name)
            env.reset(seed=seed)
            return env

        return _init

    def reset(self):
        """Reset all environments and return the initial observations."""
        return self.envs.reset()

    def step(self, actions):
        """
        Take a step in all environments with the given actions.

        :param actions: List of actions for each environment.
        :return: Tuple of (observations, rewards, dones, info)
        """
        return self.envs.step(actions)

    def close(self) -> None:
        """Close all environments."""
        self.envs.close()

    def get_num_envs(self) -> int:
        """Return the number of parallel environments."""
        return self.num_envs
