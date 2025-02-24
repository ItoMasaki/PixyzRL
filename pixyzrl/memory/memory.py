from typing import Any

import torch
from numpy.typing import NDArray


class BaseBuffer:
    """Base class for replay buffers.

    This class provides a simple interface for storing and sampling experiences.

    Args:
        buffer_size (int): Size of the replay buffer.
        env_dict (Dict[str, Any]): Environment dictionary for replay buffer.
        device (str): Device to store the replay buffer.
        n_step (int): Number of steps for n-step returns.

    Example:
        >>> buffer = BaseBuffer(1000, {"obs": {"shape": (4,)}, "action": {"shape": (1,)}, "reward": {"shape": (1,)}, "done": {"shape": (1,)}}, "cpu", 1)
    """

    def __init__(self, buffer_size: int, env_dict: dict[str, Any], device: str, n_envs: int = 1) -> None:
        """
        Initialize the replay buffer with flexible env_dict settings.

        Args:
            buffer_size (int): Size of the replay buffer.
            env_dict (Dict[str, Any]): Environment dictionary for replay buffer.
            device (str): Device to store the replay buffer.
            n_step (int): Number of steps for n-step returns.

        Example:
            >>> buffer = BaseBuffer(1000, {"obs": {"shape": (4,)}, "action": {"shape": (1,)}, "reward": {"shape": (1,)}, "done": {"shape": (1,)}}, "cpu", 1)
        """
        self.buffer = {}

        self.buffer_size = buffer_size
        self.env_dict = env_dict
        self.device = device
        self.n_envs = n_envs
        self.pos = 0

        for k, v in env_dict.items():
            self.buffer[k] = torch.empty((buffer_size, n_envs, *v["shape"]), dtype=v.get("dtype", torch.float32), device=device)

    def add(self, **kwargs: dict[str, torch.Tensor | NDArray[Any]]) -> None:
        """Add a new experience to the buffer.

        Args:
            **kwargs (dict[str, Any]): Key-value pairs of experience data.

        Example:
            >>> buffer.add(obs=obs, action=action, reward=reward, done=done)
        """
        self.pos = (self.pos + 1) % self.buffer_size
        for k, v in kwargs.items():
            self.buffer[k][self.pos] = v.reshape(self.n_envs, *v.shape)

    def sample(self, batch_size: int) -> dict[str, Any]:
        """Sample a batch of experiences.

        Args:
            batch_size (int): Number of samples per batch.

        Returns:
            dict[str, Any]: Sampled batch of experiences.

        Example:
            >>> batch = buffer.sample(32)
        """
        idx = torch.randint(0, self.buffer_size, (batch_size,))
        return {k: v[idx].to(self.device) for k, v in self.buffer.items()}

    def clear(self) -> None:
        """Clear the buffer.

        Example
            >>> buffer.clear()
        """
        for k in self.buffer:
            self.buffer[k].zero_()
        self.pos = 0

    def __len__(self) -> int:
        """Return the number of stored experiences.

        Returns:
            int: Number of stored experiences.

        Example:
            >>> len(buffer)
        """
        return self.pos


class RolloutBuffer(BaseBuffer):
    """Rollout buffer for storing trajectories.

    This class provides a simple interface for storing and sampling trajectories.

    Args:
        buffer_size (int): Size of the replay buffer.
        env_dict (Dict[str, Any]): Environment dictionary for replay buffer.
        device (str): Device to store the replay buffer.
        n_step (int): Number of steps for n-step returns.

    Example:
        >>> buffer = RolloutBuffer(1000, {"obs": {"shape": (4,)}, "action": {"shape": (1,)}, "reward": {"shape": (1,)}, "done": {"shape": (1,)}}, "cpu", 1)
    """

    def __init__(self, buffer_size: int, env_dict: dict[str, Any], device: str, n_envs: int = 1) -> None:
        """
        Initialize the replay buffer with flexible env_dict settings.

        Args:
            buffer_size (int): Size of the replay buffer.
            env_dict (Dict[str, Any]): Environment dictionary for replay buffer.
            device (str): Device to store the replay buffer.
            n_step (int): Number of steps for n-step returns.

        Example:
            >>> buffer = RolloutBuffer(1000, {"obs": {"shape": (4,)}, "action": {"shape": (1,)}, "reward": {"shape": (1,)}, "done": {"shape": (1,)}}, "cpu", 1)
        """
        super().__init__(buffer_size, env_dict, device, n_envs)

    def 
