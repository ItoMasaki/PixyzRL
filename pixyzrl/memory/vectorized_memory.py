import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer

from .memory import RolloutBuffer


class VectorizedRolloutBuffer(RolloutBuffer):
    """Rollout buffer for vectorized environments using torchrl."""

    def __init__(self, obs_shape: tuple[int], action_shape: tuple[int], buffer_size: int, num_envs: int, device: str = "cpu") -> None:
        """
        Initialize the rollout buffer.

        Args:
            obs_shape (tuple): Shape of the observations.
            action_shape (tuple): Shape of the actions.
            buffer_size (int): Maximum size of the buffer.
            num_envs (int): Number of parallel environments.
            device (str): Device to store the tensors (cpu or cuda).
        """
        super().__init__(obs_shape, action_shape, buffer_size, device)
        self.num_envs = num_envs
        self.storage = LazyTensorStorage(buffer_size * num_envs)
        self.buffer = ReplayBuffer(storage=self.storage, batch_size=buffer_size * num_envs)

    def add(self, data: TensorDict) -> None:
        """
        Add new experiences to the buffer.

        Args:
            data (TensorDict): A dictionary containing observations, actions, rewards, done flags, log probabilities, and values.
        """
        obs = data["obs"]
        action = data["action"]
        reward = data["reward"]
        done = data["done"]
        log_prob = data["log_prob"]
        value = data["value"]

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        if isinstance(reward, np.ndarray):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(-1)
        if isinstance(done, np.ndarray):
            done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(-1)
        if isinstance(log_prob, np.ndarray):
            log_prob = torch.tensor(log_prob, dtype=torch.float32, device=self.device).unsqueeze(-1)
        if isinstance(value, np.ndarray):
            value = torch.tensor(value, dtype=torch.float32, device=self.device).unsqueeze(-1)

        data = TensorDict(
            {
                "obs": obs,
                "action": action,
                "reward": reward,
                "done": done,
                "log_prob": log_prob,
                "value": value,
            },
            batch_size=self.num_envs,
        )

        self.buffer.extend(data)

    def sample(self) -> TensorDict:
        """
        Sample all experiences in the buffer.

        Returns:
            TensorDict: A batch of sampled experiences.
        """
        return self.buffer.sample().to(self.device)

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = ReplayBuffer(storage=self.storage, batch_size=self.buffer_size * self.num_envs)

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
