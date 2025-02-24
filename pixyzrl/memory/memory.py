from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pixyz.distributions import Distribution


class BaseBuffer:
    """Base class for replay buffers.

    This class provides a simple interface for storing and sampling experiences.

    Args:
        buffer_size (int): Size of the replay buffer.
        env_dict (Dict[str, Any]): Environment dictionary for replay buffer.
        device (str): Device to store the replay buffer.
        n_step (int): Number of steps for n-step returns.

    Example:
        >>> buffer_size = 1000
        >>> env_dict = {"obs": {"shape": (4,)}, "action": {"shape": (1,)}, "reward": {"shape": (1,)}, "done": {"shape": (1,)}}
        >>> key_mapping = {"obs": "obs", "action": "action", "reward": "reward", "done": "done", "returns": "returns", "advantages": "advantages"}
        >>> device = "cpu"
        >>> n_step = 1
        >>> buffer = BaseBuffer(buffer_size, env_dict, key_mapping, device, n_step)
    """

    def __init__(self, buffer_size: int, env_dict: dict[str, Any], key_mapping: dict[str, str] | None, device: str, n_envs: int = 1) -> None:
        """
        Initialize the replay buffer with flexible env_dict settings.

        Args:
            buffer_size (int): Size of the replay buffer.
            env_dict (Dict[str, Any]): Environment dictionary for replay buffer.
            key_mapping (Dict[str, str]): Key mapping for the replay buffer.
            device (str): Device to store the replay buffer.
            n_step (int): Number of steps for n-step returns.

        Example:
            >>> buffer = BaseBuffer(1000, {"obs": {"shape": (4,)}, "action": {"shape": (1,)}, "reward": {"shape": (1,)}, "done": {"shape": (1,)}}, "cpu", 1)
        """
        self.buffer = {}

        self.buffer_size = buffer_size
        self.env_dict = env_dict
        self.key_mapping = key_mapping if key_mapping is not None else {"obs": "obs", "action": "action", "reward": "reward", "done": "done", "returns": "returns", "advantages": "advantages"}
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
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v).to(self.device)

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
        return {self.key_mapping[k]: v[idx].to(self.device) for k, v in self.buffer.items()}

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
        key_mapping (Dict[str, str]): Key mapping for the replay buffer.
        device (str): Device to store the replay buffer.
        n_step (int): Number of steps for n-step returns.

    Example:
        >>> buffer_size = 1000
        >>> env_dict = {"obs": {"shape": (4,)}, "action": {"shape": (1,)}, "reward": {"shape": (1,)}, "done": {"shape": (1,)}}
        >>> key_mapping = {"obs": "obs", "action": "action", "reward": "reward", "done": "done", "returns": "returns", "advantages": "advantages"}
        >>> device = "cpu"
        >>> n_step = 1
        >>> buffer = RolloutBuffer(buffer_size, env_dict, key_mapping, device, n_step)
    """

    def __init__(self, buffer_size: int, env_dict: dict[str, Any], key_mapping: dict[str, str] | None, device: str, n_envs: int = 1) -> None:
        """
        Initialize the replay buffer with flexible env_dict settings.

        Args:
            buffer_size (int): Size of the replay buffer.
            env_dict (Dict[str, Any]): Environment dictionary for replay buffer.
            key_mapping (Dict[str, str]): Key mapping for the replay buffer.
            device (str): Device to store the replay buffer.
            n_step (int): Number of steps for n-step returns.

        Example:
            >>> buffer = RolloutBuffer(1000, {"obs": {"shape": (4,)}, "action": {"shape": (1,)}, "reward": {"shape": (1,)}, "done": {"shape": (1,)}}, "cpu", 1)
        """
        super().__init__(buffer_size, env_dict, key_mapping, device, n_envs)

    def compute_returns_and_advantages_gae(self, last_state: torch.Tensor, gamma: float, lmbd: float, critic: Distribution) -> dict[str, torch.Tensor]:
        """Compute returns and advantages for the stored trajectories.

        Args:
            last_state (torch.Tensor): Last state of the trajectory.
            gamma (float): Discount factor.
            lmbd (float): Lambda factor for GAE.
            critic (Distribution): Critic distribution.

        Returns:
            dict[str, torch.Tensor]: Returns and advantages.

        Example:
            >>> returns_advantages = buffer.compute_returns_and_advantages_gae(last_state, gamma, lmbd, critic)
        """
        with torch.no_grad():
            last_value = critic.sample({"o": last_state})["v"]
            last_gae = 0
            last_return = last_value
            returns = torch.empty(self.buffer_size, self.n_envs, device=self.device)
            advantages = torch.empty(self.buffer_size, self.n_envs, device=self.device)

            for i in reversed(range(self.buffer_size - 1)):
                mask = 1 - self.buffer["done"][i]
                cond_param = {self.key_mapping[k]: v[i + 1] for k, v in self.buffer.items()}
                next_value = critic.sample(cond_param)["v"]
                next_value = critic.sample({critic.cond_var[0]: self.buffer["obs"][i + 1]})["v"]
                delta = self.buffer["reward"][i] + gamma * next_value * mask - critic.sample({critic.cond_var[0]: self.buffer["obs"][i]})["v"]
                last_gae = delta + gamma * lmbd * mask * last_gae
                last_return = self.buffer["reward"][i] + gamma * last_return * mask
                returns[i] = last_return
                advantages[i] = last_gae

        self.buffer |= {"returns": returns, "advantages": advantages}
        return {"returns": returns, "advantages": advantages}

    def compute_returns_and_advantages_mc(self, gamma: float, critic: Distribution) -> dict[str, torch.Tensor]:
        """Compute returns and advantages for the stored trajectories.

        Args:
            gamma (float): Discount factor.

        Returns:
            dict[str, torch.Tensor]: Returns and advantages.

        Example:
            >>> returns_advantages = buffer.compute_returns_and_advantages_mc(gamma)
        """
        returns = torch.zeros(self.buffer_size, self.n_envs, device=self.device)
        discounted_return = torch.zeros(self.n_envs, device=self.device)
        for i in reversed(range(self.buffer_size)):
            discounted_return = self.buffer["reward"][i] + gamma * discounted_return * (1 - self.buffer["done"][i])
            returns[i] = discounted_return
        advantages = returns - critic.sample({"o": self.buffer["obs"]})["v"]

        self.buffer |= {"returns": returns, "advantages": advantages}
        return {"returns": returns, "advantages": advantages}

    def compute_returns_and_advantages_n_step(self, gamma: float, n_step: int, critic: Distribution) -> dict[str, torch.Tensor]:
        """Compute returns and advantages for the stored trajectories.

        Args:
            gamma (float): Discount factor.
            n_step (int): Number of steps for n-step returns.

        Returns:
            dict[str, torch.Tensor]: Returns and advantages.

        Example:
            >>> returns_advantages = buffer.compute_returns_and_advantages_n_step(gamma, n_step)
        """
        returns = torch.zeros(self.buffer_size, self.n_envs, device=self.device)
        for i in reversed(range(self.buffer_size)):
            discounted_return = torch.zeros(self.n_envs, device=self.device)
            for step in range(n_step):
                idx = min(i + step, self.buffer_size - 1)
                discounted_return += (gamma**step) * self.buffer["reward"][idx].squeeze(-1)
                if self.buffer["done"][idx]:
                    break
            next_idx = min(i + n_step, self.buffer_size - 1)
            if not self.buffer["done"][next_idx]:
                discounted_return += (gamma**n_step) * critic.sample({"o": self.buffer["obs"][next_idx]})["v"].squeeze(-1)
            returns[i] = discounted_return
        advantages = returns - critic.sample({"o": self.buffer["obs"]})["v"]

        self.buffer |= {"returns": returns, "advantages": advantages}
        return {"returns": returns, "advantages": advantages}
