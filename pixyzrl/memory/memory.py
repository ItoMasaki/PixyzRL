from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pixyz.distributions import Distribution


class BaseBuffer:
    """Base class for replay buffers."""

    def __init__(self, buffer_size: int, env_dict: dict[str, Any], device: str, n_envs: int = 1) -> None:
        """
        Initialize the replay buffer with flexible env_dict settings.

        Args:
            buffer_size (int): Size of the replay buffer.
            env_dict (Dict[str, Any]): Environment dictionary for replay buffer.
            device (str): Device to store the replay buffer.
            n_step (int): Number of steps for n-step returns.

        Example:
            >>> buffer = BaseBuffer(1000, {
            ...     "obs": {"shape": (4,), "map": "o"},
            ...     "action": {"shape": (1,), "map": "a"},
            ...     "reward": {"shape": (1,), "map": "r"},
            ...     "done": {"shape": (1,), "map": "d"}
            ... }, "cpu", 1)
        """
        self.buffer = {}

        self.buffer_size = buffer_size
        self.env_dict = env_dict
        self.key_mapping = {k: v.get("map", k) for k, v in env_dict.items()}
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
            >>> buffer = BaseBuffer(1000, {
            ...     "obs": {"shape": (4,), "map": "o"},
            ...     "action": {"shape": (1,), "map": "a"},
            ...     "reward": {"shape": (1,), "map": "r"},
            ...     "done": {"shape": (1,), "map": "d"}
            ... }, "cpu", 1)
            >>> obs = np.random.rand(4)
            >>> action = np.random.rand(1)
            >>> reward = np.random.rand(1)
            >>> done = np.random.rand(1)
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
            >>> buffer = BaseBuffer(1000, {
            ...     "obs": {"shape": (4,), "map": "o"},
            ...     "action": {"shape": (1,), "map": "a"},
            ...     "reward": {"shape": (1,), "map": "r"},
            ...     "done": {"shape": (1,), "map": "d"}
            ... }, "cpu", 1)
            >>>
            >>> buffer.add(obs=np.random.rand(4), action=np.random.rand(1), reward=np.random.rand(1), done=np.random.rand(1))
            >>> buffer.add(obs=np.random.rand(4), action=np.random.rand(1), reward=np.random.rand(1), done=np.random.rand(1))
            >>>
            >>> batch = buffer.sample(1)
            >>> len(batch["o"])
            1
        """
        idx = torch.randint(0, self.pos - 1, (batch_size,))
        return {self.key_mapping[k]: v[idx].to(self.device).detach().squeeze() for k, v in self.buffer.items()}

    def clear(self) -> None:
        """Clear the buffer.

        Example
            >>> buffer = BaseBuffer(1000, {
            ...     "obs": {"shape": (4,), "map": "o"},
            ...     "action": {"shape": (1,), "map": "a"},
            ...     "reward": {"shape": (1,), "map": "r"},
            ...     "done": {"shape": (1,), "map": "d"}
            ... }, "cpu", 1)
            >>>
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
            >>> buffer = BaseBuffer(1000, {
            ...     "obs": {"shape": (4,), "map": "o"},
            ...     "action": {"shape": (1,), "map": "a"},
            ...     "reward": {"shape": (1,), "map": "r"},
            ...     "done": {"shape": (1,), "map": "d"}
            ... }, "cpu", 1)
            >>>
            >>> buffer.add(obs=np.random.rand(4), action=np.random.rand(1), reward=np.random.rand(1), done=np.random.rand(1))
            >>> buffer.add(obs=np.random.rand(4), action=np.random.rand(1), reward=np.random.rand(1), done=np.random.rand(1))
            >>>
            >>> len(buffer)
            2
        """
        return self.pos

    @abstractmethod
    def compute_returns_and_advantages_gae(self, last_value: torch.Tensor, gamma: float, lam: float) -> dict[str, torch.Tensor]:
        """Compute returns and advantages for the stored trajectories.

        Args:
            last_state (torch.Tensor): Last state of the trajectory.
            gamma (float): Discount factor.
            lam (float): Lambda factor for GAE.
            critic (Distribution): Critic distribution.

        Returns:
            dict[str, torch.Tensor]: Returns and advantages.

        Example:
            >>> buffer = BaseBuffer(1000, {
            ...     "obs": {"shape": (4,), "map": "o"},
            ...     "action": {"shape": (1,), "map": "a"},
            ...     "reward": {"shape": (1,), "map": "r"},
            ...     "done": {"shape": (1,), "map": "d"}
            ... }, "cpu", 1)
            >>>
            >>> buffer.add(obs=np.zeros(4), action=np.zeros(1), reward=np.ones(1), done=np.zeros(1))
            >>> last_value = torch.zeros(1)
            >>> buffer.compute_returns_and_advantages_gae(last_value, 0.99, 0.95)
        """
        ...

    @abstractmethod
    def compute_returns_and_advantages_mc(self, gamma: float) -> dict[str, torch.Tensor]:
        """Compute returns and advantages for the stored trajectories.

        Args:
            gamma (float): Discount factor.

        Returns:
            dict[str, torch.Tensor]: Returns and advantages.

        Example:
            >>> buffer = BaseBuffer(1000, {
            ...     "obs": {"shape": (4,), "map": "o"},
            ...     "action": {"shape": (1,), "map": "a"},
            ...     "reward": {"shape": (1,), "map": "r"},
            ...     "done": {"shape": (1,), "map": "d"}
            ... }, "cpu", 1)
            >>>
            >>> buffer.add(obs=np.zeros(4), action=np.zeros(1), reward=np.ones(1), done=np.zeros(1))
            >>> last_value = torch.zeros(1)
            >>>
            >>> buffer.compute_returns_and_advantages_mc(0.99)
        """
        ...

    @abstractmethod
    def compute_returns_and_advantages_n_step(self, gamma: float, n_step: int) -> dict[str, torch.Tensor]:
        """Compute returns and advantages for the stored trajectories.

        Args:
            gamma (float): Discount factor.
            n_step (int): Number of steps for n-step returns.

        Returns:
            dict[str, torch.Tensor]: Returns and advantages.

        Example:
            >>> buffer = BaseBuffer(1000, {
            ...     "obs": {"shape": (4,), "map": "o"},
            ...     "action": {"shape": (1,), "map": "a"},
            ...     "reward": {"shape": (1,), "map": "r"},
            ...     "done": {"shape": (1,), "map": "d"}
            ... }, "cpu", 1)
            >>>
            >>> buffer.add(obs=np.zeros(4), action=np.zeros(1), reward=np.ones(1), done=np.zeros(1))
            >>> last_value = torch.zeros(1)
            >>>
            >>> returns_advantages = buffer.compute_returns_and_advantages_n_step(0.99, 2)
        """
        ...


class RolloutBuffer(BaseBuffer):
    """Rollout buffer for storing trajectories."""

    def __init__(self, buffer_size: int, env_dict: dict[str, Any], device: str, n_envs: int = 1) -> None:
        """
        Initialize the replay buffer with flexible env_dict settings.

        Args:
            buffer_size (int): Size of the replay buffer.
            env_dict (Dict[str, Any]): Environment dictionary for replay buffer.
            device (str): Device to store the replay buffer.
            n_step (int): Number of steps for n-step returns.

        Example:
            >>> buffer = RolloutBuffer(1000, {
            ...     "obs": {"shape": (4,), "map": "o"},
            ...     "action": {"shape": (1,), "map": "a"},
            ...     "reward": {"shape": (1,), "map": "r"},
            ...     "done": {"shape": (1,), "map": "d"}
            ... }, "cpu", 1)
        """
        super().__init__(buffer_size, env_dict, device, n_envs)

    def compute_returns_and_advantages_gae(self, last_value: torch.Tensor, gamma: float, lam: float) -> dict[str, torch.Tensor]:
        """Compute returns and advantages for the stored trajectories.

        Args:
            last_state (torch.Tensor): Last state of the trajectory.
            gamma (float): Discount factor.
            lmbd (float): Lambda factor for GAE.
            critic (Distribution): Critic distribution.

        Returns:
            dict[str, torch.Tensor]: Returns and advantages.

        Example:
            >>> buffer = RolloutBuffer(3, {
            ...     "obs": {"shape": (4,), "map": "o"},
            ...     "action": {"shape": (1,), "map": "a"},
            ...     "reward": {"shape": (1,), "map": "r"},
            ...     "done": {"shape": (1,), "map": "d"},
            ...     "value": {"shape": (1,), "map": "v"}
            ... }, "cpu", 1)
            >>>
            >>> buffer.add(obs=np.zeros(4), action=np.zeros(1), reward=np.ones(1), done=np.zeros(1), value=np.zeros(1))
            >>> buffer.add(obs=np.zeros(4), action=np.zeros(1), reward=np.ones(1), done=np.zeros(1), value=np.zeros(1))
            >>> buffer.add(obs=np.zeros(4), action=np.zeros(1), reward=np.ones(1), done=np.zeros(1), value=np.zeros(1))
            >>>
            >>> last_value = torch.zeros(1)
            >>> return_advantage = buffer.compute_returns_and_advantages_gae(last_value, 0.99, 0.95)
            >>> "returns" in return_advantage and "advantages" in return_advantage
            True
        """
        with torch.no_grad():
            advantages = torch.zeros_like(self.buffer["reward"])
            gae = 0.0

            for i in reversed(range(self.buffer_size)):
                if i == self.buffer_size - 1:
                    # バッチ末端の next_value は last_value（実装により異なる）
                    next_value = last_value
                    next_nonterminal = 1.0 - self.buffer["done"][i]
                else:
                    next_value = self.buffer["value"][i + 1]
                    next_nonterminal = 1.0 - self.buffer["done"][i + 1]

                delta = self.buffer["reward"][i] + gamma * next_value * next_nonterminal - self.buffer["value"][i]
                gae = delta + gamma * lam * next_nonterminal * gae
                advantages[i] = gae

            # 最終的にReturnを作る場合は以下
            returns = advantages + self.buffer["value"]

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.buffer |= {"returns": returns.detach(), "advantages": advantages.detach()}
        return {"returns": returns, "advantages": advantages}

    def compute_returns_and_advantages_mc(self, gamma: float) -> dict[str, torch.Tensor]:
        """Compute returns and advantages for the stored trajectories.

        Args:
            gamma (float): Discount factor.

        Returns:
            dict[str, torch.Tensor]: Returns and advantages.

        Example:
            >>> buffer = RolloutBuffer(3, {
            ...     "obs": {"shape": (4,), "map": "o"},
            ...     "action": {"shape": (1,), "map": "a"},
            ...     "reward": {"shape": (1,), "map": "r"},
            ...     "done": {"shape": (1,), "map": "d"},
            ...     "value": {"shape": (1,), "map": "v"}
            ... }, "cpu", 1)
            >>>
            >>> buffer.add(obs=np.zeros(4), action=np.zeros(1), reward=np.ones(1), done=np.zeros(1), value=np.zeros(1))
            >>> buffer.add(obs=np.zeros(4), action=np.zeros(1), reward=np.ones(1), done=np.zeros(1), value=np.zeros(1))
            >>> buffer.add(obs=np.zeros(4), action=np.zeros(1), reward=np.ones(1), done=np.zeros(1), value=np.zeros(1))
            >>>
            >>> last_value = torch.zeros(1)
            >>> return_advantage = buffer.compute_returns_and_advantages_mc(0.99)
            >>> "returns" in return_advantage and "advantages" in return_advantage
            True
        """
        returns = torch.zeros(self.buffer_size, self.n_envs, device=self.device)
        discounted_return = torch.zeros(self.n_envs, device=self.device)
        for i in reversed(range(self.buffer_size)):
            discounted_return = self.buffer["reward"][i] + gamma * discounted_return * (1 - self.buffer["done"][i])
            returns[i] = discounted_return

        advantages = returns - self.buffer["value"]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.buffer |= {"returns": returns.detach(), "advantages": advantages.detach()}
        return {"returns": returns, "advantages": advantages}

    def compute_returns_and_advantages_n_step(self, gamma: float, n_step: int) -> dict[str, torch.Tensor]:
        """Compute returns and advantages for the stored trajectories.

        Args:
            gamma (float): Discount factor.
            n_step (int): Number of steps for n-step returns.

        Returns:
            dict[str, torch.Tensor]: Returns and advantages.

        Example:
            >>> buffer = RolloutBuffer(3, {
            ...     "obs": {"shape": (4,), "map": "o"},
            ...     "action": {"shape": (1,), "map": "a"},
            ...     "reward": {"shape": (1,), "map": "r"},
            ...     "done": {"shape": (1,), "map": "d"},
            ...     "value": {"shape": (1,), "map": "v"}
            ... }, "cpu", 1)
            >>>
            >>> buffer.add(obs=np.zeros(4), action=np.zeros(1), reward=np.ones(1), done=np.zeros(1), value=np.zeros(1))
            >>> buffer.add(obs=np.zeros(4), action=np.zeros(1), reward=np.ones(1), done=np.zeros(1), value=np.zeros(1))
            >>> buffer.add(obs=np.zeros(4), action=np.zeros(1), reward=np.ones(1), done=np.zeros(1), value=np.zeros(1))
            >>>
            >>> last_value = torch.zeros(1)
            >>> returns_advantages = buffer.compute_returns_and_advantages_n_step(0.99, 2)
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
                discounted_return += (gamma**n_step) * self.buffer["value"][next_idx].squeeze(-1)
            returns[i] = discounted_return
        advantages = returns - self.buffer["value"]

        self.buffer |= {"returns": returns, "advantages": advantages}
        return {"returns": returns, "advantages": advantages}


class ExperienceReplay(BaseBuffer):
    """Standard Experience Replay Buffer for DQN."""

    def __init__(self, state_shape: tuple, action_shape: tuple, buffer_size: int = 10000, batch_size: int = 64, device: str = "cpu"):
        """Initialize the buffer."""
        env_dict = {"states": {"shape": state_shape}, "actions": {"shape": action_shape}, "rewards": {"shape": (1,)}, "next_states": {"shape": state_shape}, "dones": {"shape": (1,), "dtype": torch.bool}}
        super().__init__(buffer_size, env_dict, None, device)
        self.batch_size = batch_size

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Add an experience to the buffer."""
        super().add(states=state, actions=action, rewards=reward, next_states=next_state, dones=done)

    def sample(self) -> dict[str, torch.Tensor]:
        """Sample a batch of experiences."""
        return super().sample(self.batch_size)

    def clear(self) -> None:
        """Clear the buffer."""
        super().clear()


class PrioritizedExperienceReplay(BaseBuffer):
    """Prioritized Experience Replay Buffer for DQN."""

    def __init__(self, state_shape: tuple, action_shape: tuple, buffer_size: int = 10000, batch_size: int = 64, device: str = "cpu", alpha: float = 0.6, beta: float = 0.4):
        """Initialize the buffer with prioritization."""
        env_dict = {"states": {"shape": state_shape}, "actions": {"shape": action_shape}, "rewards": {"shape": (1,)}, "next_states": {"shape": state_shape}, "dones": {"shape": (1,), "dtype": torch.bool}, "priorities": {"shape": (1,), "dtype": torch.float32}}
        super().__init__(buffer_size, env_dict, None, device)
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool, priority: float = 1.0) -> None:
        """Add an experience to the buffer with priority."""
        super().add(states=state, actions=action, rewards=reward, next_states=next_state, dones=done, priorities=priority**self.alpha)

    def sample(self) -> dict[str, torch.Tensor]:
        """Sample a batch of experiences with prioritization."""
        priorities = self.buffer["priorities"][: self.pos] if not self.full else self.buffer["priorities"]
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(probabilities), self.batch_size, p=probabilities.numpy(), replace=False)
        weights = (len(probabilities) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        batch = {k: v[indices] for k, v in self.buffer.items() if k != "priorities"}
        batch["weights"] = torch.tensor(weights, dtype=torch.float32, device=self.device)
        return batch

    def clear(self) -> None:
        """Clear the buffer."""
        super().clear()
