import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer


class Memory:
    """Base class for memory buffer."""

    def __init__(self, obs_shape, action_shape, buffer_size, num_envs=1, device="cpu", key_mapping=None):
        self.device = device
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.storage = LazyTensorStorage(buffer_size * num_envs)
        self.buffer = ReplayBuffer(storage=self.storage, batch_size=buffer_size * num_envs)
        self.key_mapping = key_mapping if key_mapping else {"obs": "s", "action": "a", "reward": "r", "done": "d", "log_prob": "log_p", "value": "v"}

    def add(self, obs, action, reward, done, log_prob=None, value=None):
        """Add a new experience to the buffer."""
        data = TensorDict(
            {
                self.key_mapping["obs"]: torch.tensor(obs, dtype=torch.float32, device=self.device),
                self.key_mapping["action"]: torch.tensor(action, dtype=torch.float32, device=self.device),
                self.key_mapping["reward"]: torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(-1),
                self.key_mapping["done"]: torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(-1),
            },
            batch_size=self.num_envs,
        )

        if log_prob is not None:
            data[self.key_mapping["log_prob"]] = torch.tensor(log_prob, dtype=torch.float32, device=self.device)
        if value is not None:
            data[self.key_mapping["value"]] = torch.tensor(value, dtype=torch.float32, device=self.device)

        self.buffer.add(data)

    def sample(self):
        """Sample a batch of experiences."""
        return self.buffer.sample().to(self.device)

    def clear(self):
        """Clear the buffer."""
        self.buffer = ReplayBuffer(storage=LazyTensorStorage(self.buffer_size * self.num_envs), batch_size=self.buffer_size * self.num_envs)

    def __len__(self):
        return len(self.buffer)


class ExperienceMemory(Memory):
    """Experience replay buffer for off-policy learning."""

    pass  # 現在のMemoryがそのまま利用可能


class RolloutBuffer(Memory):
    """Rollout buffer for on-policy algorithms like PPO."""

    def __init__(self, obs_shape, action_shape, buffer_size, num_envs=1, device="cpu", key_mapping=None):
        super().__init__(obs_shape, action_shape, buffer_size, num_envs, device, key_mapping)

    def compute_returns(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute advantage and returns using GAE (Generalized Advantage Estimation)."""
        advantages = torch.zeros_like(self.buffer.storage[:][self.key_mapping["reward"]], device=self.device)
        last_advantage = 0

        for t in reversed(range(len(self.buffer.storage[:][self.key_mapping["reward"]]))):
            mask = 1.0 - self.buffer.storage[:][self.key_mapping["done"]][t]
            delta = self.buffer.storage[:][self.key_mapping["reward"]][t] + gamma * last_value * mask - self.buffer.storage[:][self.key_mapping["value"]][t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * mask * last_advantage
            last_value = self.buffer.storage[:][self.key_mapping["value"]][t]

        returns = advantages + self.buffer.storage[:][self.key_mapping["value"]]
        self.buffer.storage[:]["advantage"] = advantages
        self.buffer.storage[:]["return"] = returns


class VectorizedMemory(RolloutBuffer):
    """Vectorized version of RolloutBuffer for parallel environments."""

    def __init__(self, obs_shape, action_shape, buffer_size, num_envs, device="cpu", key_mapping=None):
        super().__init__(obs_shape, action_shape, buffer_size, num_envs, device, key_mapping)

    def add(self, data: TensorDict):
        """Override add method to handle vectorized environments."""
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
        self.buffer.add(data)
