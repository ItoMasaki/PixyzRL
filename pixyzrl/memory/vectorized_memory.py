import numpy as np
import torch
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDict


class VectorizedReplayBuffer:
    """Experience replay buffer for vectorized environments using torchrl."""

    def __init__(self, obs_shape: tuple[int], action_shape: tuple[int], buffer_size: int, num_envs: int, batch_size: int, device="cpu"):
        """
        Initialize the replay buffer.

        Args:
            obs_shape (tuple): Shape of the observations.
            action_shape (tuple): Shape of the actions.
            buffer_size (int): Maximum size of the buffer.
            num_envs (int): Number of parallel environments.
            batch_size (int): Batch size for sampling.
            device (str): Device to store the tensors (cpu or cuda).
        """
        self.device = device
        self.batch_size = batch_size

        # torchrlのReplayBufferを使用
        self.buffer = ReplayBuffer(storage=LazyTensorStorage(buffer_size), batch_size=batch_size)

        # 各環境ごとのデータ用のテンソル
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.num_envs = num_envs

    def add(self, obs, action, reward, done) -> None:
        """
        Add new experiences to the buffer.

        Args:
            obs (np.array or torch.Tensor): Observations from the environments.
            action (np.array or torch.Tensor): Actions taken by the policy.
            reward (np.array or torch.Tensor): Rewards received.
            done (np.array or torch.Tensor): Whether episodes are done.
        """
        # NumPyをtorch.Tensorに変換
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        if isinstance(reward, np.ndarray):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(-1)
        if isinstance(done, np.ndarray):
            done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # `TensorDict` を作成
        data = TensorDict(
            {
                "obs": obs,
                "action": action,
                "reward": reward,
                "done": done,
            },
            batch_size=self.num_envs,
        )

        # `ReplayBuffer` に追加
        self.buffer.extend(data)

    def sample(self):
        """
        Sample a batch of experiences.

        Returns:
            TensorDict: A batch of sampled experiences.
        """
        return self.buffer.sample().to(self.device)

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
