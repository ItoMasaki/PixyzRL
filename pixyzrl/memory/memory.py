"""The implementation of the experience replay memory and rollout buffer."""

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class ExperienceReplay(Dataset):
    """Experience replay memory."""

    def __init__(self, batch_size: int, batch_length: int, length_num: int) -> None:
        """Initialize the experience replay memory."""
        self.batch_size = batch_size
        self.batch_length = batch_length
        self.length_num = length_num

        self.reset()

        self.idx = 0
        self._reset = True

    def reset(self) -> None:
        """Reset the memory."""
        self.observations: list[NDArray[np.float64]] = []
        self.actions: list[NDArray[np.float64]] = []
        self.rewards: list[NDArray[np.float64]] = []
        self.nonterminated: list[NDArray[np.float64]] = []

        self.idx = 0
        self.counter = 0

    def add(self, observation: NDArray[np.float64], action: NDArray[np.float64], reward: float, nonterminated: bool) -> None:
        """Add a new experience to the memory.

        Example:
        -------
        >>> import numpy as np
        >>> memory = ExperienceReplay(32, 16)
        >>> observation = np.random.rand(96, 96, 3)
        >>> action = np.random.rand(3)
        >>> reward = 0.0
        >>> nonterminated = True
        >>> memory.add(observation, action, reward, nonterminated)

        """
        if not nonterminated:
            np.save(
                f"data/{self.idx}.npy",
                {
                    "observations": np.array(self.observations),
                    "actions": np.array(self.actions),
                    "rewards": np.array(self.rewards),
                    "nonterminated": np.array(self.nonterminated),
                },
            )

            self.observations = []
            self.actions = []
            self.rewards = []
            self.nonterminated = []

            self.idx += 1
            self._reset = False
        else:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append([reward])
            self.nonterminated.append([nonterminated])

    def prepare(self) -> None:
        observations: list[NDArray[np.float64]] = [np.ndarray((0, 96, 96, 3), dtype=np.int64) for _ in range(self.batch_size)]
        actions: list[NDArray[np.float64]] = [np.ndarray((0, 3)) for _ in range(self.batch_size)]
        rewards: list[NDArray[np.float64]] = [np.ndarray((0, 1)) for _ in range(self.batch_size)]
        nonterminated: list[NDArray[np.float64]] = [np.ndarray((0, 1)) for _ in range(self.batch_size)]

        for i in range(self.batch_size):
            while observations[i].shape[0] < self.batch_length * self.length_num:
                idx = np.random.randint(0, self.idx)
                load_object = np.load(f"data/{idx}.npy", allow_pickle=True).item()
                _observation = load_object["observations"]
                _action = load_object["actions"]
                _reward = load_object["rewards"]
                _nonterminated = load_object["nonterminated"]

                observations[i] = np.concatenate([observations[i], _observation])
                actions[i] = np.concatenate([actions[i], _action])
                rewards[i] = np.concatenate([rewards[i], _reward])
                nonterminated[i] = np.concatenate([nonterminated[i], _nonterminated])

            observations[i] = self.split_by_chunk(observations[i], self.batch_length)[: self.length_num]
            actions[i] = self.split_by_chunk(actions[i], self.batch_length)[: self.length_num]
            rewards[i] = self.split_by_chunk(rewards[i], self.batch_length)[: self.length_num]
            nonterminated[i] = self.split_by_chunk(nonterminated[i], self.batch_length)[: self.length_num]

        self.sample_observations = observations
        self.sample_actions = actions
        self.sample_rewards = rewards
        self.sample_nonterminated = nonterminated

        self.max_length = len(self.sample_observations[0])

    def __len__(self) -> int:
        if not self._reset:
            self.prepare()
            self._reset = True

        """Return the number of experiences in the memory."""
        if self.counter < len(self.sample_observations[0]):
            self.counter += 1
        else:
            self.counter = 1

        return len(self.sample_observations)

    def __getitem__(self, idx: int) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Sample a batch of experiences from the memory.

        Example:
        -------
        >>> import numpy as np
        >>> memory = ExperienceReplay(32, 16)
        >>> observation = np.random.rand(96, 96, 3)
        >>> action = np.random.rand(3)
        >>> reward = 0.0
        >>> nonterminated = False
        >>> observations, actions, rewards, nonterminated = memory.sample(1)

        """
        return (
            np.array(self.sample_observations[idx][self.counter - 1]).transpose(0, 3, 1, 2),
            np.array(self.sample_actions[idx][self.counter - 1]),
            np.array(self.sample_rewards[idx][self.counter - 1]),
            np.array(self.sample_nonterminated[idx][self.counter - 1]),
        )

    def split_by_chunk(self, data: NDArray[np.float64], chunk_size: int) -> list[NDArray[np.float64]]:
        """Split the data by chunk size.

        Example:
        -------
        >>> import numpy as np
        >>> memory = ExperienceReplay(32, 16)
        >>> data = np.random.rand(64, 96, 96, 3)
        >>> chunk_size = 16
        >>> splited_data = memory.split_by_chunk(data, chunk_size)
        >>> len(splited_data)
        4

        """
        return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.beliefs = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.beliefs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def compute_returns_and_advantage(self, last_values, dones):
        last_values = last_values.cpu().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.to(torch.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values
