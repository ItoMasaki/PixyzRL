"""Deep Q-Network (DQN) agent using Pixyz distributions."""

from __future__ import annotations

import random
from copy import deepcopy

import torch
from pixyz import distributions as dists
from torch.optim import Adam
from torch.utils.data import DataLoader

from pixyzrl.memory import BaseBuffer
from pixyzrl.models.base_model import RLModel


class DQN(RLModel):
    """Deep Q-Network (DQN) for discrete action spaces."""

    def __init__(
        self,
        q_network: dists.Distribution,
        lr: float = 1e-3,
        gamma: float = 0.99,
        target_update_interval: int = 100,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10_000,
        action_var: str = "a",
        obs_var: str | None = None,
        next_obs_var: str = "o_next",
        reward_var: str = "r",
        done_var: str = "d",
        device: str = "cpu",
    ) -> None:
        self._is_on_policy = False
        self._action_var = action_var

        self.q_network = q_network.to(device)
        self.target_q_network = deepcopy(q_network).to(device)

        self.obs_var = obs_var or q_network.cond_var[0]
        self.next_obs_var = next_obs_var
        self.reward_var = reward_var
        self.done_var = done_var
        self.q_var = q_network.var[0]

        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.device = device

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.global_step = 0

        dummy_loss = -self.q_network.log_prob().mean()
        super().__init__(
            dummy_loss,
            distributions=[self.q_network, self.target_q_network],
            optimizer=Adam,
            optimizer_params={},
        )

        self.optimizer = Adam(self.q_network.parameters(), lr=lr)
        self.transfer_state_dict()

    def _epsilon(self) -> float:
        if self.global_step >= self.epsilon_decay_steps:
            return self.epsilon_end
        ratio = self.global_step / self.epsilon_decay_steps
        return self.epsilon_start + ratio * (self.epsilon_end - self.epsilon_start)

    @torch.no_grad()
    def select_action(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        obs = state[self.obs_var].to(self.device)
        q_values = self.q_network.sample_mean({self.obs_var: obs})
        greedy_actions = torch.argmax(q_values, dim=-1)

        epsilon = self._epsilon()
        if random.random() < epsilon:
            random_actions = torch.randint(
                0,
                q_values.shape[-1],
                size=greedy_actions.shape,
                device=self.device,
            )
            action_idx = random_actions
        else:
            action_idx = greedy_actions

        self.global_step += 1
        action_one_hot = torch.nn.functional.one_hot(
            action_idx,
            num_classes=q_values.shape[-1],
        ).float()

        return {self.action_var: action_one_hot}

    def train_step(
        self,
        memory: BaseBuffer,
        batch_size: int = 64,
        num_epochs: int = 1,
    ) -> float:
        dataloader = DataLoader(memory, batch_size=batch_size, shuffle=True)
        total_loss = 0.0

        for _ in range(num_epochs):
            for batch in dataloader:
                obs = batch[self.obs_var].to(self.device)
                next_obs = batch[self.next_obs_var].to(self.device)
                rewards = batch[self.reward_var].to(self.device)
                dones = batch[self.done_var].to(self.device).float()
                actions = batch[self.action_var].to(self.device)

                if actions.ndim > 1 and actions.shape[-1] > 1:
                    action_indices = actions.argmax(dim=-1, keepdim=True).long()
                else:
                    action_indices = actions.long().view(-1, 1)

                q_values = self.q_network.sample_mean({self.obs_var: obs})
                chosen_q = q_values.gather(1, action_indices)

                with torch.no_grad():
                    target_q_values = self.target_q_network.sample_mean(
                        {self.obs_var: next_obs}
                    )
                    next_q = target_q_values.max(dim=-1, keepdim=True).values
                    target = rewards + (1.0 - dones) * self.gamma * next_q

                loss = torch.nn.functional.mse_loss(chosen_q, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.global_step += 1
                if self.global_step % self.target_update_interval == 0:
                    self.transfer_state_dict()

                total_loss += float(loss.detach())

        return total_loss / max(1, len(dataloader) * num_epochs)

    @torch.no_grad()
    def transfer_state_dict(self) -> None:
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_q_network": self.target_q_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = int(checkpoint.get("global_step", 0))
