"""Soft Actor-Critic (SAC) agent using Pixyz distributions."""

from __future__ import annotations

from copy import deepcopy

import torch
from pixyz import distributions as dists
from torch.optim import Adam
from torch.utils.data import DataLoader

from pixyzrl.memory import BaseBuffer
from pixyzrl.models.base_model import RLModel


class SAC(RLModel):
    """Soft Actor-Critic (SAC) for continuous control."""

    def __init__(
        self,
        actor: dists.Distribution,
        critic1: dists.Distribution,
        critic2: dists.Distribution,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy: float | None = None,
        action_var: str = "a",
        obs_var: str | None = None,
        next_obs_var: str = "o_next",
        reward_var: str = "r",
        done_var: str = "d",
        device: str = "cpu",
    ) -> None:
        self._is_on_policy = False
        self._action_var = action_var

        self.actor = actor.to(device)
        self.critic1 = critic1.to(device)
        self.critic2 = critic2.to(device)
        self.target_critic1 = deepcopy(critic1).to(device)
        self.target_critic2 = deepcopy(critic2).to(device)

        self.obs_var = obs_var or actor.cond_var[0]
        self.next_obs_var = next_obs_var
        self.reward_var = reward_var
        self.done_var = done_var

        self.gamma = gamma
        self.tau = tau
        self.lr_alpha = lr_alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        self.target_entropy = target_entropy
        self.device = device

        if self.target_entropy is None:
            self.target_entropy = -1.0

        dummy_loss = -self.actor.log_prob().mean()

        super().__init__(
            dummy_loss,
            distributions=[
                self.actor,
                self.critic1,
                self.critic2,
                self.target_critic1,
                self.target_critic2,
            ],
            optimizer=Adam,
            optimizer_params={},
        )

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(
            [
                {"params": self.critic1.parameters(), "lr": lr_critic},
                {"params": self.critic2.parameters(), "lr": lr_critic},
            ]
        )

        if self.auto_entropy_tuning:
            self.log_alpha = torch.nn.Parameter(
                torch.log(torch.tensor([float(alpha)], device=device))
            )
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_alpha)
        else:
            self.log_alpha = torch.log(torch.tensor([float(alpha)], device=device))
            self.alpha_optimizer = None

        self.transfer_state_dict()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        actor_input = {self.obs_var: state[self.obs_var].to(self.device)}
        return self.actor.sample(actor_input)

    def _critic_value(
        self,
        critic: dists.Distribution,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        q = critic.sample({critic.cond_var[0]: obs, critic.cond_var[1]: action})[
            critic.var[0]
        ]
        return q

    def train_step(
        self,
        memory: BaseBuffer,
        batch_size: int = 256,
        num_epochs: int = 1,
    ) -> float:
        dataloader = DataLoader(memory, batch_size=batch_size, shuffle=True)
        total_loss = 0.0

        for _ in range(num_epochs):
            for batch in dataloader:
                obs = batch[self.obs_var].to(self.device)
                actions = batch[self.action_var].to(self.device)
                rewards = batch[self.reward_var].to(self.device)
                dones = batch[self.done_var].to(self.device).float()
                next_obs = batch[self.next_obs_var].to(self.device)

                with torch.no_grad():
                    next_action = self.actor.sample({self.obs_var: next_obs})[
                        self.action_var
                    ]
                    next_log_prob = (
                        self.actor.log_prob()
                        .eval({self.obs_var: next_obs, self.action_var: next_action})
                        .sum(dim=-1, keepdim=True)
                    )
                    target_q1 = self._critic_value(
                        self.target_critic1,
                        next_obs,
                        next_action,
                    )
                    target_q2 = self._critic_value(
                        self.target_critic2,
                        next_obs,
                        next_action,
                    )
                    target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
                    target = rewards + (1.0 - dones) * self.gamma * target_q

                current_q1 = self._critic_value(self.critic1, obs, actions)
                current_q2 = self._critic_value(self.critic2, obs, actions)
                critic_loss = torch.nn.functional.mse_loss(
                    current_q1,
                    target,
                ) + torch.nn.functional.mse_loss(current_q2, target)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                sampled_action = self.actor.sample({self.obs_var: obs})[self.action_var]
                log_prob = (
                    self.actor.log_prob()
                    .eval({self.obs_var: obs, self.action_var: sampled_action})
                    .sum(dim=-1, keepdim=True)
                )
                q1_pi = self._critic_value(self.critic1, obs, sampled_action)
                q2_pi = self._critic_value(self.critic2, obs, sampled_action)
                min_q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                alpha_loss = torch.tensor(0.0, device=self.device)
                if self.auto_entropy_tuning and self.alpha_optimizer is not None:
                    alpha_loss = -(
                        self.log_alpha * (log_prob + self.target_entropy).detach()
                    ).mean()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()

                self._soft_update_targets()
                total_loss += float((critic_loss + actor_loss + alpha_loss).detach())

        return total_loss / max(1, len(dataloader) * num_epochs)

    @torch.no_grad()
    def _soft_update_targets(self) -> None:
        for target_param, param in zip(
            self.target_critic1.parameters(), self.critic1.parameters(), strict=False
        ):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

        for target_param, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters(), strict=False
        ):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

    @torch.no_grad()
    def transfer_state_dict(self) -> None:
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target_critic1": self.target_critic1.state_dict(),
                "target_critic2": self.target_critic2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "alpha_optimizer": self.alpha_optimizer.state_dict()
                if self.alpha_optimizer is not None
                else None,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        if self.auto_entropy_tuning:
            self.log_alpha = torch.nn.Parameter(checkpoint["log_alpha"].to(self.device))
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_alpha)
            if checkpoint["alpha_optimizer"] is not None:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        else:
            self.log_alpha = checkpoint["log_alpha"].to(self.device)
