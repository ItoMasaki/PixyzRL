"""Proximal Policy Optimization (PPO) agent using Pixyz."""

from copy import deepcopy

import torch
from pixyz import distributions as dists
from pixyz.losses import Entropy, MinLoss, Parameter
from pixyz.losses import Expectation as E  # noqa: N817
from pixyz.models import Model
from torch.optim import Adam

from pixyzrl.losses import ClipLoss, MSELoss, RatioLoss


class PPO(Model):
    """PPO agent using Pixyz."""

    def __init__(
        self,
        actor: dists.Distribution,
        critic: dists.Distribution,
        shared_cnn: dists.Distribution | None,
        gamma: float,
        eps_clip: float,
        k_epochs: int,
        lr_actor: float,
        lr_critic: float,
        device: str,
    ) -> None:
        """Initialize the PPO agent."""
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = k_epochs
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.device = device

        # Shared CNN layers (optional)
        self.shared_cnn = shared_cnn

        # Actor network
        self.actor = actor
        self.actor_old = deepcopy(actor)
        self.actor_old.name = "old"

        # Critic network
        self.critic = critic

        advantage = Parameter("A")
        ratio = RatioLoss(self.actor, self.actor_old)
        clip = ClipLoss(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
        ppo_loss = -MinLoss(clip * advantage, ratio * advantage)

        mse_loss = MSELoss(self.critic, "r")

        # Loss calculation supports both A2C (shared_cnn) and TRPO (independent actor/critic)
        if self.shared_cnn is not None:
            loss = E(self.shared_cnn, ppo_loss + 0.5 * mse_loss - 0.01 * Entropy(self.actor)).mean()
        else:
            loss = (ppo_loss + 0.5 * mse_loss - 0.01 * Entropy(self.actor)).mean()

        super().__init__(loss, distributions=[self.actor, self.critic] + ([self.shared_cnn] if self.shared_cnn else []), optimizer=Adam, optimizer_params={})

        # Optimizer
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": self.lr_actor},
                {"params": self.critic.parameters(), "lr": self.lr_critic},
            ],
        )

    def select_action(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Select an action."""
        with torch.no_grad():
            if self.shared_cnn is not None:
                state = self.shared_cnn.sample({"o": state.to(self.device)})
            return self.actor_old.sample({"s": state}) | self.critic.sample({"s": state})
