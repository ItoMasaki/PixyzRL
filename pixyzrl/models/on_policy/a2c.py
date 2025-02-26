"""Advantage Actor-Critic (A2C) agent using Pixyz."""

import torch
from pixyz import distributions as dists
from pixyz.losses import Entropy, Parameter
from pixyz.losses import Expectation as E  # noqa: N817
from pixyz.models import Model
from torch.optim import Adam

from pixyzrl.losses import MSELoss


class A2C(Model):
    """A2C agent using Pixyz."""

    def __init__(
        self,
        actor: dists.Distribution,
        critic: dists.Distribution,
        lr_actor: float,
        lr_critic: float,
        device: str,
        mse_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ) -> None:
        """Initialize the A2C agent."""
        self.mse_coef = mse_coef
        self.entropy_coef = entropy_coef
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.device = device
        self.on_policy = True

        # Actor network
        self.actor = actor

        # Critic network
        self.critic = critic

        advantage = Parameter("A")
        actor_loss = -(E(self.actor, advantage)).mean()
        mse_loss = MSELoss(self.critic, "r")

        loss = actor_loss + self.mse_coef * mse_loss.mean() - self.entropy_coef * Entropy(self.actor).mean()

        super().__init__(loss, distributions=[self.actor, self.critic], optimizer=Adam, optimizer_params={})

        # Optimizer
        self.optimizer = Adam(
            [
                {"params": self.actor.parameters(), "lr": self.lr_actor},
                {"params": self.critic.parameters(), "lr": self.lr_critic},
            ],
        )

    def select_action(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Select an action."""
        with torch.no_grad():
            return self.actor.sample({self.actor.cond_var[0]: state}) | self.critic.sample({self.critic.cond_var[0]: state})
