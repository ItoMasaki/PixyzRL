"""Proximal Policy Optimization (PPO) agent using Pixyz."""

from copy import deepcopy

import torch
from pixyz import distributions as dists
from pixyz.losses import Entropy, MinLoss, Parameter
from pixyz.losses import Expectation as E  # noqa: N817
from torch.optim import Adam

from pixyzrl.losses import ClipLoss, MSELoss, RatioLoss
from pixyzrl.models.base_model import RLModel


class PPO(RLModel):
    """PPO agent using Pixyz."""

    def __init__(
        self,
        actor: dists.Distribution,
        critic: dists.Distribution,
        shared_net: dists.Distribution | None = None,
        eps_clip: float = 0.2,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        device: str = "cpu",
        mse_coef: float = 0.5,
        entropy_coef: float = 0.01,
        action_var: str = "a",
    ) -> None:
        """Initialize the PPO agent."""
        self.mse_coef = mse_coef
        self.entropy_coef = entropy_coef
        self.eps_clip = eps_clip
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.device = device
        self._is_on_policy = True
        self._action_var = action_var

        # Shared CNN layers (optional)
        self.shared_net = shared_net

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

        if self.shared_net is not None:  # A2C
            loss = E(self.shared_net, ppo_loss + self.mse_coef * mse_loss - self.entropy_coef * Entropy(self.actor)).mean()
        else:  # TRPO
            loss = (ppo_loss + self.mse_coef * mse_loss - self.entropy_coef * Entropy(self.actor)).mean()

        super().__init__(loss, distributions=[self.actor, self.critic] + ([self.shared_net] if self.shared_net else []), optimizer=Adam, optimizer_params={})

        # Optimizer
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": self.lr_actor},
                {"params": self.critic.parameters(), "lr": self.lr_critic},
            ],
        )

    def select_action(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Select an action."""
        with torch.no_grad():
            if self.shared_net is not None:
                state = self.shared_net.sample(state)
            return self.actor_old.sample(state) | self.critic.sample(state)
