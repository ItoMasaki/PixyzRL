"""Proximal Policy Optimization (PPO) agent using Pixyz."""

from copy import deepcopy

import torch
from pixyz import distributions as dists
from pixyz.losses import Entropy, MinLoss, Parameter
from pixyz.losses import Expectation as E  # noqa: N817
from torch.optim import Adam

from pixyzrl.losses import ClipLoss, MSELoss, RatioLoss
from pixyzrl.memory import BaseBuffer
from pixyzrl.models.base_model import RLModel


class PPO(RLModel):
    """PPO agent using Pixyz."""

    def __init__(
        self,
        actor: dists.Distribution,
        critic: dists.Distribution,
        shared_net: dists.Distribution | None = None,
        eps_clip: float = 0.2,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        device: str = "cpu",
        mse_coef: float = 0.5,
        entropy_coef: float = 0.01,
        action_var: str = "a",
    ) -> None:
        """Initialize the PPO agent.

        Args:
            actor (dists.Distribution): Actor network.
            critic (dists.Distribution): Critic network.
            shared_net (dists.Distribution): Shared network (optional).
            eps_clip (float): PPO clip parameter.
            lr_actor (float): Actor learning rate.
            lr_critic (float): Critic learning rate.
            device (str): Device to use.
            mse_coef (float): Coefficient for the MSE loss.
            entropy_coef (float): Coefficient for the entropy loss.
            action_var (str): Action variable

        Examples:
            >>> import torch
            >>> from torch.nn import functional as F
            >>> from pixyz.distributions import Bernoulli, Normal
            >>> from pixyz.models import Model
            >>> from pixyzrl.models import PPO
            >>>
            >>> class P(Normal):
            ...     def __init__(self):
            ...         super().__init__(var=["z"],cond_var=["x"],name="p")
            ...         self.fc1 = torch.nn.Linear(128, 128)
            ...     def forward(self, x):
            ...         h = F.relu(self.fc1(x))
            ...         return {"loc": h, "scale": F.softplus(h)}
            >>>
            >>> class Q(Normal):
            ...     def __init__(self):
            ...         super().__init__(var=["z"],cond_var=["x"],name="q")
            ...         self.fc1 = torch.nn.Linear(128, 128)
            ...     def forward(self, x):
            ...         h = F.relu(self.fc1(x))
            ...         return {"loc": h, "scale": F.softplus(h)}
            >>>
            >>> actor = P()
            >>> critic = P()
            >>> ppo = PPO(actor, critic)
        """
        self.mse_coef = mse_coef
        self.entropy_coef = entropy_coef
        self.eps_clip = eps_clip
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self._is_on_policy = True
        self._action_var = action_var

        # Shared CNN layers (optional)
        self.shared_net = shared_net

        # Actor network
        self.actor = actor
        self.actor_old = deepcopy(actor).to(device)
        self.actor_old.name += "_{old}"

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

        for dist in self.distributions:
            dist.to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": self.lr_actor, "clip_grad_value": 0.5, "clip_grad_norm": 0.5},
                {"params": self.critic.parameters(), "lr": self.lr_critic, "clip_grad_value": 0.5, "clip_grad_norm": 0.5},
            ],
        )

    def select_action(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Select an action.

        Args:
            state (dict[str, torch.Tensor]): State.

        Returns:
            dict[str, torch.Tensor]: Action.

        Examples:
            >>> import torch
            >>> from torch.nn import functional as F
            >>> from pixyz.distributions import Bernoulli, Normal
            >>> from pixyz.models import Model
            >>> from pixyzrl.models import PPO
            >>>
            >>> class P(Normal):
            ...     def __init__(self):
            ...         super().__init__(var=["z"],cond_var=["x"],name="p")
            ...         self.fc1 = torch.nn.Linear(128, 128)
            ...     def forward(self, x):
            ...         h = F.relu(self.fc1(x))
            ...         return {"loc": h, "scale": F.softplus(h)}
            >>>
            >>> class Q(Normal):
            ...     def __init__(self):
            ...         super().__init__(var=["z"],cond_var=["x"],name="q")
            ...         self.fc1 = torch.nn.Linear(128, 128)
            ...     def forward(self, x):
            ...         h = F.relu(self.fc1(x))
            ...         return {"loc": h, "scale": F.softplus(h)}
            >>>
            >>> actor = P()
            >>> critic = P()
            >>> ppo = PPO(actor, critic)
            >>> state = {"x": torch.zeros(1, 128)}
            >>> ppo.select_action(state)["z"].shape
            torch.Size([1, 128])
        """
        with torch.no_grad():
            if self.shared_net is not None:
                state = self.shared_net.sample(state)
            return self.actor_old.sample(state) | self.critic.sample(state)

    def train_step(self, memory: BaseBuffer, batch_size: int = 128, num_epochs: int = 4) -> float:
        """Perform a single training step.

        Args:
            memory (BaseBuffer): Replay buffer.
            batch_size (int): Batch size.
            num_epochs (int): Number of epochs.

        Returns:
            float: Total loss.
        """
        total_loss = 0

        for _ in range(num_epochs):
            batch = memory.sample(batch_size)
            loss = self.train(batch)
            total_loss += loss

        self.actor_old.load_state_dict(self.actor.state_dict())

        return total_loss / num_epochs
