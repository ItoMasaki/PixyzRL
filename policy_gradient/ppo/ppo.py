"""Proximal Policy Optimization (PPO) agent using Pixyz."""  # noqa: INP001

from copy import deepcopy
from typing import Any

import torch
from numpy.typing import NDArray
from pixyz import distributions as dists
from pixyz.losses import Entropy, MinLoss, Parameter
from pixyz.losses import Expectation as E  # noqa: N817
from pixyz.models import Model
from torch.optim import Adam

from pixyzrl.losses import ClipLoss, MSELoss, RatioLoss
from pixyzrl.memory import RolloutBuffer


class PPO(Model):
    """PPO agent using Pixyz."""

    def __init__(self, actor: dists.Distribution, critic: dists.Distribution, shared_cnn: dists.Distribution, gamma: float, eps_clip: float, k_epochs: int, lr_actor: float, lr_critic: float, device: str) -> None:
        """Initialize the PPO agent."""
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = k_epochs
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.device = device

        # Shared CNN layers
        self.shared_cnn = shared_cnn

        # Actor network
        self.actor = actor
        self.actor_old = deepcopy(actor)
        self.actor_old.name = "old"

        # Critic network
        self.critic = critic

        # Buffer for storing rollout data
        self.buffer = RolloutBuffer()

        advantage = Parameter("A")
        ratio = RatioLoss(self.actor, self.actor_old)
        clip = ClipLoss(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
        ppo_loss = -MinLoss(clip * advantage, ratio * advantage)

        mse_loss = MSELoss(self.critic, "r")

        loss = E(self.shared_cnn, ppo_loss + 0.5 * mse_loss - 0.01 * Entropy(self.actor)).mean()

        super().__init__(loss, distributions=[self.actor, self.critic, self.shared_cnn], optimizer=Adam, optimizer_params={})

        # Optimizer
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": self.lr_actor},
                {"params": self.critic.parameters(), "lr": self.lr_critic},
            ]
        )

    def store_transition(self, reward: NDArray[Any], done: int | bool) -> None:
        """Store transition data in the buffer."""
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)

    def select_action(self, state: tuple[torch.Tensor | NDArray[Any]] | torch.Tensor | NDArray[Any]) -> NDArray[Any]:
        """Select an action."""

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            s = self.shared_cnn.sample({"o": state})["s"]
            action = self.actor_old.sample({"s": s})["a"]
            state_val = self.critic.sample({"s": s})["v"]

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()

    def update(self) -> None:
        """Update the agent."""
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals), strict=False):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.cat(self.buffer.states, dim=0).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach()

        # Calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            self.train({"o": old_states, "a": old_actions, "A": advantages, "r": rewards})

        # Copy new weights into old policy
        self.actor_old.load_state_dict(self.actor.state_dict())

        # Clear buffer
        self.buffer.clear()
