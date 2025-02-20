"""Proximal Policy Optimization (PPO) agent using Pixyz."""

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pixyz.losses import Entropy, MinLoss, Parameter
from pixyz.losses import Expectation as E
from pixyz.models import Model
from torch.optim import Adam

from utils.utils import get_env_properties

from .losses import ClipLoss, MSELoss, RatioLoss
from .model import Actor, Critic, FeatureExtractor

################################## set device ##################################
device = torch.device("mps")

TWO_CH = 2
THREE_CH = 3


class RolloutBuffer:
    """Buffer for storing rollout data."""

    def __init__(self) -> None:
        """Initialize the buffer."""
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self) -> None:
        """Clear the buffer."""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class PPO(Model):
    """PPO agent using Pixyz."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the PPO agent."""
        env_properties = get_env_properties(config["env"]["name"])
        self.action_dim = env_properties["action_dim"]
        self.gamma = config["agent"]["hyperparameters"]["gamma"]
        self.eps_clip = config["agent"]["hyperparameters"]["eps_clip"]
        self.K_epochs = config["agent"]["hyperparameters"]["K_epochs"]
        self.lr_actor = config["agent"]["hyperparameters"]["lr_actor"]
        self.lr_critic = config["agent"]["hyperparameters"]["lr_critic"]

        # Shared CNN layers
        self.shared_cnn = FeatureExtractor().to(device)

        # Actor network
        self.actor = Actor(self.action_dim, "new").to(device)
        self.actor_old = Actor(self.action_dim, "old").to(device)

        # Critic network
        self.critic = Critic().to(device)

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
            ],
        )

    def preprocess_state(self, state: tuple[torch.Tensor | NDArray[Any]] | torch.Tensor | NDArray[Any]) -> NDArray[Any]:
        """Preprocess the state."""
        if isinstance(state, tuple):
            state = state[0]

        # NumPy配列に変換
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()

        # グレースケールの場合は3チャネルに複製
        if len(state.shape) == TWO_CH:
            state = np.stack([state] * 3, axis=-1)

        # 正規化 (0-255 -> 0-1)
        if state.dtype == np.uint8:
            state = state.astype(np.float32) / 255.0

        # チャネルの順序を変更 (H, W, C) -> (C, H, W)
        if len(state.shape) == THREE_CH:
            state = np.transpose(state, (2, 0, 1))

        return state

    def store_transition(self, reward: NDArray[Any], done: int | bool) -> None:
        """Store transition data in the buffer."""
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)

    def select_action(self, state: tuple[torch.Tensor | NDArray[Any]] | torch.Tensor | NDArray[Any]) -> NDArray[Any]:
        """Select an action."""
        state = self.preprocess_state(state)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
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
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
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
