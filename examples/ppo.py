"""Proximal Policy Optimization (PPO) agent using Pixyz."""  # noqa: INP001

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from pixyz import distributions as dists
from torch import nn

from pixyzrl.memory import RolloutBuffer
from pixyzrl.policy_gradient.ppo import PPO

################################## set device ##################################


TWO_CH = 2
THREE_CH = 3

# Buffer for storing rollout data
buffer = RolloutBuffer()


class FeatureExtractor(dists.Deterministic):
    """Feature extractor network."""

    def __init__(self) -> None:
        """Initialize the feature extractor network."""
        super().__init__(cond_var=["o"], var=["s"], name="f")

        self.feature_extract = nn.Sequential(
            nn.LazyConv2d(32, kernel_size=8, stride=4),
            nn.SiLU(),
            nn.LazyConv2d(64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.LazyConv2d(64, kernel_size=3, stride=1),
            nn.SiLU(),
            nn.Flatten(),
        )

    def forward(self, o: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass."""
        return {"s": self.feature_extract(o)}


class Actor(dists.Normal):
    """Actor network."""

    def __init__(self, action_dim: int, name: str) -> None:
        """Initialize the actor network."""
        super().__init__(cond_var=["s"], var=["a"], name=name)

        self.loc = nn.Sequential(
            nn.LazyLinear(512),
            nn.SiLU(),
            nn.LazyLinear(action_dim),
            nn.Tanh(),
        )

        self.scale = nn.Sequential(
            nn.LazyLinear(512),
            nn.SiLU(),
            nn.LazyLinear(action_dim),
            nn.Softplus(),
        )

    def forward(self, s: torch.Tensor) -> dict[str, torch.Tensor]:  # type: ignore  # noqa: PGH003
        """Forward pass."""
        loc = self.loc(s)
        scale = self.scale(s)
        return {"loc": loc, "scale": scale}


class Critic(dists.Deterministic):
    """Critic network."""

    def __init__(self) -> None:
        """Initialize the critic network."""
        super().__init__(var=["v"], cond_var=["s"], name="critic")

        self.value = nn.Sequential(
            nn.LazyLinear(512),
            nn.SiLU(),
            nn.LazyLinear(1),
        )

    def forward(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass."""
        return {"v": self.value(s)}


def preprocess_state(state: NDArray[Any]) -> NDArray[Any]:
    """Preprocess the state."""
    # 正規化 (0-255 -> 0-1)
    if state.dtype == np.uint8:
        state = state.astype(np.float32) / 255.0

    # チャネルの順序を変更 (H, W, C) -> (C, H, W)
    if len(state.shape) == THREE_CH:
        state = np.transpose(state, (2, 0, 1))

    return state


def main() -> None:
    """Main function."""
    action_dim = 3
    gamma = 0.99
    eps_clip = 0.2
    k_epochs = 80
    lr_actor = 0.0003
    lr_critic = 0.001
    max_episodes = 5000
    update_episodes = 5
    device = "mps"

    # Shared CNN layers
    shared_cnn = FeatureExtractor().to(device)

    # Actor network
    actor = Actor(action_dim, "new").to(device)

    # Critic network
    critic = Critic().to(device)

    agent = PPO(actor, critic, shared_cnn, gamma, eps_clip, k_epochs, lr_actor, lr_critic, device)

    # Define environment
    env = gym.make("CarRacing-v3", render_mode="rgb_array")

    # Training loop
    for i_episode in range(max_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            state = preprocess_state(state)
            action = agent.select_action(state)
            state, reward, truncated, terminated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(reward, done)
            total_reward += reward
            print(f" Episode: {i_episode + 1}, Total reward: {total_reward}       ", end="\r")

            # plt.cla()
            # plt.imshow(state)
            # plt.pause(0.01)

        if i_episode % update_episodes == 0:
            agent.update()

        if i_episode % 100 == 0:
            agent.save("ppo_agent.pth")

        print()


if __name__ == "__main__":
    main()


def store_transition(reward: NDArray[Any], done: int | bool) -> None:
    """Store transition data in the buffer."""
    buffer.rewards.append(reward)
    buffer.is_terminals.append(done)


def select_action(state: tuple[torch.Tensor | NDArray[Any]] | torch.Tensor | NDArray[Any]) -> NDArray[Any]:
    """Select an action."""

    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        s = shared_cnn.sample({"o": state})["s"]
        action = actor_old.sample({"s": s})["a"]
        state_val = critic.sample({"s": s})["v"]

    buffer.states.append(state)
    buffer.actions.append(action)
    buffer.state_values.append(state_val)

    return action.detach().cpu().numpy().flatten()


def update() -> None:
    """Update the agent."""
    # Monte Carlo estimate of returns
    rewards = []
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals), strict=False):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        rewards.insert(0, discounted_reward)

    # Normalizing the rewards
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

    # Convert list to tensor
    old_states = torch.cat(buffer.states, dim=0).detach()
    old_actions = torch.squeeze(torch.stack(buffer.actions, dim=0)).detach()
    old_state_values = torch.squeeze(torch.stack(buffer.state_values, dim=0)).detach()

    # Calculate advantages
    advantages = rewards.detach() - old_state_values.detach()

    # Optimize policy for K epochs
    print()
    for idx, _ in enumerate(range(K_epochs)):
        # Evaluating old actions and values
        loss = train({"o": old_states.detach(), "a": old_actions, "A": advantages, "r": rewards})
        print(f" {idx + 1} Loss: {loss}", end="\r")

    # Copy new weights into old policy
    actor_old.load_state_dict(actor.state_dict())

    # Clear buffer
    buffer.clear()
