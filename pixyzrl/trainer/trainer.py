from abc import ABC, abstractmethod

import torch

from pixyzrl.environments import BaseEnv
from pixyzrl.logger import Logger
from pixyzrl.memory import BaseBuffer
from pixyzrl.models.base_model import RLModel


class BaseTrainer(ABC):
    """Base class for reinforcement learning trainers."""

    def __init__(self, env: BaseEnv, memory: BaseBuffer, agent: RLModel, device: torch.device | str = "cpu", logger: Logger | None = None):
        """Initialize the trainer.

        :param env: Environment for training.
        :param memory: Memory buffer.
        :param agent: RL agent.
        :param device: Device to run training on.
        :param logger: Optional logger.
        """
        self.env = env
        self.memory = memory
        self.agent = agent
        self.device = device
        self.logger = logger

        if self.logger:
            self.logger.log("Trainer initialized.")

    @abstractmethod
    def collect_experiences(self) -> None:
        """Collect experiences from the environment."""
        ...

    @abstractmethod
    def train_step(self) -> None:
        """Perform a single training step."""
        ...

    def save_model(self, path: str) -> None:
        """Save the trained model."""
        self.agent.save(path)
        if self.logger:
            self.logger.log(f"Model saved at {path}.")

    def load_model(self, path: str) -> None:
        """Load a trained model."""
        self.agent.load(path)
        if self.logger:
            self.logger.log(f"Model loaded from {path}.")


class OnPolicyTrainer(BaseTrainer):
    """Trainer class for on-policy reinforcement learning methods (e.g., PPO, A2C)."""

    def __init__(self, env: BaseEnv, memory: BaseBuffer, agent: RLModel, device: torch.device | str = "cpu", logger: Logger | None = None) -> None:
        super().__init__(env, memory, agent, device, logger)

    def collect_experiences(self) -> None:
        obs, info = self.env.reset()
        done = False

        while len(self.memory) < self.memory.buffer_size:
            action = self.agent.select_action({"o": obs.to(self.device)})

            if self.env.is_discrete:
                next_obs, reward, truncated, terminated, _ = self.env.step(torch.argmax(action[self.agent.action_var].cpu()))
                done = truncated or terminated
            else:
                next_obs, reward, truncated, terminated, _ = self.env.step(action[self.agent.action_var].cpu())
                done = truncated or terminated

            self.memory.add(obs=obs.detach(), action=action[self.agent.action_var].detach(), reward=reward, done=done, value=action[self.agent.critic.var[0]].cpu().detach())
            obs = next_obs

            if done:
                obs, info = self.env.reset()

        action = self.agent.select_action({"o": obs.to(self.device)})
        self.memory.compute_returns_and_advantages_gae(action[self.agent.critic.var[0]].cpu(), 0.99, 0.95)
        if self.logger:
            self.logger.log("Collected on-policy experiences.")

    def train_step(self, batch_size: int = 128, num_epochs: int = 40) -> None:
        if len(self.memory) < self.memory.buffer_size:
            return

        print("Training...")
        total_loss = 0
        for _ in range(num_epochs):
            batch = self.memory.sample(batch_size)
            loss = self.agent.train(batch)
            print(loss)
            total_loss += loss

        self.memory.clear()
        self.agent.actor_old.load_state_dict(self.agent.actor.state_dict())

        if self.logger:
            self.logger.log(f"On-policy training step completed. Loss: {total_loss}")

    def train(self, num_iterations: int, batch_size: int = 128, num_epochs: int = 40) -> None:
        for iteration in range(num_iterations):
            self.collect_experiences()
            self.train_step(batch_size, num_epochs)
            if self.logger:
                self.logger.log(f"On-policy Iteration {iteration + 1}/{num_iterations} completed.")


class OffPolicyTrainer(BaseTrainer):
    """Trainer class for off-policy reinforcement learning methods (e.g., DQN, DDPG)."""

    def __init__(self, env: BaseEnv, memory: BaseBuffer, agent: RLModel, device: torch.device | str = "cpu", logger: Logger | None = None) -> None:
        super().__init__(env, memory, agent, device, logger)

    def collect_experiences(self) -> None:
        obs, info = self.env.reset()
        done = False

        while not done:
            action = self.agent.select_action({"o": obs.to(self.device)})

            if self.env.is_discrete:
                next_obs, reward, done, _, _ = self.env.step(torch.argmax(action[self.agent.action_var].cpu()))
            else:
                next_obs, reward, done, _, _ = self.env.step(action[self.agent.action_var].cpu().numpy())

            self.memory.add(obs=obs, action=action[self.agent.action_var].cpu().numpy(), reward=reward, done=done, value=action[self.agent.critic.var[0]].cpu().detach())
            obs = next_obs

        if self.logger:
            self.logger.log("Collected off-policy experiences.")

    def train_step(self, batch_size: int = 128, num_epochs: int = 4) -> None:
        if len(self.memory) < self.memory.buffer_size:
            return

        for _ in range(num_epochs):
            batch = self.memory.sample(batch_size)
            loss = self.agent.train(batch)

        self.memory.clear()

        if self.logger:
            self.logger.log(f"Off-policy training step completed. Loss: {loss}")

    def train(self, num_iterations: int, batch_size: int = 128, num_epochs: int = 4) -> None:
        for iteration in range(num_iterations):
            self.collect_experiences()
            self.train_step(batch_size, num_epochs)
            if self.logger:
                self.logger.log(f"Off-policy Iteration {iteration + 1}/{num_iterations} completed.")


def create_trainer(env: BaseEnv, memory: BaseBuffer, agent: RLModel, device: torch.device | str = "cpu", logger: Logger | None = None) -> BaseTrainer:
    """Create a trainer based on the type of agent."""
    if agent.is_on_policy:
        return OnPolicyTrainer(env, memory, agent, device, logger)

    return OffPolicyTrainer(env, memory, agent, device, logger)
