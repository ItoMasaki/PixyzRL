from abc import ABC, abstractmethod
from math import e

import torch
from pixyz.models import Model

from pixyzrl.environments import BaseEnv, env
from pixyzrl.logger import Logger
from pixyzrl.memory import BaseBuffer


class BaseTrainer(ABC):
    """Base class for reinforcement learning trainers."""

    def __init__(self, env: BaseEnv, memory: BaseBuffer, agent: Model, device: torch.device | str = "cpu", logger: Logger | None = None):
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

    def __init__(self, env: BaseEnv, memory: BaseBuffer, agent: Model, device: torch.device | str = "cpu", logger: Logger | None = None) -> None:
        super().__init__(env, memory, agent, device, logger)

    def collect_experiences(self) -> None:
        obs, info = self.env.reset()
        done = False

        while not done:
            action = self.agent.select_action({"o": torch.tensor(obs, dtype=torch.float32, device=self.device)})

            if env.is_discrete:
                next_obs, reward, done, _, _ = self.env.step(torch.argmax(action[self.agent.actor.var[0]].cpu().numpy()))
            else:
                next_obs, reward, done, _, _ = self.env.step(action[self.agent.actor.var[0]].cpu().numpy())

            self.memory.add(obs=obs.detach(), action=action[self.agent.actor.var[0]].detach(), reward=reward, done=done, value=action[self.agent.critic.var[0]].detach)
            obs = next_obs

        if self.logger:
            self.logger.log("Collected on-policy experiences.")

    def train_step(self) -> None:
        if len(self.memory) < 128:
            return

        batch = self.memory.sample(128)
        loss = self.agent.train(batch)
        self.memory.clear()

        if self.logger:
            self.logger.log(f"On-policy training step completed. Loss: {loss}")

    def train(self, num_iterations: int) -> None:
        for iteration in range(num_iterations):
            self.collect_experiences()
            self.train_step()
            if self.logger:
                self.logger.log(f"On-policy Iteration {iteration + 1}/{num_iterations} completed.")


class OffPolicyTrainer(BaseTrainer):
    """Trainer class for off-policy reinforcement learning methods (e.g., DQN, DDPG)."""

    def collect_experiences(self) -> None:
        obs, info = self.env.reset()
        done = False

        while not done:
            action = self.agent.select_action({"o": torch.tensor(obs, dtype=torch.float32, device=self.device)})

            if env.is_discrete:
                next_obs, reward, done, _, _ = self.env.step(torch.argmax(action[self.agent.actor.var[0]].cpu().numpy()))
            else:
                next_obs, reward, done, _, _ = self.env.step(action[self.agent.actor.var[0]].cpu().numpy())

            self.memory.add(obs=obs, action=action["a"].cpu().numpy(), reward=reward, done=done, value=action["v"].cpu().numpy())
            obs = next_obs

            self.train_step()  # Train while collecting experiences

        if self.logger:
            self.logger.log("Collected off-policy experiences.")

    def train_step(self) -> None:
        if len(self.memory) < 128:
            return

        batch = self.memory.sample(128)
        loss = self.agent.train(batch)

        if self.logger:
            self.logger.log(f"Off-policy training step completed. Loss: {loss}")

    def train(self, num_iterations: int) -> None:
        for iteration in range(num_iterations):
            self.collect_experiences()
            if self.logger:
                self.logger.log(f"Off-policy Iteration {iteration + 1}/{num_iterations} completed.")


def create_trainer(env: BaseEnv, memory: BaseBuffer, agent: Model, device: torch.device | str = "cpu", logger: Logger | None = None) -> BaseTrainer:
    """Create a trainer based on the type of agent."""
    if agent.is_on_policy:
        return OnPolicyTrainer(env, memory, agent, device, logger)

    return OffPolicyTrainer(env, memory, agent, device, logger)
