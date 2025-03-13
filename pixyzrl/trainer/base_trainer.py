from abc import ABC, abstractmethod

import torch

from pixyzrl.environments import BaseEnv
from pixyzrl.logger import Logger
from pixyzrl.memory import BaseBuffer
from pixyzrl.models.base_model import RLModel


class BaseTrainer(ABC):
    """Base class for reinforcement learning trainers."""

    def __init__(self, env: BaseEnv, memory: BaseBuffer, agent: RLModel, device: torch.device | str = "cpu", logger: Logger | None = None) -> None:
        """Initialize the trainer.

        Args:
            env (BaseEnv): Environment.
            memory (BaseBuffer): Replay buffer.
            agent (RLModel): Reinforcement learning agent.
            device (torch.device | str): Device to use.
            logger (Logger | None): Logger to use.
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
    def train_model(self) -> None:
        """Perform a single training step."""
        ...

    def save_model(self, path: str) -> None:
        """Save the trained model."""
        dists = [dist.state_dict() for dist in self.agent.distributions]
        torch.save({"model": dists, "optimizer": self.agent.optimizer.state_dict()}, path)
        if self.logger:
            self.logger.log(f"Model saved at {path}.")

    def load_model(self, path: str) -> None:
        """Load a trained model."""
        checkpoint = torch.load(path)
        dists = [dist.state_dict() for dist in self.agent.distributions]
        for dist, state_dict in zip(dists, checkpoint["model"], strict=False):
            dist.load_state_dict(state_dict)
        self.agent.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.logger:
            self.logger.log(f"Model loaded from {path}.")
