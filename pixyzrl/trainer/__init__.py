import torch

from pixyzrl.environments import BaseEnv
from pixyzrl.logger import Logger
from pixyzrl.memory import BaseBuffer
from pixyzrl.models.base_model import RLModel

from .base_trainer import BaseTrainer
from .off_policy_trainer.trainer import OffPolicyTrainer
from .on_policy_trainer.trainer import OnPolicyTrainer


def create_trainer(env: BaseEnv, memory: BaseBuffer, agent: RLModel, device: torch.device | str = "cpu", logger: Logger | None = None) -> BaseTrainer:
    """Create a trainer based on the type of agent.

    Args:
        env (BaseEnv): Environment.
        memory (BaseBuffer): Replay buffer.
        agent (RLModel): Reinforcement learning agent.
        device (torch.device | str): Device to use.
        logger (Logger | None): Logger to use.

    Returns:
        BaseTrainer: Trainer instance.

    Example:
    """
    if agent.is_on_policy:
        return OnPolicyTrainer(env, memory, agent, device, logger)

    return OffPolicyTrainer(env, memory, agent, device, logger)


__all__ = ["BaseTrainer", "OffPolicyTrainer", "OnPolicyTrainer", "create_trainer"]
