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

    def __init__(self, env: BaseEnv, memory: BaseBuffer, agent: RLModel, value_estimate: str = "gae", device: torch.device | str = "cpu", logger: Logger | None = None) -> None:
        """Initialize the on-policy trainer.

        Args:
            env (BaseEnv): Environment.
            memory (BaseBuffer): Replay buffer.
            agent (RLModel): Reinforcement learning agent.
            device (torch.device | str): Device to use.
            logger (Logger | None): Logger to use.

        Example:
        >>> import torch
        >>> from pixyz.distributions import Categorical, Deterministic
        >>> from torch import nn

        >>> from pixyzrl.environments import Env
        >>> from pixyzrl.logger import Logger
        >>> from pixyzrl.memory import RolloutBuffer
        >>> from pixyzrl.models import PPO
        >>> from pixyzrl.trainer import OnPolicyTrainer

        >>> env = Env("CartPole-v1")
        >>> action_dim = env.action_space.n

        >>> class Actor(Categorical):
        ...     def __init__(self):
        ...         super().__init__(var=["a"], cond_var=["o"], name="p")
        ...         self.net = nn.Sequential(
        ...             nn.LazyLinear(action_dim),
        ...             nn.Softmax(dim=-1),
        ...     )
        ...     def forward(self, o: torch.Tensor):
        ...         probs = self.net(o)
        ...         return {"probs": probs}

        >>> class Critic(Deterministic):
        ...     def __init__(self):
        ...         super().__init__(var=["v"], cond_var=["o"], name="f")
        ...         self.net = nn.Sequential(
        ...             nn.LazyLinear(1),
        ...         )
        ...     def forward(self, o: torch.Tensor):
        ...         v = self.net(o)
        ...         return {"v": v}

        >>> actor = Actor()
        >>> critic = Critic()

        >>> ppo = PPO(actor, critic, entropy_coef=0.0, mse_coef=1.0)

        >>> buffer = RolloutBuffer(
        ...     2048,
        ...     {
        ...         "obs": {"shape": (4,), "map": "o"},
        ...         "value": {"shape": (1,), "map": "v"},
        ...         "action": {"shape": (2,), "map": "a"},
        ...         "reward": {"shape": (1,)},
        ...         "done": {"shape": (1,)},
        ...         "returns": {"shape": (1,), "map": "r"},
        ...         "advantages": {"shape": (1,), "map": "A"},
        ...     },
        ...     "cpu",
        ...     1,
        ... )
        >>> logger = Logger("logs")
        >>> trainer = OnPolicyTrainer(env, buffer, ppo, "cpu", logger)
        """
        super().__init__(env, memory, agent, device, logger)
        self.value_estimate = value_estimate

    def collect_experiences(self) -> None:
        """Collect experiences from the environment.

        Example:
        >>> import torch
        >>> from pixyz.distributions import Categorical, Deterministic
        >>> from torch import nn

        >>> from pixyzrl.environments import Env
        >>> from pixyzrl.logger import Logger
        >>> from pixyzrl.memory import RolloutBuffer
        >>> from pixyzrl.models import PPO
        >>> from pixyzrl.trainer import OnPolicyTrainer

        >>> env = Env("CartPole-v1")
        >>> action_dim = env.action_space.n

        >>> class Actor(Categorical):
        ...     def __init__(self):
        ...         super().__init__(var=["a"], cond_var=["o"], name="p")
        ...         self.net = nn.Sequential(
        ...             nn.LazyLinear(action_dim),
        ...             nn.Softmax(dim=-1),
        ...     )
        ...     def forward(self, o: torch.Tensor):
        ...         probs = self.net(o)
        ...         return {"probs": probs}

        >>> class Critic(Deterministic):
        ...     def __init__(self):
        ...         super().__init__(var=["v"], cond_var=["o"], name="f")
        ...         self.net = nn.Sequential(
        ...             nn.LazyLinear(1),
        ...         )
        ...     def forward(self, o: torch.Tensor):
        ...         v = self.net(o)
        ...         return {"v": v}

        >>> actor = Actor()
        >>> critic = Critic()

        >>> ppo = PPO(actor, critic, entropy_coef=0.0, mse_coef=1.0)

        >>> buffer = RolloutBuffer(
        ...     100,
        ...     {
        ...         "obs": {"shape": (4,), "map": "o"},
        ...         "value": {"shape": (1,), "map": "v"},
        ...         "action": {"shape": (2,), "map": "a"},
        ...         "reward": {"shape": (1,)},
        ...         "done": {"shape": (1,)},
        ...         "returns": {"shape": (1,), "map": "r"},
        ...         "advantages": {"shape": (1,), "map": "A"},
        ...     },
        ...     "cpu",
        ...     1,
        ... )
        >>> logger = Logger("logs")
        >>> trainer = OnPolicyTrainer(env, buffer, ppo, "cpu")
        >>> trainer.collect_experiences()
        """
        obs, info = self.env.reset()
        done = False
        total_reward = 0

        with torch.no_grad():
            while len(self.memory) < self.memory.buffer_size - 1:
                if len(obs.shape) == 1:
                    obs = obs.unsqueeze(0)
                elif len(obs.shape) == 3:
                    obs = obs.permute(2, 0, 1).unsqueeze(0) / 255.0

                action = self.agent.select_action({"o": obs.to(self.device)})

                if self.env.is_discrete:
                    next_obs, reward, truncated, terminated, _ = self.env.step(torch.argmax(action[self.agent.action_var].cpu()))
                    done = truncated or terminated
                else:
                    next_obs, reward, truncated, terminated, _ = self.env.step(action[self.agent.action_var].cpu().squeeze())
                    done = truncated or terminated

                self.memory.add(obs=obs.detach(), action=action[self.agent.action_var].detach(), reward=reward.detach(), done=done.detach(), value=action[self.agent.critic.var[0]].cpu().detach())
                obs = next_obs
                total_reward += reward

                if done:
                    if self.logger:
                        self.logger.log(f"Collected on-policy experiences. Total reward: {total_reward.detach().item()}")

                    obs, info = self.env.reset()
                    done = False
                    total_reward = 0

            if self.logger:
                self.logger.log(f"Collected on-policy experiences. Total reward: {total_reward.detach().item()}")

            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
            elif len(obs.shape) == 3:
                obs = obs.permute(2, 0, 1).unsqueeze(0)

            if self.value_estimate == "gae":
                self.memory.compute_returns_and_advantages_gae()
            elif self.value_estimate == "mc":
                self.memory.compute_returns_and_advantages_mc()
            else:
                pass

    def train_model(self, batch_size: int = 128, num_epochs: int = 40) -> None:
        """Perform a single training step.

        Args:
            batch_size (int, optional): Batch size for training. Defaults to 128.
            num_epochs (int, optional): Number of epochs for training. Defaults to 40.

        Example:
        >>> import torch
        >>> from pixyz.distributions import Categorical, Deterministic
        >>> from torch import nn

        >>> from pixyzrl.environments import Env
        >>> from pixyzrl.logger import Logger
        >>> from pixyzrl.memory import RolloutBuffer
        >>> from pixyzrl.models import PPO
        >>> from pixyzrl.trainer import OnPolicyTrainer

        >>> env = Env("CartPole-v1")
        >>> action_dim = env.action_space.n

        >>> class Actor(Categorical):
        ...     def __init__(self):
        ...         super().__init__(var=["a"], cond_var=["o"], name="p")
        ...         self.net = nn.Sequential(
        ...             nn.LazyLinear(action_dim),
        ...             nn.Softmax(dim=-1),
        ...     )
        ...     def forward(self, o: torch.Tensor):
        ...         probs = self.net(o)
        ...         return {"probs": probs}

        >>> class Critic(Deterministic):
        ...     def __init__(self):
        ...         super().__init__(var=["v"], cond_var=["o"], name="f")
        ...         self.net = nn.Sequential(
        ...             nn.LazyLinear(1),
        ...         )
        ...     def forward(self, o: torch.Tensor):
        ...         v = self.net(o)
        ...         return {"v": v}

        >>> actor = Actor()
        >>> critic = Critic()

        >>> ppo = PPO(actor, critic, entropy_coef=0.0, mse_coef=1.0)

        >>> buffer = RolloutBuffer(
        ...     100,
        ...     {
        ...         "obs": {"shape": (4,), "map": "o"},
        ...         "value": {"shape": (1,), "map": "v"},
        ...         "action": {"shape": (2,), "map": "a"},
        ...         "reward": {"shape": (1,)},
        ...         "done": {"shape": (1,)},
        ...         "returns": {"shape": (1,), "map": "r"},
        ...         "advantages": {"shape": (1,), "map": "A"},
        ...     },
        ...     "cpu",
        ...     1,
        ... )
        >>> logger = Logger("logs")
        >>> trainer = OnPolicyTrainer(env, buffer, ppo, "cpu")
        >>> trainer.collect_experiences()
        >>> trainer.train_model()
        """
        if len(self.memory) < self.memory.buffer_size - 1:
            return

        total_loss = self.agent.train_step(self.memory, batch_size, num_epochs)

        if self.logger:
            self.logger.log(f"On-policy training step completed. Loss: {total_loss}")

    def train(self, num_iterations: int, batch_size: int = 128, num_epochs: int = 40) -> None:
        """Train the agent.

        Args:
            num_iterations (int): Number of training iterations.
            batch_size (int, optional): Batch size for training. Defaults to 128.
            num_epochs (int, optional): Number of epochs for training. Defaults to 40.

        Example:
        >>> import torch
        >>> from pixyz.distributions import Categorical, Deterministic
        >>> from torch import nn

        >>> from pixyzrl.environments import Env
        >>> from pixyzrl.logger import Logger
        >>> from pixyzrl.memory import RolloutBuffer
        >>> from pixyzrl.models import PPO
        >>> from pixyzrl.trainer import OnPolicyTrainer

        >>> env = Env("CartPole-v1")
        >>> action_dim = env.action_space.n

        >>> class Actor(Categorical):
        ...     def __init__(self):
        ...         super().__init__(var=["a"], cond_var=["o"], name="p")
        ...         self.net = nn.Sequential(
        ...             nn.LazyLinear(action_dim),
        ...             nn.Softmax(dim=-1),
        ...     )
        ...     def forward(self, o: torch.Tensor):
        ...         probs = self.net(o)
        ...         return {"probs": probs}

        >>> class Critic(Deterministic):
        ...     def __init__(self):
        ...         super().__init__(var=["v"], cond_var=["o"], name="f")
        ...         self.net = nn.Sequential(
        ...             nn.LazyLinear(1),
        ...         )
        ...     def forward(self, o: torch.Tensor):
        ...         v = self.net(o)
        ...         return {"v": v}

        >>> actor = Actor()
        >>> critic = Critic()

        >>> ppo = PPO(actor, critic, entropy_coef=0.0, mse_coef=1.0)

        >>> buffer = RolloutBuffer(
        ...     100,
        ...     {
        ...         "obs": {"shape": (4,), "map": "o"},
        ...         "value": {"shape": (1,), "map": "v"},
        ...         "action": {"shape": (2,), "map": "a"},
        ...         "reward": {"shape": (1,)},
        ...         "done": {"shape": (1,)},
        ...         "returns": {"shape": (1,), "map": "r"},
        ...         "advantages": {"shape": (1,), "map": "A"},
        ...     },
        ...     "cpu",
        ...     1,
        ... )
        >>> logger = Logger("logs")
        >>> trainer = OnPolicyTrainer(env, buffer, ppo, "cpu")
        >>> trainer.train(1)
        """

        for iteration in range(num_iterations):
            self.collect_experiences()
            self.train_model(batch_size, num_epochs)
            self.memory.clear()
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

    def train_model(self, batch_size: int = 128, num_epochs: int = 4) -> None:
        if len(self.memory) < self.memory.buffer_size:
            return

        total_loss = self.agent.train_step(self.memory, batch_size, num_epochs)

        if self.logger:
            self.logger.log(f"Off-policy training step completed. Loss: {total_loss / num_epochs}")

    def train(self, num_iterations: int, batch_size: int = 128, num_epochs: int = 4) -> None:
        for iteration in range(num_iterations):
            self.collect_experiences()
            self.train_model(batch_size, num_epochs)
            if self.logger:
                self.logger.log(f"Off-policy Iteration {iteration + 1}/{num_iterations} completed.")


def create_trainer(env: BaseEnv, memory: BaseBuffer, agent: RLModel, device: torch.device | str = "cpu", logger: Logger | None = None) -> BaseTrainer:
    """Create a trainer based on the type of agent."""
    if agent.is_on_policy:
        return OnPolicyTrainer(env, memory, agent, device, logger)

    return OffPolicyTrainer(env, memory, agent, device, logger)
