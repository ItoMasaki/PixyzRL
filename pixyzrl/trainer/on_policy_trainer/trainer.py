from datetime import datetime
from threading import Thread
from typing import Any

import numpy as np
import torch
from matplotlib import animation
from matplotlib import pyplot as plt
from torchvision.transforms import Compose

from pixyzrl.environments import BaseEnv
from pixyzrl.logger import Logger
from pixyzrl.memory import BaseBuffer
from pixyzrl.models.base_model import RLModel
from pixyzrl.trainer.base_trainer import BaseTrainer


class OnPolicyTrainer(BaseTrainer):
    """Trainer class for on-policy reinforcement learning methods (e.g., PPO, A2C)."""

    def __init__(
        self,
        env: BaseEnv,
        memory: BaseBuffer,
        agent: RLModel,
        value_estimate: str = "gae",
        device: torch.device | str = "cpu",
        logger: Logger | None = None,
        transform: Compose | None = None,
    ) -> None:
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
        >>> action_dim = env.action_space

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
        self.transform = transform
        memory.device = device
        self.episode = 0
        self.log_dir = (
            logger.log_dir
            if logger
            else f"logs/{datetime.now(datetime.now().astimezone().tzinfo).strftime('%Y%m%d-%H%M%S')}"
        )

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
        total_rewards = []
        idx = 1

        with torch.no_grad():
            # Collect on-policy experiences
            while len(self.memory) < self.memory.buffer_size - 1:
                idx += 1

                if len(obs[0].shape) == 3:
                    obs = obs.permute(0, 3, 1, 2) / 255.0

                action = self.agent.select_action({"o": obs.to(self.device)})
                next_obs, reward, terminated, truncated, info = self.env.step(
                    action[self.agent.action_var].cpu()
                )
                done = torch.logical_or(terminated, truncated)

                self.memory.add(
                    obs=obs.detach(),
                    action=action[self.agent.action_var].detach(),
                    reward=reward.detach(),
                    done=done.detach(),
                    value=action[self.agent.critic.var[0]].cpu().detach(),
                )
                obs = next_obs
                total_reward += reward.detach().numpy().astype(float)

                if done.any():
                    for _reward in total_reward[done]:
                        total_rewards.append(np.round(_reward.item(), 1))

                total_reward *= 1 - done.detach().numpy().astype(bool)

                if self.env.render_mode == "rgb_array":
                    self.env.render()

            total_rewards += np.round(total_reward.reshape(-1), 1).tolist()

            if self.logger:
                self.logger.log(f"Rewards: {' '.join([str(r) for r in total_rewards])}")

            if self.value_estimate == "gae":
                if self.logger:
                    self.logger.log("Computing GAE returns and advantages.")
                self.memory.compute_returns_and_advantages_gae()
            elif self.value_estimate == "mc":
                if self.logger:
                    self.logger.log("Computing Monte Carlo returns and advantages.")
                self.memory.compute_returns_and_advantages_mc()
            elif self.value_estimate == "grpo":
                if self.logger:
                    self.logger.log("Computing GRPO returns and advantages.")
                self.memory.compute_advantages_grpo()
            else:
                if self.logger:
                    self.logger.log("Invalid value estimate method. Using GAE instead.")
                self.memory.compute_returns_and_advantages_gae()

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

        total_loss = 0
        for _ in range(num_epochs):
            total_loss += self.agent.train_step(self.memory, batch_size)

        self.agent.transfer_state_dict()

        if self.logger:
            self.logger.log(
                f"On-policy training step completed. Loss: {total_loss/num_epochs}"
            )

    def test(self) -> None:
        """Test the agent.

        Args:
            num_episodes (int, optional): Number of episodes to test. Defaults to 10.

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
        ...         )
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
        ...         "advantages": {"shape": (1,), "map":
        ...     },
        ...     "cpu",
        ...     1,
        ... )
        >>> logger = Logger("logs")
        >>> trainer = OnPolicyTrainer(env, buffer, ppo, "cpu")
        >>> trainer.test(10)
        """
        total_reward = 0
        total_rewards = []

        obs, info = self.env.reset()
        idx = 0
        self.frames = []

        with torch.no_grad():
            for _ in range(5):
                done = [False]

                while not done[0]:
                    idx += 1

                    if len(obs[0].shape) == 3:
                        obs = obs.permute(0, 3, 1, 2) / 255.0

                    action = self.agent.select_action({"o": obs.to(self.device)})
                    next_obs, reward, terminated, truncated, _ = self.env.step(
                        action[self.agent.action_var].cpu().numpy()
                    )
                    done = torch.logical_or(terminated, truncated)

                    obs = next_obs

                    total_reward += reward.detach().numpy().astype(float)

                    if done[0]:
                        idx = 0
                        self.frames.append(np.zeros_like(self.frames[-1]))
                        total_rewards.append(np.round(total_reward[0].item(), 1))
                        total_reward *= 1 - done[0].detach().numpy()
                        break

                    self.frames.append(self.env.render(return_frame=True)[0])

        fig = plt.figure()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        Thread(target=self.save_video, args=(fig, self.frames)).start()

        if self.logger:
            self.logger.log(f"Testing completed. Total reward: {total_rewards}")

    def save_video(self, fig: plt.Figure, frames: list[Any]) -> None:
        """Save the test video."""
        ims = []
        for frame in frames:
            plt.axis("off")
            im = plt.imshow(frame)
            ims.append([im])

        ani = animation.ArtistAnimation(
            fig, ims, interval=50, blit=True, repeat_delay=1000
        )
        ani.save(f"{self.log_dir}/test_{self.episode - 1}.mp4")
        fig.clear()

    def train(
        self,
        num_iterations: int,
        batch_size: int = 128,
        num_epochs: int = 40,
        test_interval: int = 100,
        save_interval: int = 10,
    ) -> None:
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
        ...             nn.LazyLinear(64),
        ...             nn.ReLU(),
        ...             nn.LazyLinear(action_dim),
        ...             nn.ReLU(),
        ...             nn.Softmax(dim=-1),
        ...         )
        ...     def forward(self, o: torch.Tensor):
        ...         probs = self.net(o)
        ...         return {"probs": probs}

        >>> class Critic(Deterministic):
        ...     def __init__(self):
        ...         super().__init__(var=["v"], cond_var=["o"], name="f")
        ...         self.net = nn.Sequential(
        ...             nn.LazyLinear(64),
        ...             nn.ReLU(),
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
            self.episode += 1

            self.collect_experiences()
            self.train_model(batch_size, num_epochs)
            self.memory.clear()

            if self.logger:
                self.logger.log(
                    f"On-policy Iteration {iteration + 1}/{num_iterations} completed."
                )

            if (iteration + 1) % test_interval == 0:
                self.test()

            if (iteration + 1) % save_interval == 0:
                self.save_model(f"{self.log_dir}/model_{iteration + 1}.pt")
