import torch
from pixyz.distributions import Categorical, Deterministic
from torch import nn

from pixyzrl.environments import Env
from pixyzrl.logger import Logger
from pixyzrl.losses import PPOClipLoss
from pixyzrl.memory import RolloutBuffer
from pixyzrl.memory.memory import BaseBuffer
from pixyzrl.models import PPO, RLModel
from pixyzrl.trainer import OnPolicyTrainer
from pixyzrl.utils import print_latex

env = Env("CartPole-v1", 2, render_mode="rgb_array")
action_dim = env.action_space
obs_dim = env.observation_space


class Actor(Categorical):
    def __init__(self):
        super().__init__(var=["a"], cond_var=["s"], name="p")

        self._prob = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, s: torch.Tensor):
        probs = self._prob(s)
        return {"probs": probs}


class Critic(Deterministic):
    def __init__(self):
        super().__init__(var=["v"], cond_var=["s"], name="f")

        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, s: torch.Tensor):
        v = self.net(s)
        return {"v": v}


actor = Actor()
actor_old = Actor()
critic = Critic()

loss = PPOClipLoss(actor, actor_old, 0.2)


class OnPolicy(RLModel):
    def __init__(self) -> None:
        pass

    def select_action(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        pass

    def train_step(
        self, memory: BaseBuffer, batch_size: int = 128, num_epochs: int = 4
    ) -> float:
        return super().train_step(memory, batch_size, num_epochs)


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
        scheduler: lr_scheduler.LRScheduler | None = None,
        **kwargs: dict[str, Any],
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
        self.scheduler = scheduler

        # Shared CNN layers (optional)
        self.shared_net = shared_net

        # Actor network
        self.actor = actor
        self.actor_old = deepcopy(actor).to(device)
        self.actor_old.name += "_{old}"

        # Critic network
        self.critic = critic

        ppo_loss = PPOClipLoss(self.actor, self.actor_old, self.eps_clip, "A")

        self.mse_loss = MSELoss(self.critic, "r", reduction="none")
        # self.mse_loss = ValueClipLoss(self.critic, "r", 0.2)

        if self.shared_net is not None:  # A2C
            loss = E(
                self.shared_net,
                ppo_loss
                # + self.mse_coef * self.mse_loss
                - self.entropy_coef * Entropy(self.actor),
            ).mean()
        else:  # TRPO
            loss = (
                ppo_loss
                # + self.mse_coef * self.mse_loss
                - self.entropy_coef * Entropy(self.actor)
            ).mean()

        super().__init__(
            loss,
            distributions=[self.actor, self.critic]
            + ([self.shared_net] if self.shared_net else []),
            optimizer=Adam,
            optimizer_params={},
            **kwargs,
        )

        for dist in self.distributions:
            dist.to(device)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": self.lr_actor},
                {"params": self.critic.parameters(), "lr": self.lr_critic},
            ],
        )

        if self.scheduler is not None:
            self.scheduler = self.scheduler(self.optimizer, **kwargs)

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

    def train_step(self, memory: BaseBuffer, batch_size: int = 128) -> float:
        """Perform a single training step.

        Args:
            memory (BaseBuffer): Replay buffer.
            batch_size (int): Batch size.

        Returns:
            float: Total loss.
        """
        total_loss = 0

        dataloader = DataLoader(memory, batch_size=batch_size, shuffle=True)

        for batch in dataloader:
            loss = self.train(batch)
            total_loss += loss

        return total_loss / len(dataloader)

    def transfer_state_dict(self) -> None:
        """Transfer the state dictionary.

        Args:
            state_dict (dict[str, torch.Tensor]): State dictionary.
        """
        self.actor_old.load_state_dict(self.actor.state_dict())


# ppo = PPO(
#     actor,
#     critic,
#     entropy_coef=0.01,
#     mse_coef=0.5,
#     lr_actor=1e-4,
#     lr_critic=3e-4,
#     device="mps",
#     # clip_grad_norm=0.5,
# )
# print_latex(ppo)

# buffer = RolloutBuffer(
#     1024,
#     {
#         "obs": {
#             "shape": (*obs_dim,),
#             "map": "o",
#         },
#         "value": {
#             "shape": (1,),
#             "map": "v",
#         },
#         "action": {
#             "shape": (action_dim,),
#             "map": "a",
#         },
#         "reward": {
#             "shape": (1,),
#         },
#         "done": {
#             "shape": (1,),
#         },
#         "returns": {
#             "shape": (1,),
#             "map": "r",
#         },
#         "advantages": {
#             "shape": (1,),
#             "map": "A",
#         },
#     },
#     2,
#     advantage_normalization=True,
#     lam=0.95,
#     gamma=0.99,
# )

# logger = Logger("cartpole_v1_ppo_discrete_trainer", log_types=["print"])

# trainer = OnPolicyTrainer(env, buffer, ppo, "gae", "mps", logger=logger)
# trainer.train(1000000, 32, 10, save_interval=50, test_interval=20)
