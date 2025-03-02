import torch
from pixyz.distributions import Deterministic, Normal
from torch import nn
from torch.nn import functional as F

from pixyzrl.environments import Env
from pixyzrl.logger import Logger
from pixyzrl.memory import RolloutBuffer
from pixyzrl.models import PPO
from pixyzrl.trainer import OnPolicyTrainer

env = Env("CarRacing-v3")
action_dim = env.action_space.shape[0]


class Actor(Normal):
    def __init__(self):
        super().__init__(var=["a"], cond_var=["o"], name="p")

        self.feature_extract = nn.Sequential(
            nn.LazyConv2d(32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.LazyConv2d(64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.LazyConv2d(64, kernel_size=3, stride=1),
            nn.Flatten(),
        )

        self.net = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(action_dim * 2),
        )

    def forward(self, o: torch.Tensor):
        h = self.feature_extract(o)
        out = self.net(h)
        mean, log_std = out.chunk(2, dim=-1)
        return {"loc": F.tanh(mean), "scale": F.softplus(log_std) + 1e-5}


class Critic(Deterministic):
    def __init__(self):
        super().__init__(var=["v"], cond_var=["o"], name="f")

        self.feature_extract = nn.Sequential(
            nn.LazyConv2d(32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.LazyConv2d(64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.LazyConv2d(64, kernel_size=3, stride=1),
            nn.Flatten(),
        )

        self.net = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(1),
        )

    def forward(self, o: torch.Tensor):
        h = self.feature_extract(o)
        v = self.net(h)
        return {"v": v}


actor = Actor()
critic = Critic()

ppo = PPO(actor, critic, entropy_coef=0.01, mse_coef=0.5)

buffer = RolloutBuffer(
    2048,
    {
        "obs": {
            "shape": (3, 96, 96),
            "map": "o",
        },
        "value": {"shape": (1,), "map": "v"},
        "action": {"shape": (3,), "map": "a"},
        "reward": {"shape": (1,)},
        "done": {"shape": (1,)},
        "returns": {"shape": (1,), "map": "r"},
        "advantages": {"shape": (1,), "map": "A"},
    },
    "cpu",
    1,
)

logger = Logger("cartpole_v1_ppo_discrete_trainer", log_types=["print"])

trainer = OnPolicyTrainer(env, buffer, ppo, "cpu", logger=logger)
trainer.train(1000)
