import re

import torch
from pixyz.distributions import Deterministic, Normal
from torch import nn

from pixyzrl.environments import Env
from pixyzrl.logger import Logger
from pixyzrl.memory import RolloutBuffer
from pixyzrl.models import PPO
from pixyzrl.trainer import OnPolicyTrainer
from pixyzrl.utils import print_latex

env = Env("CarRacing-v3", render_mode="rgb_array")
action_dim = env.action_space.shape[0]


class Extractor(Deterministic):
    def __init__(self):
        super().__init__(var=["s"], cond_var=["o"], name="f")

        self.feature_extract = nn.Sequential(
            nn.LazyConv2d(32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.LazyConv2d(64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.LazyConv2d(64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, o: torch.Tensor):
        return {"s": self.feature_extract(o)}


class Actor(Normal):
    def __init__(self):
        super().__init__(var=["a"], cond_var=["s"], name="p")

        self._loc = nn.Sequential(
            nn.LazyLinear(2048),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(action_dim),
            nn.Tanh(),
        )

        self._scale = nn.Sequential(
            nn.LazyLinear(2048),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(action_dim),
            nn.Softplus(),
        )

    def forward(self, s: torch.Tensor):
        loc = self._loc(s)
        scale = self._scale(s) + 1e-5
        return {"loc": loc, "scale": scale}


class Critic(Deterministic):
    def __init__(self):
        super().__init__(var=["v"], cond_var=["s"], name="f")

        self.net = nn.Sequential(
            nn.LazyLinear(2048),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(1),
            # nn.LayerNorm(1),
        )

    def forward(self, s: torch.Tensor):
        v = self.net(s)
        return {"v": v}


actor = Actor().to("mps")
critic = Critic().to("mps")
extractor = Extractor().to("mps")

ppo = PPO(actor, critic, extractor, entropy_coef=0.01, mse_coef=0.5, lr_actor=1e-4, lr_critic=3e-4)
print_latex(ppo)

buffer = RolloutBuffer(
    10240,
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
    "mps",
    1,
    reward_normalization=False,
    advantage_normalization=True,
)

logger = Logger("cartpole_v1_ppo_discrete_trainer", log_types=["print"])

trainer = OnPolicyTrainer(env, buffer, ppo, "mps", logger=logger)
trainer.train(1000, 1024, 10)
