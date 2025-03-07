from venv import logger

import torch
from pixyz.distributions import Categorical, Deterministic
from torch import nn

from pixyzrl.environments import Env
from pixyzrl.logger import Logger
from pixyzrl.memory import RolloutBuffer
from pixyzrl.models import PPO
from pixyzrl.trainer import OnPolicyTrainer

env = Env("CartPole-v1", 8, render_mode="rgb_array")
action_dim = env.action_space[0].n


class Actor(Categorical):
    def __init__(self):
        super().__init__(var=["a"], cond_var=["o"], name="p")

        self.net = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(action_dim),
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, o: torch.Tensor):
        probs = self.net(o)
        probs = self.softmax(probs)
        return {"probs": probs}


class Critic(Deterministic):
    def __init__(self):
        super().__init__(var=["v"], cond_var=["o"], name="f")

        self.net = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(1),
        )

    def forward(self, o: torch.Tensor):
        v = self.net(o)
        return {"v": v}


actor = Actor()
critic = Critic()

ppo = PPO(actor, critic, entropy_coef=0.0, mse_coef=1.0)

buffer = RolloutBuffer(
    2048,
    {
        "obs": {"shape": (4,), "map": "o"},
        "value": {"shape": (1,), "map": "v"},
        "action": {"shape": (2,), "map": "a"},
        "reward": {"shape": (1,)},
        "done": {"shape": (1,)},
        "returns": {"shape": (1,), "map": "r"},
        "advantages": {"shape": (1,), "map": "A"},
    },
    "cpu",
    8,
)

logger = Logger("logs", log_types=["print"])
trainer = OnPolicyTrainer(env, buffer, ppo, "gae", "cpu", logger=logger)
trainer.train(10000, 256, 10)
