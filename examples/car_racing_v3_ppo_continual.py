import torch
from pixyz.distributions import Deterministic, Normal
from torch import nn

from pixyzrl.environments import Env
from pixyzrl.logger import Logger
from pixyzrl.memory import RolloutBuffer
from pixyzrl.models import PPO
from pixyzrl.trainer import OnPolicyTrainer
from pixyzrl.utils import print_latex

env = Env("CarRacing-v3", 8, render_mode="rgb_array")
action_dim = env.action_space
obs_dim = env.observation_space[::-1]


class Actor(Normal):
    def __init__(self, action_dim: int) -> None:
        super().__init__(var=["a"], cond_var=["o"], name="p")

        self.feature_extract = nn.Sequential(
            nn.LazyConv2d(32, kernel_size=8, stride=4),
            nn.Tanh(),
            nn.LazyConv2d(64, kernel_size=4, stride=2),
            nn.Tanh(),
            nn.LazyConv2d(64, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.Flatten(),
        )

        self._loc = nn.Sequential(
            nn.LazyLinear(1024),
            nn.Tanh(),
            nn.LazyLinear(action_dim),
            nn.Tanh(),
        )

        self._scale = nn.Sequential(
            nn.LazyLinear(2048),
            nn.Tanh(),
            nn.LazyLinear(action_dim),
            nn.Softplus(),
        )

    def forward(self, o: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.feature_extract(o)
        loc = self._loc(h)
        scale = self._scale(h) + 1e-5
        return {"loc": loc, "scale": scale}


class Critic(Deterministic):
    def __init__(self) -> None:
        super().__init__(var=["v"], cond_var=["o"], name="f")

        self.feature_extract = nn.Sequential(
            nn.LazyConv2d(32, kernel_size=8, stride=4),
            nn.Tanh(),
            nn.LazyConv2d(64, kernel_size=4, stride=2),
            nn.Tanh(),
            nn.LazyConv2d(64, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.Flatten(),
        )

        self.net = nn.Sequential(
            nn.LazyLinear(1024),
            nn.Tanh(),
            nn.LazyLinear(1),
        )

    def forward(self, o: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.feature_extract(o)
        v = self.net(h)
        return {"v": v}


actor = Actor(action_dim)
critic = Critic()

ppo = PPO(
    actor,
    critic,
    entropy_coef=0.01,
    mse_coef=0.5,
    lr_actor=1e-4,
    lr_critic=3e-4,
    device="mps",
)
print_latex(ppo)

buffer = RolloutBuffer(
    5120,
    {
        "obs": {
            "shape": (*obs_dim,),
            "map": "o",
        },
        "value": {"shape": (1,), "map": "v"},
        "action": {"shape": (action_dim,), "map": "a"},
        "reward": {"shape": (1,)},
        "done": {"shape": (1,)},
        "returns": {"shape": (1,), "map": "r"},
        "advantages": {"shape": (1,), "map": "A"},
    },
    8,
    advantage_normalization=True,
)

logger = Logger("car_racing_v3_ppo_continual", log_types=["print"])
trainer = OnPolicyTrainer(env, buffer, ppo, "gae", "mps", logger=logger)
# trainer.load_model("cartpole_v1_ppo_discrete_trainer/model_1200.pt")
trainer.train(1000000, 512, 10, save_interval=50, test_interval=20)
