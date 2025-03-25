import torch
from pixyz.distributions import Deterministic, Normal
from torch import nn

from pixyzrl.environments import Env
from pixyzrl.logger import Logger
from pixyzrl.memory import RolloutBuffer
from pixyzrl.models import PPO
from pixyzrl.trainer import OnPolicyTrainer
from pixyzrl.utils import print_latex

env = Env("BipedalWalker-v3", 8, render_mode="rgb_array")
action_dim = env.action_space
obs_dim = env.observation_space


class Actor(Normal):
    def __init__(self):
        super().__init__(var=["a"], cond_var=["o"], name="p")

        self.feature_extract = nn.Sequential(
            nn.Linear(*obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        self._loc = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )

        self._scale = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softplus(),
        )

    def forward(self, o: torch.Tensor):
        h = self.feature_extract(o)
        loc = self._loc(h)
        scale = self._scale(h) + 1e-5
        return {"loc": loc, "scale": scale}


class Critic(Deterministic):
    def __init__(self):
        super().__init__(var=["v"], cond_var=["o"], name="f")

        self.net = nn.Sequential(
            nn.Linear(*obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, o: torch.Tensor):
        v = self.net(o)
        return {"v": v}


actor = Actor()
critic = Critic()

ppo = PPO(
    actor,
    critic,
    entropy_coef=0.01,
    mse_coef=1.0,
    lr_actor=1e-4,
    lr_critic=3e-4,
    device="mps",
    clip_grad_norm=0.5,
)
print_latex(ppo)

buffer = RolloutBuffer(
    8192,
    {
        "obs": {
            "shape": (*obs_dim,),
            "map": "o",
        },
        "value": {
            "shape": (1,),
            "map": "v",
        },
        "action": {
            "shape": (action_dim,),
            "map": "a",
        },
        "reward": {
            "shape": (1,),
            "transform": lambda x: torch.clamp(x, -10, 10),
        },
        "done": {
            "shape": (1,),
        },
        "returns": {
            "shape": (1,),
            "map": "r",
        },
        "advantages": {
            "shape": (1,),
            "map": "A",
        },
    },
    8,
    advantage_normalization=True,
    gamma=0.99,
    lam=0.95,
)

logger = Logger("bipedal_walker_v3_ppo_continual", log_types=["print"])

trainer = OnPolicyTrainer(env, buffer, ppo, "gae", "mps", logger=logger)
# trainer.load_model("cartpole_v1_ppo_discrete_trainer/model_1200.pt")
trainer.train(1000000, 1024, 10, save_interval=50, test_interval=10)
