import torch
from pixyz.distributions import Deterministic, Normal
from torch import nn

from pixyzrl.environments import Env
from pixyzrl.logger import Logger
from pixyzrl.memory import RolloutBuffer
from pixyzrl.models import PPO
from pixyzrl.trainer import OnPolicyTrainer
from pixyzrl.utils import print_latex

env = Env("LunarLander-v3", 10, render_mode="rgb_array", continuous=True)
action_dim = env.action_space
obs_dim = env.observation_space


class Actor(Normal):
    def __init__(self):
        super().__init__(var=["a"], cond_var=["o"], name="p")

        self.feature_extract = nn.Sequential(
            nn.LazyLinear(256),
            nn.SiLU(),
            nn.LazyLinear(256),
            nn.SiLU(),
        )

        self._loc = nn.Sequential(
            nn.LazyLinear(256),
            nn.Tanh(),
            nn.LazyLinear(action_dim),
            nn.Tanh(),
        )

        self._scale = nn.Sequential(
            nn.LazyLinear(256),
            nn.Tanh(),
            nn.LazyLinear(action_dim),
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
            nn.LazyLinear(256),
            nn.SiLU(),
            nn.LazyLinear(256),
            nn.SiLU(),
            nn.LazyLinear(256),
            nn.SiLU(),
            nn.LazyLinear(1),
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
    mse_coef=0.5,
    lr_actor=0.00003,
    lr_critic=0.0001,
    device="mps",
    # clip_grad_norm=0.5,
)
print_latex(ppo)

buffer = RolloutBuffer(
    10240,
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
    10,
    advantage_normalization=True,
    lam=0.95,
    gamma=0.99,
)

logger = Logger("lunar_lander_v3_ppo_continue_trainer", log_types=["print"])

trainer = OnPolicyTrainer(env, buffer, ppo, "gae", "mps", logger=logger)
# trainer.load_model("cartpole_v1_ppo_discrete_trainer/model_1200.pt")
trainer.train(1000000, 1024, 10, save_interval=50, test_interval=10)
