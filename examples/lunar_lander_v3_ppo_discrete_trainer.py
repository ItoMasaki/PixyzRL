import torch
from pixyz.distributions import Categorical, Deterministic
from torch import nn

from pixyzrl.environments import Env
from pixyzrl.logger import Logger
from pixyzrl.memory import RolloutBuffer
from pixyzrl.models import PPO
from pixyzrl.trainer import OnPolicyTrainer
from pixyzrl.utils import print_latex

env = Env("LunarLander-v3", 2, render_mode="rgb_array")
action_dim = env.action_space
obs_dim = env.observation_space


class FeatureExtractor(Deterministic):
    def __init__(self):
        super().__init__(var=["s"], cond_var=["o"])

        self._net = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
        )

    def forward(self, o: torch.Tensor):
        return {"s": self._net(o)}


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
critic = Critic()
extractor = FeatureExtractor()

ppo = PPO(
    actor,
    critic,
    extractor,
    entropy_coef=0.01,
    mse_coef=0.5,
    lr_actor=1e-4,
    lr_critic=3e-4,
    device="mps",
    # clip_grad_norm=0.5,
)
print_latex(ppo)

buffer = RolloutBuffer(
    1024,
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
    2,
    advantage_normalization=True,
    lam=0.95,
    gamma=0.99,
)

logger = Logger("lunar_lander_v3_ppo_discrete_trainer", log_types=["print"])

trainer = OnPolicyTrainer(env, buffer, ppo, "gae", "mps", logger=logger)
# trainer.load_model("cartpole_v1_ppo_discrete_trainer/model_1200.pt")
trainer.train(1000000, 32, 10, save_interval=50, test_interval=10)
