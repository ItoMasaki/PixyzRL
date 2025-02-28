import numpy as np
import torch
from pixyz.distributions import Categorical, Deterministic
from torch import nn

from pixyzrl.environments import Env
from pixyzrl.losses import ppo
from pixyzrl.memory import RolloutBuffer
from pixyzrl.models import PPO
from pixyzrl.trainer import OnPolicyTrainer

# def test_env():
#     env = Env("CartPole-v1")
#     obs, info = env.reset()
#     assert obs.shape == (4,)

#     action = torch.Tensor(np.zeros(1))
#     print(type(action))
#     next_obs, reward, truncated, terminated, info = env.step(action.detach())
#     assert next_obs.shape == (4,)
#     assert reward.shape == (1,)


def test_actor_critic():
    env = Env("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    class Actor(Categorical):
        def __init__(self):
            super().__init__(var=["a"], cond_var=["o"], name="p")
            self.net = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, action_dim), nn.Softmax(dim=-1))

        def forward(self, o: torch.Tensor):
            return {"probs": self.net(o)}

    class Critic(Deterministic):
        def __init__(self):
            super().__init__(var=["v"], cond_var=["o"], name="f")
            self.net = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, o: torch.Tensor):
            return {"v": self.net(o)}

    actor = Actor()
    critic = Critic()

    obs = torch.randn(1, state_dim)
    action_out = actor.forward(obs)
    critic_out = critic.forward(obs)

    assert "probs" in action_out
    assert "v" in critic_out


def test_memory():
    buffer = RolloutBuffer(
        100,
        {
            "obs": {"shape": (4,), "map": "o"},
            "action": {"shape": (1,), "map": "a"},
            "reward": {"shape": (1,)},
            "done": {"shape": (1,)},
        },
        "cpu",
        1,
    )

    assert len(buffer) == 1
    buffer.add(obs=torch.randn(1, 4), action=torch.tensor([1]), reward=torch.tensor([1.0]), done=torch.tensor([0]))
    assert len(buffer) == 2


def test_loss():
    env = Env("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Categorical(var=["a"], cond_var=["o"], name="p")
    actor_old = Categorical(var=["a"], cond_var=["o"], name="old")
    loss = ppo(actor, actor_old)

    assert loss is not None


def test_trainer():
    env = Env("CartPole-v1")
    buffer = RolloutBuffer(
        100,
        {
            "obs": {"shape": (4,), "map": "o"},
            "action": {"shape": (2,), "map": "a"},
            "reward": {"shape": (1,)},
            "value": {"shape": (1,)},
            "returns": {"shape": (1,), "map": "r"},
            "advantages": {"shape": (1,), "map": "A"},
            "done": {"shape": (1,)},
        },
        "cpu",
        1,
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    class Actor(Categorical):
        def __init__(self):
            super().__init__(var=["a"], cond_var=["o"], name="p")
            self.net = nn.Sequential(
                nn.LazyLinear(64),
                nn.ReLU(),
                nn.LazyLinear(64),
                nn.ReLU(),
                nn.LazyLinear(action_dim),
                nn.Softmax(dim=-1),
            )

        def forward(self, o: torch.Tensor):
            return {"probs": self.net(o)}

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
            return {"v": self.net(o)}

    actor = Actor()
    critic = Critic()

    agent = PPO(actor, critic, device="cpu")
    trainer = OnPolicyTrainer(env, buffer, agent, "cpu")
    trainer.train(0)

    assert len(buffer) == 1
