import re
from math import e

import torch
from anyio import value
from pixyz.distributions import Categorical, Deterministic, Multinomial
from torch import nn
from torch.nn import functional as F

from pixyzrl.environments import Env
from pixyzrl.memory import RolloutBuffer
from pixyzrl.models import PPO

env = Env("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


class Actor(Categorical):
    def __init__(self):
        super().__init__(var=["a"], cond_var=["o"], name="actor")
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_logits = nn.Linear(64, action_dim)

    def forward(self, o):
        h = F.relu(self.fc1(o))
        h = F.relu(self.fc2(h))
        return {"probs": F.softmax(self.fc_logits(h) + 1e-6, dim=-1)}


class Critic(Deterministic):
    def __init__(self):
        super().__init__(var=["v"], cond_var=["o"], name="critic")
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_v = nn.Linear(64, 1)

    def forward(self, o):
        h = F.relu(self.fc1(o))
        h = F.relu(self.fc2(h))
        return {"v": self.fc_v(h)}


actor = Actor()
critic = Critic()

ppo = PPO(actor, critic, None, 0.2, 3e-4, 1e-3, "cpu", entropy_coef=0.0, mse_coef=1.0)

buffer = RolloutBuffer(512, {"obs": {"shape": (4,)}, "value": {"shape": (1,)}, "action": {"shape": (1,)}, "reward": {"shape": (1,)}, "done": {"shape": (1,)}}, {"obs": "o", "action": "a", "reward": "reward", "value": "v", "done": "d", "returns": "r", "advantages": "A"}, "cpu", 1)

obs, info = env.reset()

for _ in range(2000):
    obs, info = env.reset()
    total_reward = 0
    while len(buffer) < 512:
        sample = ppo.select_action({"o": obs.unsqueeze(0)})
        action = torch.argmax(sample["a"]).detach()
        value = sample["v"].detach()
        next_obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        print(f"action: {action}", total_reward, end="\r")
        buffer.add(obs=obs, action=action, value=value, reward=reward, done=done)
        obs = next_obs

        if done:
            obs, info = env.reset()
            total_reward = 0
            print()

    sample = ppo.select_action({"o": next_obs.unsqueeze(0)})
    value = sample["v"].detach()
    # buffer.compute_returns_and_advantages_gae(value, 0.99, 0.95)
    buffer.compute_returns_and_advantages_mc(0.99)

    for _ in range(4):
        loss = ppo.train(buffer.sample(128))
        print(f"loss: {loss}")

    buffer.clear()
    ppo.actor_old.load_state_dict(ppo.actor.state_dict())
