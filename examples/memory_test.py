import re

import torch
from pixyz.distributions import Categorical, Deterministic
from torch import le, nn
from torch.nn import functional as F

from pixyzrl.environments import Env
from pixyzrl.memory import BaseBuffer, RolloutBuffer

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
        return {"probs": F.softmax(self.fc_logits(h), dim=-1)}


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

buffer = RolloutBuffer(2048, {"obs": {"shape": (4,)}, "action": {"shape": (1,)}, "reward": {"shape": (1,)}, "done": {"shape": (1,)}}, {"obs": "o", "action": "a", "reward": "reward", "done": "d", "returns": "r", "advantages": "A"}, "cpu", 1)

obs, info = env.reset()

for _ in range(20):
    while True:
        action = torch.argmax(actor.sample({"o": obs})["a"])
        next_obs, reward, trancated, terminated, _ = env.step(action)
        done = trancated or terminated
        print(f"obs.shape: {obs.shape}, action.shape: {action.shape}, reward.shape: {reward.shape}, done.shape: {done.shape}")
        buffer.add(obs=obs, action=action, reward=reward, done=done)
        obs = next_obs
        print(f"buffer.size: {len(buffer)}")

        if done:
            obs, info = env.reset()
            returns_and_advantages_gae = buffer.compute_returns_and_advantages_gae(next_obs, 0.99, 0.95, critic)
            break

print(buffer.sample(64))
