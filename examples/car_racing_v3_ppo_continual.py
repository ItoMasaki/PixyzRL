import torch
from anyio import value
from pixyz.distributions import Categorical, Deterministic
from torch import nn

from pixyzrl.environments import Env
from pixyzrl.memory import RolloutBuffer
from pixyzrl.models import PPO
from pixyzrl.utils import print_latex

env = Env("CarRacing-v3")
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
        probs = self.net(o)
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
print_latex(actor)
print_latex(critic)


ppo = PPO(actor, critic, entropy_coef=0.0, mse_coef=1.0)
print_latex(ppo)

buffer = RolloutBuffer(2048, {"obs": {"shape": (4,)}, "value": {"shape": (1,)}, "action": {"shape": (2,)}, "reward": {"shape": (1,)}, "done": {"shape": (1,)}}, {"obs": "o", "action": "a", "reward": "reward", "value": "v", "done": "d", "returns": "r", "advantages": "A"}, "cpu", 1)

obs, info = env.reset()

for _ in range(2000):
    obs, info = env.reset()
    total_reward = 0
    while len(buffer) < 2048:
        sample = ppo.select_action({"o": obs.unsqueeze(0)})
        action = sample["a"].detach()
        value = sample["v"].detach()
        next_obs, reward, done, _, _ = env.step(torch.argmax(action))
        total_reward += reward
        print(f"action: {action}", total_reward, end="\r")
        buffer.add(obs=obs.detach(), action=action.detach(), value=value.detach(), reward=reward.detach(), done=done.detach())
        obs = next_obs

        if done:
            obs, info = env.reset()
            total_reward = 0
            print()

    sample = ppo.select_action({"o": next_obs.unsqueeze(0)})
    value = sample["v"].detach()
    buffer.compute_returns_and_advantages_gae(value, 0.99, 0.95)

    for _ in range(40):
        batch = buffer.sample(128)
        loss = ppo.train(batch)
        print(f"loss: {loss}")

    buffer.clear()
    ppo.actor_old.load_state_dict(ppo.actor.state_dict())
