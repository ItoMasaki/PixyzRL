import torch
import torch.nn as nn
import torch.nn.functional as F
from pixyz.distributions import Categorical, Deterministic

from pixyzrl.environments import Env
from pixyzrl.memory import Memory
from pixyzrl.policy_gradient.ppo import PPO

# 環境のセットアップ
env = Env("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


def create_actor(state_dim: int, action_dim: int) -> Categorical:
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

    return Actor()


def create_critic(state_dim: int) -> Deterministic:
    class Critic(Deterministic):
        def __init__(self):
            super().__init__(var=["v"], cond_var=["o"], name="critic")
            self.fc1 = nn.Linear(state_dim, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc_out = nn.Linear(64, 1)

        def forward(self, o):
            h = F.relu(self.fc1(o))
            h = F.relu(self.fc2(h))
            return {"v": self.fc_out(h)}

    return Critic()


def compute_returns(rewards, dones, gamma=0.99):
    returns = []
    discounted_sum = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            discounted_sum = 0
        discounted_sum = reward + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    return torch.tensor(returns, dtype=torch.float32)


def compute_advantage(returns, values):
    return returns - values.squeeze()


# ネットワークの初期化
actor = create_actor(state_dim, action_dim)
critic = create_critic(state_dim)

# PPOエージェントの作成
agent = PPO(
    actor=actor,
    critic=critic,
    shared_net=None,
    gamma=0.99,
    eps_clip=0.2,
    k_epochs=4,
    lr_actor=3e-4,
    lr_critic=1e-3,
    mse_coef=1.0,
    entropy_coef=0.0,
    device="cpu",
)

# メモリバッファ
memory = Memory(buffer_size=500, batch_size=500, key_mapping={"obs": "o", "next_obs": "o_n", "action": "a", "reward": "r", "done": "t"})

# 学習ループ
for episode in range(100000):
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.select_action(torch.tensor(state, dtype=torch.float32))
        next_state, reward, truncated, terminated, _ = env.step(action)
        done = truncated or terminated
        memory.add(o=state, a=action["a"], r=reward, o_n=next_state, t=done)
        state = next_state

    if len(memory) >= memory.buffer_size:
        print(f"Episode {episode + 1}/{1000}")
        for _ in range(80):
            batch = memory.sample()
            batch["r"] = compute_returns(batch["r"], batch["t"])
            batch["v"] = critic(torch.tensor(batch["o"], dtype=torch.float32))["v"].detach()
            batch["A"] = compute_advantage(batch["r"], batch["v"])
            loss = agent.train(batch)
            print(f"Loss: {loss}")

        memory.clear()

# モデルの保存
agent.save_model("ppo_cartpole.pth")
