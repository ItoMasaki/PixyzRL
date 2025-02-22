import torch
import torch.nn as nn
import torch.nn.functional as F
from pixyz.distributions import Deterministic, Normal

from pixyzrl.environments import Env
from pixyzrl.memory import Memory
from pixyzrl.policy_gradient.ppo import PPO
from pixyzrl.trainer import Trainer

# 環境のセットアップ
env = Env("CartPole-v1")

# 状態とアクションの次元
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] if len(env.action_space.shape) > 0 else 1


# Actor ネットワーク
class Actor(Normal):
    def __init__(self, state_dim, action_dim):
        super().__init__(var=["a"], cond_var=["s"], name="actor")
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_loc = nn.Linear(64, action_dim)  # 平均
        self.fc_scale = nn.Linear(64, action_dim)  # 標準偏差

    def forward(self, s):
        h = F.relu(self.fc1(s))
        h = F.relu(self.fc2(h))
        loc = self.fc_loc(h)
        scale = F.softplus(self.fc_scale(h)) + 1e-6  # 正の値にするために softplus を適用
        return {"loc": loc, "scale": scale}


# Critic ネットワーク
class Critic(Deterministic):
    def __init__(self, state_dim):
        super().__init__(var=["v"], cond_var=["s"], name="critic")
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, s):
        h = F.relu(self.fc1(s))
        h = F.relu(self.fc2(h))
        return {"v": self.fc_out(h)}


# Actor-Critic の初期化
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)

# PPO エージェントの作成
agent = PPO(
    actor=actor,
    critic=critic,
    shared_cnn=None,  # 画像入力などを考慮する場合はCNNを追加
    gamma=0.99,
    eps_clip=0.2,
    k_epochs=4,
    lr_actor=3e-4,
    lr_critic=1e-3,
    device="cpu",
)

# メモリバッファ
memory = Memory(obs_shape=(state_dim,), action_shape=(action_dim,), buffer_size=10000)

# トレーナーの作成
trainer = Trainer(env, memory, agent, device="cpu")

# 学習の実行
trainer.train(num_iterations=1000)

# モデルの保存
agent.save_model("ppo_model.pth")
