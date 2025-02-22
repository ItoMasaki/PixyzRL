import torch
from pixyz.distributions import Normal
from torch import nn

from pixyzrl.environments import Env
from pixyzrl.memory import ExperienceMemory
from pixyzrl.policy_gradient.ppo import PPO
from pixyzrl.trainer import Trainer

# 環境の作成
env = Env("CartPole-v1")


# アクターネットワークの作成
class Actor(Normal):
    def __init__(self):
        super().__init__(var=["a"], cond_var=["s"], name="actor")
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, s):
        h = torch.relu(self.fc1(s))
        return {"loc": self.fc2(h), "scale": torch.ones_like(self.fc2(h))}


class Critic(Normal):
    def __init__(self):
        super().__init__(var=["v"], cond_var=["s"], name="critic")
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, s):
        h = torch.relu(self.fc1(s))
        return {"loc": self.fc2(h), "scale": torch.ones_like(self.fc2(h))}


actor = Actor()
critic = Critic()

# PPOエージェントの作成
ppo_agent = PPO(actor=actor, critic=critic, shared_cnn=None, gamma=0.99, eps_clip=0.2, k_epochs=4, lr_actor=3e-4, lr_critic=1e-3, device="cpu")

# メモリの作成
memory = ExperienceMemory(obs_shape=(4,), action_shape=(1,), buffer_size=10000)

# トレーナーの作成
trainer = Trainer(env, memory, ppo_agent, device="cpu")

# 学習の実行
trainer.train(num_iterations=1000)

# モデルの保存
ppo_agent.save_model("ppo_cartpole.pth")

print("Training completed and model saved.")
