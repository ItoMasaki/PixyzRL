import torch
from pixyz.distributions import Normal

from pixyzrl.environments import Env
from pixyzrl.memory import ExperienceReplay
from pixyzrl.policy_gradient.ppo import PPO
from pixyzrl.trainer import Trainer

# 環境のセットアップ
env_name = "CarRacing-v3"
env = Env(env_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 観測空間と行動空間の取得
obs_shape = env.observation_space.shape
action_shape = env.action_space.shape

# メモリバッファの初期化
buffer_size = 5000
batch_size = 64
memory = ExperienceReplay(obs_shape, action_shape, buffer_size, batch_size, device)

# PPOエージェントの設定
gamma = 0.99
eps_clip = 0.2
k_epochs = 4
lr_actor = 3e-4
lr_critic = 1e-3

# ActorとCriticの定義
actor = Normal(loc="s", scale="s", var=["a"], cond_var=["s"], name="actor")
critic = Normal(loc="s", scale="s", var=["v"], cond_var=["s"], name="critic")

# PPOエージェントの作成
agent = PPO(actor, critic, None, gamma, eps_clip, k_epochs, lr_actor, lr_critic, device)

# トレーナーの作成
trainer = Trainer(env, memory, agent, device)

# 学習の実行
num_iterations = 1000
trainer.train(num_iterations)

# モデルの保存
trainer.save_model("ppo_carracing.pth")
