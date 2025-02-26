# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from pixyz.distributions import Deterministic, Normal

# from pixyzrl.environments.env import Env
# from pixyzrl.memory import RolloutBuffer
# from pixyzrl.on_policy.ppo import PPO

# # 1. 環境の作成
# env = Env("CartPole-v1")
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n


# # 2. ActorとCriticの定義
# class Actor(Normal):
#     def __init__(self, state_dim, action_dim):
#         super().__init__(var=["a"], cond_var=["s"], name="actor")
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc_loc = nn.Linear(64, action_dim)
#         self.fc_scale = nn.Linear(64, action_dim)

#     def forward(self, s):
#         h = F.relu(self.fc1(s))
#         h = F.relu(self.fc2(h))
#         loc = self.fc_loc(h)
#         scale = F.softplus(self.fc_scale(h)) + 1e-6
#         return {"loc": loc, "scale": scale}


# class Critic(Deterministic):
#     def __init__(self, state_dim):
#         super().__init__(var=["v"], cond_var=["s"], name="critic")
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc_out = nn.Linear(64, 1)

#     def forward(self, s):
#         h = F.relu(self.fc1(s))
#         h = F.relu(self.fc2(h))
#         return {"v": self.fc_out(h)}


# actor = Actor(state_dim, action_dim)
# critic = Critic(state_dim)

# # 3. PPOエージェントの作成
# agent = PPO(actor, critic, shared_net=None, gamma=0.99, eps_clip=0.2, k_epochs=4, lr_actor=3e-4, lr_critic=1e-3, device="cpu")

# # 4. RolloutBufferの作成
# buffer = RolloutBuffer(buffer_size=1000, env_dict={"obs": {"shape": (state_dim,)}, "action": {"shape": (1,)}, "reward": {"shape": (1,)}, "done": {"shape": (1,)}}, key_mapping=None, device="cpu", n_envs=1)

# # 5. 経験の収集
# obs, _ = env.reset()
# for _ in range(100):  # 100ステップ分のサンプルを収集
#     action_dict = agent.select_action(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
#     action = torch.argmax(action_dict["a"]).detach().cpu().numpy()
#     next_obs, reward, truncated, terminated, _ = env.step(action)
#     done = truncated or terminated
#     buffer.add(obs=obs, action=action, reward=reward, done=done)
#     obs = next_obs if not done else env.reset()[0]

# # 6. GAEの計算
# returns_advantages = buffer.compute_returns_and_advantages_gae(last_state=torch.tensor(obs, dtype=torch.float32).unsqueeze(0), gamma=0.99, lmbd=0.95, critic=critic)

# print("GAE 計算結果:")
# print(returns_advantages)
