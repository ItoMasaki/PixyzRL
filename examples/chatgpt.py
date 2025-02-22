from pixyzrl.environments.env import Env
from pixyzrl.memory import Memory

# 環境の作成
env_name = "CartPole-v1"
env = Env(env_name)

# メモリの作成
obs_shape = env.observation_space.shape
action_shape = (1,)
buffer_size = 1000  # 記録する経験の最大数
memory = Memory(obs_shape, action_shape, buffer_size, device="cpu")

# エピソードのシミュレーション
episodes = 10
max_steps = 200

for episode in range(episodes):
    obs, _ = env.reset()
    done = False
    step = 0

    while not done and step < max_steps:
        action = env.action_space.sample()  # ランダムな行動を選択
        next_obs, reward, done, _, _ = env.step(action)

        # メモリにデータを保存
        memory.add(obs, action, reward, done)

        obs = next_obs
        step += 1

    print(f"Episode {episode + 1}: Steps = {step}")

env.close()

# メモリのサンプルデータを取得
sample = memory.sample()
print("Sample from Memory:", sample)
