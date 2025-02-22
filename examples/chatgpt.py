import numpy as np
import torch
from pixyz.distributions import Normal

from pixyzrl.environments import Env
from pixyzrl.memory import Memory
from pixyzrl.policy_gradient.ppo import PPO
from pixyzrl.trainer import Trainer


def main():
    # 環境の設定
    env = Env("CartPole-v1")
    obs_shape = env.observation_space.shape
    action_shape = (1,)  # CartPoleは離散アクション

    # メモリの設定
    memory = Memory(obs_shape, action_shape, buffer_size=10000)

    # PPOエージェントの作成
    actor = Normal(loc="s", scale="s", var=["a"], cond_var=["s"], name="actor")
    critic = Normal(loc="s", scale="s", var=["v"], cond_var=["s"], name="critic")
    ppo_agent = PPO(actor, critic, shared_cnn=None, gamma=0.99, eps_clip=0.2, k_epochs=4, lr_actor=3e-4, lr_critic=1e-3, device="cpu")

    # トレーナーの作成
    trainer = Trainer(env, memory, ppo_agent, device="cpu")

    # トレーニングの実行
    trainer.train(num_iterations=1000)

    # テストの実行
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action_dict = ppo_agent.select_action(torch.tensor(obs, dtype=torch.float32))
        action = action_dict["a"].detach().numpy()
        next_obs, reward, done, _ = env.step(action)
        memory.add(obs, action, reward, done)
        obs = next_obs
        total_reward += reward

    print(f"Total Reward: {total_reward}")


if __name__ == "__main__":
    main()
