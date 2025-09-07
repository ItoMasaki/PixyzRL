from __future__ import annotations

import warnings
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from memory import ExperienceReplay
from models import (
    Actor,
    Critic,
    RecurrentStateSpaceModel,
)
from pixyz.losses import LogProb, Parameter
from pixyz.models import Model
from tensorboardX import SummaryWriter
from torch import optim
from tqdm import tqdm
from utils import (
    FreezeParameters,
    _images_to_observation,
    imagine_ahead,
    lambda_return,
    postprocess_observation,
)

action_size = 3
batch_size = 50
batch_length = 50
belief_size = 200
state_size = 30
planning_horizon = 15
max_episode = 1
free_nats = 1.0
train_steps = 100

max_episodes = 100
prefill = 50

# データ作成
np.random.seed(0)
device = "cpu"  # cpu

memory = ExperienceReplay(
    size=1000 * 1000,
    symbolic_env=False,
    observation_size=(3, 64, 64),
    action_size=action_size,
    bit_depth=5,
    device=device,
)

# TensorBoardのセットアップ
writer = SummaryWriter(f"runs/dreamer/{datetime.now().strftime('%Y%m%d-%H%M%S')}")

####################
# Model definition #
####################
rssm = RecurrentStateSpaceModel(state_size, belief_size, device)

print("RSSM Models")
print(rssm)


####################
# Critic           #
####################
critic = Critic().to(device)
critic_loss = -LogProb(critic).mean()
critic_model = Model(
    critic_loss,
    distributions=[critic],
    optimizer=optim.Adam,
    optimizer_params={"lr": 8e-5},
    clip_grad_norm=100.0,
)

####################
# Actor            #
####################
policy = Actor(action_size, "new").to(device)
policy_loss = -Parameter("r").mean()
policy_model = Model(
    policy_loss,
    distributions=[policy],
    optimizer=optim.Adam,
    optimizer_params={"lr": 8e-5},
    clip_grad_norm=100.0,
)

max_step = 1000

total_rewards = []
total_pred_rewards = []
losses = []
actor_losses = []
value_losses = []
frames = []
rssm_losses = []
actor_losses = []
value_losses = []


best_reward = -100000
env = gym.make("CarRacing-v3")

C = 3  # チャネル数
H = 64  # 高さ
W = 64  # 幅

for i in range(10000000):
    with torch.no_grad():
        h_ts = []
        s_ts = []
        imgs = []
        recs = []
        rwds = []

        done = False
        step = 0
        total_reward = 0.0

        # Test phase: collect data

        for ep in range(10):
            observation, info = env.reset()
            total_rewards.append(0.0)
            total_pred_rewards.append(0.0)
            h_t = torch.zeros(1, belief_size).to(device)
            a_t = torch.zeros(1, action_size).to(device)

            with tqdm(range(max_step)) as pbar:
                for step in pbar:
                    # 4フレームをチャネル方向に連結
                    observation = _images_to_observation(observation, 5, (64, 64)).to(device)

                    return_dict = rssm.predict(observation, a_t, h_t, torch.tensor([[1.0]], device=device))

                    # Policyから行動をサンプル
                    a_t = policy.sample_action(h_t, return_dict["s_t"])

                    action = a_t.squeeze(0).detach().cpu().numpy() * np.array([1.0, 0.5, 0.5]) + np.array([0.0, 0.5, 0.5])

                    # 環境へアクション入力
                    next_obs, reward, terminated, truncated, info = env.step(action)  # スケーリングを調整
                    done = terminated or truncated
                    e_t = 1.0 - torch.tensor([float(done)], device=device)

                    memory.append(
                        observation,  # torch.Tensor (C,H,W)
                        a_t,
                        reward,
                        done,
                    )

                    h_t = return_dict["h_tp1"]

                    observation = next_obs

                    total_rewards[-1] += reward
                    total_pred_rewards[-1] += return_dict["r_t"].detach().cpu().item()

                    pbar.set_postfix_str(
                        f"Discount: {round(return_dict['d_t'].item(), 3)} | Predicted reward: {round(total_pred_rewards[-1], 1)} | Reward: {round(total_rewards[-1], 1)}",
                    )

                    if ep % 10 == 0:
                        h_ts.append(return_dict["h_tp1"].detach().cpu().numpy())
                        s_ts.append(return_dict["s_t"].detach().cpu().numpy())
                        imgs.append(observation)
                        recs.append(postprocess_observation(return_dict["o_t"].detach().cpu().numpy(), 5))
                        rwds.append(return_dict["r_t"].detach().cpu().numpy())

                    if done:
                        break

                total_reward += total_rewards[-1]

    print(total_reward)
    # TensorBoardにログを記録
    writer.add_scalar("reward/actual", np.mean(total_rewards[-10:-1]), i)
    writer.add_scalar("reward/predicted", np.mean(total_pred_rewards[-10:-1]), i)

    if i % 10 == 0 or best_reward < total_rewards[-1]:
        best_reward = max(best_reward, total_rewards[-1])

    pbar = tqdm(range(100), desc="Training", total=100)

    for idx in pbar:
        obs, act, rwd, trm = memory.sample(batch_size, batch_length + 10)

        test_loss, return_dict = rssm.test({"o_t": obs[:10], "a_t": act[:10], "h_t": torch.zeros(batch_size, belief_size).to(device), "r_t": rwd[:10].reshape(-1, batch_size, 1), "e_t": trm[:10], "d_t": trm[:10] * 0.999})
        train_loss, return_dict = rssm.train({"o_t": obs[10:], "a_t": act[10:], "h_t": return_dict.get("h_t"), "r_t": rwd[10:].reshape(-1, batch_size, 1), "e_t": trm[10:], "d_t": trm[10:] * 0.999})

        with FreezeParameters(rssm.distributions):
            imged_beliefs, imged_prior_states, actions = imagine_ahead(return_dict.get("s_t").detach(), return_dict.get("h_t").detach(), policy, rssm.transition, rssm.stochastic, planning_horizon=50)

        with FreezeParameters([*rssm.distributions, critic]):
            imged_reward = rssm.reward_decoder.sample_mean({"h_t": imged_beliefs.reshape(-1, belief_size), "s_t": imged_prior_states.reshape(-1, state_size)}).reshape(-1, batch_size, 1)
            imged_discount = rssm.discount.sample_mean({"h_t": imged_beliefs.reshape(-1, belief_size), "s_t": imged_prior_states.reshape(-1, state_size)}).reshape(-1, batch_size, 1)
            imged_value = critic.sample_mean({"h_t": imged_beliefs.reshape(-1, belief_size), "s_t": imged_prior_states.reshape(-1, state_size)}).reshape(-1, batch_size, 1)

        returns = lambda_return(imged_reward[:-1], imged_value[:-1], bootstrap=imged_value[-1], discount=imged_discount[:-1], lambda_=0.95)
        actor_loss, _ = policy_model.train({"r": returns})

        # Dreamer implementation: value loss calculation and optimization
        # detach the input tensor from the transition network.
        value_loss, _ = critic_model.train({"h_t": imged_beliefs[1:].reshape(-1, belief_size).detach(), "s_t": imged_prior_states[1:].reshape(-1, state_size).detach(), "v_t": returns.reshape(-1, 1).detach()})

        pbar.set_postfix_str(f"{i} | RSSM: {round(train_loss, 4)} | Actor: {round(actor_loss, 4)} | Value: {round(value_loss, 4)} | Predicted reward: {round(total_pred_rewards[-1], 1)} | Reward: {round(total_rewards[-1], 1)} ({round(best_reward, 1)})")

    # TensorBoardにログを記録
    writer.add_scalar("loss/rssm", train_loss, i)
    writer.add_scalar("loss/actor", actor_loss, i)
    writer.add_scalar("loss/value", value_loss, i)

    rssm.save(f"weights/rssm_{i}.pt")
    torch.save(critic.state_dict(), f"weights/critic_{i}.pt")
    torch.save(policy.state_dict(), f"weights/actor_{i}.pt")
