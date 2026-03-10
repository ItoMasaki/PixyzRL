from __future__ import annotations
from datetime import datetime
import torch
from torch import optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import gc

from pixyzrl.environments import Env
from pixyz.losses import LogProb, Parameter
from pixyz.models import Model
from memory import ExperienceReplay
from models import Actor, Critic, RecurrentStateSpaceModel
from utils import (
    FreezeParameters,
    imagine_ahead,
    lambda_return,
)
from pixyzrl.environments.env import ResizeObservation, BitDepthQuantize, ScaleAction




# ============================================
# Config
# ============================================
device = "cpu"
num_envs = 8
action_size = 3

belief_size = 200
state_size = 30

batch_size = 50
batch_length = 50
planning_horizon = 15

max_step = 1000
max_episodes = 200
train_steps = 100


# ============================================
# Environment
# ============================================
def build_env():
    return Env(
        "CarRacing-v3",
        num_envs=num_envs,
        enable_render=True,
        transforms=[
            ResizeObservation(
                width=64,
                height=64,
                grayscale=False,
                channel_first=True,
            ),
            BitDepthQuantize(
                bit_depth=5
            ),
            ScaleAction(
                from_low=[-1, -1,-1],
                from_high=[1, 1, 1],
                to_low=[-1, 0, 0],
                to_high=[1, 1, 1],
            )
        ],
    )


# ============================================
# Models
# ============================================
rssm = RecurrentStateSpaceModel(state_size, belief_size, device)

critic = Critic().to(device)
critic_loss = -LogProb(critic).mean()
critic_model = Model(
    critic_loss,
    distributions=[critic],
    optimizer=optim.Adam,
    optimizer_params={"lr": 8e-5},
    clip_grad_norm=100.0,
)

policy = Actor(action_size, "new").to(device)
policy_loss = -Parameter("r").mean()
# policy_loss = -(LogProb(policy) * Parameter("adv")).mean() - 0.01 * Entropy(policy).mean()
policy_model = Model(
    policy_loss,
    distributions=[policy],
    optimizer=optim.Adam,
    optimizer_params={"lr": 8e-5},
    clip_grad_norm=100.0,
)


# ============================================
# Replay
# ============================================
memory = ExperienceReplay(
    size=1_000_000,
    num_envs=num_envs,
    symbolic_env=False,
    observation_size=(3, 64, 64),
    action_size=action_size,
    bit_depth=5,
    device=device,
)


# ============================================
# Data Collection
# ============================================
def collect_data(env):
    obs, _ = env.reset()

    h_t = torch.zeros(num_envs, belief_size).to(device)
    a_t = torch.zeros(num_envs, action_size).to(device)
    prev_a_t = torch.zeros(num_envs, action_size).to(device)

    total_reward = torch.zeros(num_envs)

    p = tqdm(range(max_step), desc="Data Collection", total=max_step)
    for _ in p:

        with torch.no_grad():
            return_dict = rssm.predict(
                obs,
                a_t,
                h_t,
                torch.ones(num_envs, 1).to(device),
            )

            a_t = policy.sample_action(h_t, return_dict["s_t"])

        next_obs, reward, terminated, truncated, _ = env.step(a_t)

        # ⭐ VecEnv用 reset マスク
        reset_mask = (terminated | truncated).squeeze(-1)

        penalty = torch.sqrt((prev_a_t - a_t) ** 2).sum(dim=-1).unsqueeze(-1)

        # ⭐ Dreamer準拠：terminatedのみ保存
        memory.append(
            obs,
            a_t,
            reward - 0.01 * penalty,
            terminated | truncated,                       # termination signal
            (1.0 - terminated.float()) * 0.99         # discount target
        )

        # ⭐ RSSMの次状態
        h_t = return_dict["h_tp1"]

        # ⭐ VecEnvでは終了した環境の状態だけリセット
        h_t[reset_mask] = 0
        a_t[reset_mask] = 0
        prev_a_t[reset_mask] = 0

        obs = next_obs
        prev_a_t = a_t

        total_reward += reward.squeeze()

        p.set_postfix_str(
            f"Reward: {total_reward.mean().item():.2f}",
        )

    return total_reward.mean().item()



# ============================================
# World Model Training
# ============================================
def train_world_model():
    obs, act, rwd, trm, dis = memory.sample(batch_size, batch_length + 10)

    test_loss, return_dict = rssm.test({
        "o_t": obs[:10],
        "a_t": act[:10],
        "h_t": torch.zeros(batch_size, belief_size).to(device),
        "r_t": rwd[:10].reshape(-1, batch_size, 1),
        "e_t": trm[:10],
        "d_t": dis[:10].reshape(-1, batch_size, 1),
    })

    train_loss, return_dict = rssm.train({
        "o_t": obs[10:],
        "a_t": act[10:],
        "h_t": return_dict["h_t"],
        "r_t": rwd[10:].reshape(-1, batch_size, 1),
        "e_t": trm[10:],
        "d_t": dis[10:].reshape(-1, batch_size, 1),
    })

    return train_loss, return_dict


# ============================================
# Actor Critic Training
# ============================================
def train_actor_critic(return_dict):
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
    value_loss, _ = critic_model.train({
        "h_t": imged_beliefs[1:].reshape(-1, belief_size).detach(),
        "s_t": imged_prior_states[1:].reshape(-1, state_size).detach(),
        "v_t": returns.reshape(-1, 1).detach(),
    })

    return actor_loss, value_loss


# ============================================
# Main Loop
# ============================================
def main():

    writer = SummaryWriter(
        f"runs/dreamer/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    env = build_env()

    best_reward = -1e9

    for episode in range(max_episodes):

        reward = collect_data(env)

        writer.add_scalar("reward/actual", reward, episode)

        p = tqdm(range(train_steps), desc="Training", total=train_steps)
        for _ in p:

            rssm_loss, return_dict = train_world_model()
            actor_loss, value_loss = train_actor_critic(return_dict)

            p.set_postfix_str(
                f"RSSM: {rssm_loss:.4f} | Actor: {actor_loss:.4f} | Value: {value_loss:.4f}"
            )

        writer.add_scalar("loss/rssm", rssm_loss, episode)
        writer.add_scalar("loss/actor", actor_loss, episode)
        writer.add_scalar("loss/value", value_loss, episode)

        best_reward = max(best_reward, reward)

        print(
            f"[Episode {episode}] "
            f"Reward: {reward:.2f} "
            f"Best: {best_reward:.2f} "
            f"RSSM: {rssm_loss:.4f} "
            f"Actor: {actor_loss:.4f} "
            f"Value: {value_loss:.4f}"
        )

        rssm.save("weights/rssm.pt")
        torch.save(policy.state_dict(), "weights/actor.pt")
        torch.save(critic.state_dict(), "weights/critic.pt")

        del rssm_loss, actor_loss, value_loss, return_dict
        gc.collect()

    env.close()


if __name__ == "__main__":
    main()