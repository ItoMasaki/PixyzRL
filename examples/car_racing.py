from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from distributions import (
    Actor,
    Critic,
    Decoder,
    Encoder,
    Reward,
    Stochastic,
    Transition,
)
from matplotlib import pyplot as plt
from pixyz.losses import Expectation as E
from pixyz.losses import IterativeLoss, LogProb, MaxLoss, ValueLoss
from pixyz.losses import KullbackLeibler as KL
from pixyz.models import Model
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import (
    FreezeParameters,
    _images_to_observation,
    advantage_and_return,
    imagine_ahead,
)

from pixyzrl.losses import ppo
from pixyzrl.memory import ExperienceReplay

action_size = 3
batch_size = 50
batch_length = 50
belief_size = 200
state_size = 30
planning_horizon = 15
max_episode = 1

memory = ExperienceReplay(batch_size, batch_length, 100)


####################
# Model definition #
####################
encoder = Encoder(state_size, belief_size).to("mps")
decoder = Decoder(belief_size, state_size).to("mps")
reward_decoder = Reward(belief_size, state_size).to("mps")
transition = Transition(state_size, action_size, belief_size).to("mps")
stochastic = Stochastic(belief_size, state_size).to("mps")
model_modules = [encoder, decoder, reward_decoder, transition, stochastic]

step_loss = E(transition, MaxLoss(KL(encoder, stochastic), ValueLoss(3.0)) - E(encoder, LogProb(decoder)) - E(encoder, LogProb(reward_decoder)))
iter_loss = IterativeLoss(step_loss=step_loss, series_var=["o_t", "a_t", "t_t", "r_t"], update_value={"z_tp1": "z_t"}).mean()

model = Model(iter_loss, distributions=[encoder, decoder, transition, stochastic, reward_decoder], optimizer=torch.optim.Adam, optimizer_params={"lr": 1e-4}, clip_grad_norm=100.0)


####################
# Actor            #
####################
policy = Actor(action_size, belief_size, state_size, "new").to("mps")
policy_old = Actor(action_size, belief_size, state_size, "old").to("mps")

ppo_loss = ppo(policy, policy_old, 0.1).mean()

actor_model = Model(ppo_loss, distributions=[policy, policy_old], optimizer=optim.Adam, optimizer_params={"lr": 8e-5})


####################
# Critic           #
####################
critic = Critic(belief_size, state_size).to("mps")


optim_policy = optim.Adam(policy.parameters(), lr=8e-5)
optim_critic = optim.Adam(critic.parameters(), lr=8e-5)

max_step = 1000

total_rewards = []
total_pred_rewards = []
losses = []
actor_losses = []
value_losses = []

try:
    model.load("rssm.model.pt")
    policy.load_state_dict(torch.load("policy.model.pt"))
    critic.load_state_dict(torch.load("critic.model.pt"))
except (FileNotFoundError, RuntimeError):
    pass

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

for i in range(1, 1100001):
    env = gym.make("CarRacing-v2")

    with torch.no_grad():
        for episode in range(max_episode):
            observation, info = env.reset()

            z_t = torch.zeros(1, belief_size).to("mps")
            # s_t = encoder.sample({"o_t": torch.tensor(observation.transpose(2, 0, 1).reshape(1, 3, 96, 96) / 255, dtype=torch.float32).to("mps"), "z_t": z_t})["s_t"]
            s_t = torch.zeros(1, state_size).to("mps")

            action = np.zeros(3)
            total_rewards.append(0.0)
            total_pred_rewards.append(0.0)

            with tqdm(range(max_step)) as pbar:
                for step in pbar:
                    action = policy_old.sample({"z_t": z_t, "s_t": s_t})["a_t"].detach().cpu().numpy().astype(np.float64)[0]
                    action[0] = np.clip(action[0], -1, 1)
                    action[1] = np.clip(action[1], 0, 1)
                    action[2] = np.clip(action[2], 0, 1)
                    observation, reward, terminated, truncated, info = env.step(action)

                    if terminated or truncated:
                        pbar.close()
                        break

                    obs = _images_to_observation(observation, 8).to(device="mps")
                    memory.add(obs, action, reward, True)

                    act = torch.tensor(action.reshape(1, 3), dtype=torch.float32).to("mps")

                    s_t = encoder.sample({"o_t": obs, "z_t": z_t})["s_t"]
                    z_tp1 = transition.sample({"z_t": z_t, "s_t": s_t, "a_t": act, "t_t": torch.tensor([1.0], dtype=torch.float32).to("mps")})["z_tp1"]
                    o_t = decoder.sample_mean({"z_t": z_t, "s_t": s_t}).detach().cpu().numpy().reshape(3, 96, 96).transpose(1, 2, 0)
                    r_t = reward_decoder.sample_mean({"z_t": z_t, "s_t": s_t}).detach().cpu().item()
                    z_t = z_tp1.detach()

                    total_rewards[-1] += reward
                    total_pred_rewards[-1] += r_t

                    ax1.clear()
                    ax2.clear()
                    ax1.imshow(observation)
                    ax2.imshow(o_t)
                    ax1.axis("off")
                    ax2.axis("off")
                    fig.suptitle(f"{i}, Episode: {episode + 1}, Step: {step + 1}, Reward: {round(total_rewards[-1], 1)}, Predicted Reward: {round(total_pred_rewards[-1], 1)}")
                    plt.pause(0.0001)
                    pbar.set_postfix_str(f"Reward: {round(total_rewards[-1], 1)}")

            memory.add(observation / 255, action, reward, False)
    env.close()

    dataloader = DataLoader(memory, batch_size=batch_size, shuffle=False)

    z_t = torch.zeros(batch_size, belief_size).to("mps")
    s_t = torch.zeros(batch_size, state_size).to("mps")

    total_loss = 0.0
    losses.append(0.0)

    with tqdm(range(100)) as pbar:
        for epoch in pbar:
            observations, actions, rewards, terminates = next(iter(dataloader))

            obs = observations.transpose(1, 0).to(dtype=torch.float32, device="mps")
            act = actions.transpose(1, 0).to(dtype=torch.float32, device="mps")
            rwd = rewards.transpose(1, 0).to(dtype=torch.float32, device="mps")
            trm = terminates.transpose(1, 0).to(dtype=torch.float32, device="mps")

            loss, return_dict = model.train({"o_t": obs, "a_t": act, "z_t": z_t, "s_t": s_t, "t_t": trm, "r_t": rwd}, return_dict=True)

            total_loss += loss.detach().cpu().item() / batch_size
            losses[-1] = total_loss / (epoch + 1)
            z_t = return_dict["z_tp1"].detach()
            s_t = return_dict["s_t"].detach()

            with torch.no_grad():
                actor_states = s_t.detach()
                actor_beliefs = z_t.detach()

            with FreezeParameters(model_modules):
                imagination_traj = imagine_ahead(actor_states, actor_beliefs, policy_old, transition, stochastic, planning_horizon=12)

            imged_beliefs, imged_prior_states, imged_actions = imagination_traj

            with FreezeParameters([*model_modules, critic]):
                imged_reward = reward_decoder.sample_mean({"z_t": imged_beliefs.reshape(-1, belief_size), "s_t": imged_prior_states.reshape(-1, state_size)}).reshape(-1, batch_size, 1)
                value_pred = critic.sample_mean({"z_t": imged_beliefs.reshape(-1, belief_size), "s_t": imged_prior_states.reshape(-1, state_size)}).reshape(-1, batch_size, 1)

            adv, returns = advantage_and_return(imged_reward.detach(), value_pred.detach(), 0.99, 0.95)

            # Advantage estimation
            # ppo_loss = actor_model.train({"z_t": actor_beliefs, "s_t": actor_states, "a_t": imged_actions, "A": adv.detach()}, retain_graph=True)
            ppo_loss, _ = ppo(policy, policy_old, 0.1).mean()({"z_t": actor_beliefs, "s_t": actor_states, "a_t": imged_actions, "A": adv.detach()})
            # actor_entropy, _ = Entropy(policy).mean()({"z_t": actor_beliefs, "s_t": actor_states, "a_t": imged_actions})
            # ppo_loss = ppo_loss - 0.01 * actor_entropy

            optim_policy.zero_grad()
            ppo_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 100.0, norm_type=2)
            optim_policy.step()

            with torch.no_grad():
                value_beliefs = imged_beliefs.detach()
                value_prior_states = imged_prior_states.detach()
                target_return = returns.detach()

            value_loss = -critic.get_log_prob({"z_t": value_beliefs, "s_t": value_prior_states, "v_t": target_return}, sum_features=False)
            value_loss = 0.5 * value_loss.mean(dim=(0, 1))
            # Update model parameters
            optim_critic.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 100.0, norm_type=2)
            optim_critic.step()

            pbar.set_postfix_str(f"RSSM: {round(loss.detach().cpu().item() / batch_size, 2)} | PPO: {round(ppo_loss.detach().cpu().item(), 2)} | Value: {round(value_loss.detach().cpu().item(), 2)}")

            # actor_losses.append(actor_loss.item())
            # value_losses.append(value_loss.item())

    policy_old.load_state_dict(policy.state_dict())
    model.save("rssm.model.pt")
    torch.save(policy_old.state_dict(), "policy.model.pt")
    torch.save(critic.state_dict(), "critic.model.pt")
