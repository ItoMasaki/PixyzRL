import fcntl
import termios
import sys
import os
import numpy as np

import gymnasium as gym
import torch
from models import RecurrentStateSpaceModel, Actor
from utils import _images_to_observation, postprocess_observation
from matplotlib import pyplot as plt

belief_size = 200
state_size = 30
device = "cpu"
checkpoint_num = 131
total_rewards = [[0.0]]
img_total_rewards = [[0.0]]

rssm = RecurrentStateSpaceModel(state_size, belief_size, device)
rssm.load(f"weights/rssm_{checkpoint_num}.pt")

policy = Actor(3, "new").to(device)
policy.load_state_dict(torch.load(f"weights/actor_{checkpoint_num}.pt"))

env = gym.make("CarRacing-v3", render_mode="human")
observation, info = env.reset()
o_t = _images_to_observation(observation, 5, (64, 64))
h_t = torch.zeros(1, belief_size).to(device)
a_t = torch.zeros(1, 3).to(device)

# 1x3 plot
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plt.ion()

while True:
    with torch.no_grad():
        o_t = _images_to_observation(observation, 5, (64, 64))

        return_dict = rssm.predict(o_t.to(device), a_t, h_t, torch.tensor([[1.0]], device=device))

        a_t = policy.sample_action(h_t, return_dict["s_t"])
        action = a_t.detach().numpy()[0] * np.array([1.0, 0.5, 0.5]) + np.array([0.0, 0.5, 0.5])

        h_t = return_dict["h_tp1"]

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        prd_obs = postprocess_observation(return_dict["o_t"].detach().cpu().permute(0, 2, 3, 1).squeeze(0).numpy(), 5)
        total_rewards[-1].append(total_rewards[-1][-1] + reward)
        img_total_rewards[-1].append(img_total_rewards[-1][-1] + return_dict["r_t"].item())

        axs[0].cla()
        axs[0].imshow(prd_obs)
        axs[0].set_title("Predicted Observation")
        axs[1].cla()
        axs[1].plot(range(len(total_rewards[-1])), total_rewards[-1], color="blue")
        axs[1].plot(range(len(total_rewards[-1])), img_total_rewards[-1], color="orange")
        axs[1].set_title("Total Rewards")
        axs[2].cla()
        axs[2].bar([0, 1, 2], action)
        axs[2].set_title("Action Distribution")
        axs[2].set_xticks([0, 1, 2], ["Steering", "Acceleration", "Brake"])
        axs[2].set_ylim(-1.1, 1.1)

        plt.pause(0.001)

        if done:
            observation, info = env.reset()
            o_t = _images_to_observation(observation, 5, (64, 64))
            h_t = torch.zeros(1, belief_size).to(device)
            total_rewards.append([0.0])
            img_total_rewards.append([0.0])