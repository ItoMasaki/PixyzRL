from __future__ import annotations

import argparse
import random

import gymnasium as gym
import numpy as np
import torch
from pixyz.distributions import Deterministic, Normal
from torch import nn

from pixyzrl.memory import RolloutBuffer
from pixyzrl.models import SAC


class Actor(Normal):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__(var=["a"], cond_var=["o"], name="pi")
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.loc_head = nn.Linear(hidden_dim, action_dim)
        self.log_scale_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, o: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(o)
        loc = self.loc_head(h)
        log_scale = torch.clamp(self.log_scale_head(h), -5.0, 2.0)
        return {"loc": loc, "scale": log_scale.exp()}


class Critic(Deterministic):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, name: str = "q") -> None:
        super().__init__(var=["q"], cond_var=["o", "a"], name=name)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, o: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"q": self.net(torch.cat([o, a], dim=-1))}


def run_training(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=args.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    actor = Actor(obs_dim, action_dim)
    critic1 = Critic(obs_dim, action_dim, name="q1")
    critic2 = Critic(obs_dim, action_dim, name="q2")

    sac = SAC(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_alpha=args.lr_alpha,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        action_var="a",
        obs_var="o",
        next_obs_var="o_next",
        reward_var="r",
        done_var="d",
        device=str(device),
    )

    replay = RolloutBuffer(
        buffer_size=args.buffer_size,
        env_dict={
            "obs": {"shape": (obs_dim,), "map": "o"},
            "next_obs": {"shape": (obs_dim,), "map": "o_next"},
            "action": {"shape": (action_dim,), "map": "a"},
            "reward": {"shape": (1,), "map": "r"},
            "done": {"shape": (1,), "map": "d"},
        },
        n_envs=1,
    )

    episode_rewards: list[float] = []
    episode_reward = 0.0

    for step in range(1, args.total_steps + 1):
        if step < args.warmup_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                sample = sac.select_action({"o": torch.tensor(obs, dtype=torch.float32).unsqueeze(0)})
            action = sample["a"].squeeze(0).cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay.add(
            obs=torch.tensor(obs, dtype=torch.float32),
            next_obs=torch.tensor(next_obs, dtype=torch.float32),
            action=torch.tensor(action, dtype=torch.float32),
            reward=torch.tensor([reward], dtype=torch.float32),
            done=torch.tensor([float(done)], dtype=torch.float32),
        )

        obs = next_obs
        episode_reward += reward

        if done:
            episode_rewards.append(episode_reward)
            moving = float(np.mean(episode_rewards[-10:]))
            print(f"step={step:6d} episode_reward={episode_reward:8.2f} moving10={moving:8.2f}")
            obs, _ = env.reset()
            episode_reward = 0.0

        if len(replay) >= args.batch_size and step % args.update_interval == 0:
            sac.train_step(replay, batch_size=args.batch_size, num_epochs=args.update_epochs)

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pendulum-v1 SAC example")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--update-interval", type=int, default=1)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--lr-alpha", type=float, default=3e-4)
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
