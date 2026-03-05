from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import sympy

import gymnasium as gym
import numpy as np
import torch
from pixyz.distributions import Deterministic
from pixyz.losses.losses import Loss
from pixyz.models import Model
from torch import nn


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(Deterministic):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128, name: str = "q"):
        super().__init__(var=["q"], cond_var=["s"], name=name)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"q": self.net(s)}


class DQNLoss(Loss):
    def __init__(self, q_net: QNetwork):
        super().__init__([*q_net.cond_var, "a", "target_q"])
        self.q_net = q_net
        self.mse = nn.MSELoss()

    @property
    def _symbol(self) -> sympy.Symbol:
        return sympy.Symbol("MSE(Q(s,a), target)")

    def forward(self, x_dict: dict[str, torch.Tensor], **kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        q_values = self.q_net.sample_mean({"s": x_dict["s"]})
        chosen_q = q_values.gather(1, x_dict["a"].long())
        loss = self.mse(chosen_q, x_dict["target_q"])
        return loss, {}


class DQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, lr: float, gamma: float, device: torch.device):
        self.gamma = gamma
        self.device = device
        self.action_dim = action_dim

        self.q_dist = QNetwork(obs_dim, action_dim, name="q_online").to(device)
        self.target_q_dist = QNetwork(obs_dim, action_dim, name="q_target").to(device)
        self.target_q_dist.load_state_dict(self.q_dist.state_dict())

        self.model = Model(
            loss=DQNLoss(self.q_dist).mean(),
            distributions=[self.q_dist],
            optimizer=torch.optim.Adam,
            optimizer_params={"lr": lr},
            clip_grad_norm=10.0,
        )

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.q_dist.sample_mean({"s": state_t})
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, transitions: list[Transition]) -> float:
        states = torch.as_tensor(np.stack([t.state for t in transitions]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([t.action for t in transitions], dtype=torch.long, device=self.device).unsqueeze(-1)
        rewards = torch.as_tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.as_tensor(np.stack([t.next_state for t in transitions]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([t.done for t in transitions], dtype=torch.float32, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            next_q = self.target_q_dist.sample_mean({"s": next_states})
            next_q_max = next_q.max(dim=1, keepdim=True).values
            target_q = rewards + self.gamma * (1.0 - dones) * next_q_max

        loss = self.model.train({"s": states, "a": actions, "target_q": target_q})
        return float(loss.detach())

    def sync_target(self) -> None:
        self.target_q_dist.load_state_dict(self.q_dist.state_dict())


def linear_schedule(step: int, start: float, end: float, duration: int) -> float:
    if step >= duration:
        return end
    ratio = step / duration
    return start + ratio * (end - start)


def run_training(args: argparse.Namespace) -> float:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")
    env.reset(seed=args.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(obs_dim=obs_dim, action_dim=action_dim, lr=args.lr, gamma=args.gamma, device=device)
    replay_buffer = ReplayBuffer(args.buffer_size)

    global_step = 0
    reward_history: list[float] = []

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            epsilon = linear_schedule(global_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
            action = agent.act(state, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.append(
                Transition(state=state, action=action, reward=reward, next_state=next_state, done=done),
            )

            state = next_state
            episode_reward += reward
            global_step += 1

            if len(replay_buffer) >= max(args.warmup_steps, args.batch_size) and global_step % args.train_interval == 0:
                batch = replay_buffer.sample(args.batch_size)
                agent.update(batch)

            if global_step % args.target_update_interval == 0:
                agent.sync_target()

        reward_history.append(episode_reward)
        mean_reward = float(np.mean(reward_history[-args.eval_window :]))
        print(f"Episode {episode:4d} | reward={episode_reward:6.1f} | mean{args.eval_window}={mean_reward:6.1f} | epsilon={epsilon:.3f}")

    env.close()
    final_mean = float(np.mean(reward_history[-args.eval_window :]))
    print(f"Final mean reward ({args.eval_window} eps): {final_mean:.2f}")
    return final_mean


def main() -> None:
    parser = argparse.ArgumentParser(description="CartPole-v1 DQN example (Pixyz-based)")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--target-update-interval", type=int, default=250)
    parser.add_argument("--train-interval", type=int, default=1)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-window", type=int, default=20)
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
