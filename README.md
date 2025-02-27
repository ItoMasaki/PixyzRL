# PixyzRL: Reinforcement Learning with Pixyz
<div style="grid">
  <a href="./LICENSE">
    <img src="http://img.shields.io/badge/license-MIT-blue.svg?style=flat">
  </a>
  <img src="https://img.shields.io/badge/pytorch-2.5.1-pytorch.svg?logo=pytorch&style=flat">
  <img src="https://img.shields.io/badge/python-3.10 | 3.11 | 3.12-pytorch.svg?style=flat">
</div>

PixyzRL is a reinforcement learning (RL) library built upon the [Pixyz](https://github.com/masa-su/pixyz/tree/main) library. It provides a modular implementation of Proximal Policy Optimization (PPO) and supports interactions with environments using Gymnasium.

## Features

- **Probabilistic Modeling with Pixyz**
- **Implementation of PPO Algorithm**
- **Environment Wrappers for Gymnasium**
- **Memory Management (Rollout Buffer)**
- **Logging and Training Utilities**

## Installation

PixyzRL requires Python 3.10 or higher. Install dependencies with:

```bash
pip install torch torchaudio torchvision pixyz gymnasium[box2d] torchrl
```

Alternatively, clone and install the repository:

```bash
git clone https://github.com/ItoMasaki/PixyzRL.git
cd PixyzRL
pip install -e .
```

## Getting Started

### 1. Setup Environment

Create a Gymnasium environment wrapper using `Env` class:

```python
from pixyzrl.environments import Env

env = Env("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
```

### 2. Define Actor and Critic Networks

Use Pixyz's `Categorical` and `Deterministic` distributions for the Actor and Critic networks:

```python
import torch
from pixyz.distributions import Categorical, Deterministic
from torch import nn

class Actor(Categorical):
    def __init__(self):
        super().__init__(var=["a"], cond_var=["o"], name="p")
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, o: torch.Tensor):
        return {"probs": self.net(o)}

class Critic(Deterministic):
    def __init__(self):
        super().__init__(var=["v"], cond_var=["o"], name="f")
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, o: torch.Tensor):
        return {"v": self.net(o)}

actor = Actor()
critic = Critic()
```

#### 2.1 Display distributions as `latex`

```
>>> pixyzrl.utils.print_latex(actor)
p(a|o)

>>> pixyzrl.utils.print_latex(critic)
f(v|o)
```

### 3. Initialize PPO Agent

```python
from pixyzrl.models import PPO

ppo = PPO(actor, critic, None, eps_clip=0.2, lr_actor=3e-4, lr_critic=1e-3, device="cpu", entropy_coef=0.0, mse_coef=1.0)
```

##### 3.1 Display model as `latex`

```
>>> pixyzrl.utils.print_latex(ppo)
mean \left(1.0 MSE(f(v|o), r) - min \left(A clip(\frac{p(a|o)}{old(a|o)}, 0.8, 1.2), A \frac{p(a|o)}{old(a|o)}\right) \right)
```

![TeXclip Feb 27 2025](https://github.com/user-attachments/assets/2669eacf-bbfa-4a88-a60a-bb0d8196d704)

### 4. Setup Rollout Buffer

```python
from pixyzrl.memory import RolloutBuffer

buffer = RolloutBuffer(
    2048,
    {"obs": {"shape": (4,)}, "value": {"shape": (1,)}, "action": {"shape": (2,)}, "reward": {"shape": (1,)}, "done": {"shape": (1,)}},
    {"obs": "o", "action": "a", "reward": "reward", "value": "v", "done": "d", "returns": "r", "advantages": "A"},
    "cpu", 1
)
```

### 5. Training Loop

```python
obs, info = env.reset()

for _ in range(2000):
    obs, info = env.reset()
    total_reward = 0
    while len(buffer) < 2048:
        sample = ppo.select_action({"o": obs.unsqueeze(0)})
        action, value = sample["a"].detach(), sample["v"].detach()
        next_obs, reward, done, _, _ = env.step(torch.argmax(action))
        total_reward += reward
        buffer.add(obs=obs.detach(), action=action.detach(), value=value.detach(), reward=reward.detach(), done=done.detach())
        obs = next_obs

        if done:
            obs, info = env.reset()
            total_reward = 0

    sample = ppo.select_action({"o": next_obs.unsqueeze(0)})
    value = sample["v"].detach()
    buffer.compute_returns_and_advantages_gae(value, 0.99, 0.95)

    for _ in range(40):
        batch = buffer.sample(128)
        loss = ppo.train(batch)
        print(f"loss: {loss}")

    buffer.clear()
    ppo.actor_old.load_state_dict(ppo.actor.state_dict())
```

In the future work, we don't need to write traning loop explicitly.

https://github.com/user-attachments/assets/fdf15f97-6fb9-4f12-8522-503eccb47fe5

## Directory Structure

```
PixyzRL
├── docs
│   └── pixyz
│       └── README.pixyz.md
├── examples  # Example scripts
├── pixyzrl
│   ├── environments  # Environment wrappers
│   ├── models
│   │   ├ on_policy  # On Policy models implementations
│   │   └ off_policy  # Off Policy models implementations
│   ├── memory  # Experience replay & rollout buffer
│   ├── trainer  # Training management
│   ├── losses  # Loss function definitions
│   ├── logger  # Logging utilities
│   └── utils.py
└── pyproject.toml
```

## License

PixyzRL is released under the MIT License.

## Author

- Masaki Ito ( l1sum [at] icloud.com )
- Daisuke Nakahara

## Repository

[GitHub - ItoMasaki/PixyzRL](https://github.com/ItoMasaki/PixyzRL)

## Future Work

- [ ] Improve `Trainer` with additional optimization techniques
- [ ] Enhance `Logger` for better tracking and visualization
- [ ] Implement model free algorithms:
  - [ ] Deep Q-Network (DQN)
  - [ ] Deep Deterministic Policy Gradient (DDPG)
  - [ ] Soft Actor-Critic (SAC)
- [ ] Implement model-based algorithms:
  - [ ] Dreamer
- [ ] Collaborate with ChatGPT (MyGPT) for building architectures by natural languages.
- [ ] Collaborate with [Genesis](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/what_is_genesis.html).

## Community & Support

For more details, visit:
[PixyzRL ChatGPT Page](https://chatgpt.com/g/g-67b7c36695fc8191aca4cb7420dad17c-pixyzrl)
