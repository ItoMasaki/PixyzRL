# PixyzRL: Reinforcement Learning with Pixyz

PixyzRL is a reinforcement learning (RL) library built upon the Pixyz library. It provides a modular implementation of Proximal Policy Optimization (PPO) and supports interactions with environments using Gymnasium.

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
        super().__init__(var=["a"], cond_var=["o"], name="actor")
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
        super().__init__(var=["v"], cond_var=["o"], name="critic")
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

### 3. Initialize PPO Agent

```python
from pixyzrl.models import PPO

ppo = PPO(actor, critic, None, eps_clip=0.2, lr_actor=3e-4, lr_critic=1e-3, device="cpu", entropy_coef=0.0, mse_coef=1.0)
```

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

In future work, we don't neet to write traning loop.

https://github.com/user-attachments/assets/d2851833-38f8-46f5-8b84-6ce1bbaeb62a

## Directory Structure

```
PixyzRL
├── docs
│   └── pixyz
│       └── README.pixyz.md
├── examples  # Example scripts
├── pixyzrl
│   ├── environments  # Environment wrappers
│   ├── models  # PPO and A2C implementations
│   ├── memory  # Experience replay & rollout buffer
│   ├── trainer  # Training management
│   ├── losses  # Loss function definitions
│   ├── logger  # Logging utilities
└── pyproject.toml
```


## License

PixyzRL is released under the MIT License.

## Author

Masaki Ito (ito.masaki@em.ci.ritsumei.ac.jp)

## Repository

[GitHub - ItoMasaki/PixyzRL](https://github.com/ItoMasaki/PixyzRL)

## Future Work

- [ ] Improve `Trainer` with additional optimization techniques
- [ ] Enhance `Logger` for better tracking and visualization
- [ ] Implement additional models:
  - [ ] Deep Q-Network (DQN)
  - [ ] Deep Deterministic Policy Gradient (DDPG)
  - [ ] Soft Actor-Critic (SAC)

## Community & Support

For more details, visit:
[PixyzRL ChatGPT Page](https://chatgpt.com/g/g-67b7c36695fc8191aca4cb7420dad17c-pixyzrl)
