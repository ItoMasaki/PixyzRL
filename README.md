# PixyzRL: A Reinforcement Learning Framework with Probabilistic Generative Models

![PixyzRL Logo](https://github.com/user-attachments/assets/577b9d4b-30d0-493d-95fc-b83a2f292c28)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.5.1-pytorch.svg?logo=pytorch&style=flat)](https://pytorch.org/)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
![workflow](https://github.com/ItoMasaki/PixyzRL/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/ItoMasaki/PixyzRL/branch/main/graph/badge.svg?token=V1704I8BQT)](https://codecov.io/gh/ItoMasaki/PixyzRL)
[![Open in Visual Studio Code](https://img.shields.io/static/v1?logo=visualstudiocode&label=&message=Open%20in%20Visual%20Studio%20Code&labelColor=2c2c32&color=007acc&logoColor=007acc)](https://vscode.dev/github/ItoMasaki/PixyzRL)

[Documentation](https://docs.pixyz.io) | [Examples](https://github.com/ItoMasaki/PixyzRL/tree/main/examples) | [GitHub](https://github.com/ItoMasaki/PixyzRL)

|bipedal-walker-v3|lunar-lander-v3|
|:-:|:-:|
|<video src="https://github.com/user-attachments/assets/72045515-8c56-4308-acd8-6aca48915024"/>|<video src="https://github.com/user-attachments/assets/aba4af68-1ded-4578-b463-529f0f6d0cde"/>|


## What is PixyzRL?

[**PixyzRL**](https://github.com/ItoMasaki/PixyzRL) is a reinforcement learning (RL) framework based on **probabilistic generative models** and **Bayesian theory**. Built on top of the [Pixyz](https://github.com/masa-su/pixyz) library, it provides a modular and flexible design to enable uncertainty-aware decision-making and improve sample efficiency. PixyzRL supports:

- **Probabilistic Policy Optimization** (e.g., PPO, A2C)
- **On-policy and Off-policy Learning**
- **Memory Management for RL (Replay Buffer, Rollout Buffer)**
- **Integration with Gymnasium environments**
- **Logging and Model Training Utilities**

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.5.1+
- Gymnasium (for environment interaction)

### Install PixyzRL

#### Using `pip`

```bash
pip install torch torchvision torchaudio pixyz gymnasium[box2d] torchrl
```

#### Install from Source

```bash
git clone https://github.com/ItoMasaki/PixyzRL.git
cd PixyzRL
pip install -e .
```

## Quick Start

### 1. Set Up Environment

```python
from pixyzrl.environments import Env

env = Env("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
```

### 2. Define Actor and Critic Networks

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

```python
>>> pixyzrl.utils.print_latex(actor)
p(a|o)

>>> pixyzrl.utils.print_latex(critic)
f(v|o)
```

### 3. Prepare PPO and Buffer

```python
from pixyzrl.models import PPO
from pixyzrl.memory import RolloutBuffer
from pixyzrl.trainer import OnPolicyTrainer

agent = PPO(actor, critic, entropy_coef=0.0, mse_coef=1.0)

buffer = RolloutBuffer(
    2048,
    {
        "obs": {"shape": (4,), "map": "o"},
        "value": {"shape": (1,), "map": "v"},
        "action": {"shape": (2,), "map": "a"},
        "reward": {"shape": (1,)},
        "done": {"shape": (1,)},
        "returns": {"shape": (1,), "map": "r"},
        "advantages": {"shape": (1,), "map": "A"},
    },
    "cpu",
    1,
)
```

#### 3.1 Display model as `latex`

```python
>>> print_latex(agent)
mean \left(1.0 MSE(f(v|o), r) - min \left(A clip(\frac{p(a|o)}{old(a|o)}, 0.8, 1.2), A \frac{p(a|o)}{old(a|o)}\right) \right)
```

<img width="1272" alt="latex" src="https://github.com/user-attachments/assets/317f1f12-bf29-4015-87ee-1aa53ed6b26f" />

### 4. Training with Trainer

```python
trainer = OnPolicyTrainer(env, buffer, agent, "cpu")
trainer.train(1000)
```


## Directory Structure

```text
PixyzRL
├── docs
│   └── pixyz
│       └── README.pixyz.md
├── examples  # Example scripts
├── pixyzrl
│   ├── environments  # Environment wrappers
│   ├── models
│   │   ├── on_policy  # On-policy models (e.g., PPO, A2C)
│   │   └── off_policy  # Off-policy models (e.g., DQN)
│   ├── memory  # Experience replay & rollout buffer
│   ├── trainer  # Training utilities
│   ├── losses  # Loss function definitions
│   ├── logger  # Logging utilities
│   └── utils.py
└── pyproject.toml
```

## Future Work

- [ ] Implement **Deep Q-Network (DQN)**
- [ ] Implement **Dreamer** (model-based RL)
- [ ] Integrate with **ChatGPT for automatic architecture generation**
- [ ] Integrate with **[Genesis](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/what_is_genesis.html)**

## License

PixyzRL is released under the [MIT License](./LICENSE).

## Community & Support

For questions and discussions, please visit:

- [GitHub Issues](https://github.com/ItoMasaki/PixyzRL/issues)
- [PixyzRL ChatGPT Page](https://chatgpt.com/g/g-67b7c36695fc8191aca4cb7420dad17c-pixyzrl)
