# PixyzRL

PixyzRL は、Pixyz ライブラリを活用した強化学習（Reinforcement Learning, RL）のためのライブラリです。本ライブラリは、Proximal Policy Optimization (PPO) などの方策勾配法を簡潔に実装し、環境とのインタラクションや学習のプロセスを管理できるように設計されています。

## 特徴

- **Pixyz を活用した確率的モデリング**
- **PPO の実装**
- **環境の管理 (Gymnasium ラッパー)**
- **メモリ管理 (Experience Replay, Rollout Buffer)**
- **ロギング機能**

## インストール

PixyzRL は、Python 3.10 以上で動作し、以下の依存関係を必要とします。

```bash
pip install torch torchaudio torchvision pixyz gymnasium[box2d] torchrl
```

または、リポジトリをクローンして直接インストールできます。

```bash
git clone https://github.com/ItoMasaki/PixyzRL.git
cd PixyzRL
pip install -e .
```

## 使い方

### 1. 環境のセットアップ

PixyzRL では、環境 (`Env` クラス) を作成し、エージェントと連携させることができます。

```python
from pixyzrl.environments import Env

env = Env("CartPole-v1")
```

### 2. PPO エージェントの作成

`PPO` クラスを使用して、エージェントを作成します。

```python
from pixyz.distributions import Normal
from pixyzrl.policy_gradient.ppo import PPO

actor = Normal(loc="s", scale="s", var=["a"], cond_var=["s"], name="actor")
critic = Normal(loc="s", scale="s", var=["v"], cond_var=["s"], name="critic")
ppo_agent = PPO(actor, critic, shared_cnn=None, gamma=0.99, eps_clip=0.2, k_epochs=4, lr_actor=3e-4, lr_critic=1e-3, device="cpu")
```

### 3. トレーニングの実行

`Trainer` クラスを使用して学習を行います。

```python
from pixyzrl.trainer import Trainer
from pixyzrl.memory import ExperienceReplay

memory = ExperienceReplay((4,), (1,), buffer_size=10000, batch_size=64)
trainer = Trainer(env, memory, ppo_agent, device="cpu")
trainer.train(num_iterations=1000)
```

### 4. モデルの保存とロード

```python
ppo_agent.save_model("ppo_model.pth")
ppo_agent.load_model("ppo_model.pth")
```

## ディレクトリ構成

```
PixyzRL
├── docs
│   └── pixyz
│       ├── DeepMarkovModel.md
│       ├── DistributionAPITutorial.md
│       ├── LossAPITutorial.md
│       ├── ModelAPITutorial.md
│       ├── Overview.md
│       └── README.md
├── examples        # サンプルコード
├── pixyzrl
│   ├── environments  # 環境のラッパー
│   ├── policy_gradient  # 強化学習アルゴリズム (PPO など)
│   ├── memory  # 経験リプレイ、ロールアウトバッファ
│   ├── trainer  # トレーニング管理
│   ├── losses  # 損失関数の定義
│   ├── logger  # ロギング機能
└── pyproject.toml
```

## ライセンス

このプロジェクトは MIT ライセンスのもとで公開されています。

## 著者

Masaki Ito (ito.masaki@em.ci.ritsumei.ac.jp)

## リポジトリ

[GitHub - ItoMasaki/PixyzRL](https://github.com/ItoMasaki/PixyzRL)

## コミュニティ・サポート

PixyzRL の詳細については、以下のリンクをご覧ください。

[PixyzRL ChatGPT Page](https://chatgpt.com/g/g-67b7c36695fc8191aca4cb7420dad17c-pixyzrl)
