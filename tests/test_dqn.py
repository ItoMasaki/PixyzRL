import torch
from pixyz.distributions import Deterministic

from pixyzrl.memory import RolloutBuffer
from pixyzrl.models import DQN


class DummyQNetwork(Deterministic):
    def __init__(self) -> None:
        super().__init__(var=["q"], cond_var=["o"], name="q")
        self.net = torch.nn.Linear(4, 2)

    def forward(self, o: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"q": self.net(o)}


def test_dqn_train_step_runs() -> None:
    q_network = DummyQNetwork()
    dqn = DQN(q_network)

    buffer = RolloutBuffer(
        buffer_size=8,
        env_dict={
            "obs": {"shape": (4,), "map": "o"},
            "next_obs": {"shape": (4,), "map": "o_next"},
            "action": {"shape": (2,), "map": "a"},
            "reward": {"shape": (1,), "map": "r"},
            "done": {"shape": (1,), "map": "d"},
            "value": {"shape": (1,), "map": "v"},
        },
        n_envs=1,
    )

    for _ in range(8):
        action = torch.nn.functional.one_hot(torch.randint(0, 2, (1,)), num_classes=2)
        buffer.add(
            obs=torch.randn(4),
            next_obs=torch.randn(4),
            action=action.squeeze(0).float(),
            reward=torch.randn(1),
            done=torch.zeros(1),
            value=torch.zeros(1),
        )

    loss = dqn.train_step(buffer, batch_size=4, num_epochs=1)
    assert isinstance(loss, float)


def test_dqn_select_action_shape() -> None:
    q_network = DummyQNetwork()
    dqn = DQN(q_network, epsilon_start=0.0, epsilon_end=0.0)

    action = dqn.select_action({"o": torch.randn(3, 4)})["a"]
    assert action.shape == (3, 2)
