import torch
from pixyz.distributions import Deterministic, Normal

from pixyzrl.memory import RolloutBuffer
from pixyzrl.models import SAC


class DummyActor(Normal):
    def __init__(self) -> None:
        super().__init__(var=["a"], cond_var=["o"], name="pi")
        self.net = torch.nn.Linear(4, 2)

    def forward(self, o: torch.Tensor) -> dict[str, torch.Tensor]:
        loc = self.net(o)
        scale = torch.ones_like(loc) * 0.5
        return {"loc": loc, "scale": scale}


class DummyCritic(Deterministic):
    def __init__(self, name: str) -> None:
        super().__init__(var=["q"], cond_var=["o", "a"], name=name)
        self.net = torch.nn.Linear(6, 1)

    def forward(self, o: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"q": self.net(torch.cat([o, a], dim=-1))}


def test_sac_train_step_runs() -> None:
    actor = DummyActor()
    critic1 = DummyCritic("q1")
    critic2 = DummyCritic("q2")
    sac = SAC(actor, critic1, critic2)

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
        buffer.add(
            obs=torch.randn(4),
            next_obs=torch.randn(4),
            action=torch.randn(2),
            reward=torch.randn(1),
            done=torch.zeros(1),
            value=torch.zeros(1),
        )

    loss = sac.train_step(buffer, batch_size=4, num_epochs=1)
    assert isinstance(loss, float)
