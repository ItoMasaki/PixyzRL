import torch
from pixyz import distributions as dists
from torch import nn
from torch.nn import functional as F


class Encoder(dists.Normal):
    """Encoder model for the VAE."""

    def __init__(self, state_size: int, belief_size: int) -> None:
        """Initialize the encoder model."""
        super().__init__(var=["s_t"], cond_var=["o_t", "z_t"], name="p")
        self.feature_extractor = nn.Sequential(  # models.resnet18()
            nn.Conv2d(3, 8, 5, 2),
            nn.ReLU(),
            nn.Conv2d(8, 32, 5, 2),
            nn.ReLU(),
            nn.Conv2d(32, 128, 3, 2),
            nn.Flatten(),
            nn.Linear(12800, 1000),
            nn.ReLU(),
        )

        self.loc_scale = nn.Sequential(
            nn.Linear(1000 + belief_size, 512),
            nn.ReLU(),
            nn.Linear(512, state_size * 2),
        )

    def forward(self, o_t: torch.Tensor, z_t: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.feature_extractor(o_t)
        h = torch.cat([h, z_t], dim=-1)

        loc_scale = self.loc_scale(h)

        loc, scale = torch.chunk(loc_scale, 2, dim=-1)

        return {"loc": loc, "scale": F.softplus(scale) + 0.001}


class Decoder(dists.Normal):
    def __init__(self, state_size: int, belief_size: int) -> None:
        super().__init__(var=["o_t"], cond_var=["z_t", "s_t"], name="p")

        self.state_size = state_size
        self.belief_size = belief_size

        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size + belief_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.loc = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 6, stride=5),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 6, stride=5),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 2, stride=1),
        )

    def forward(self, z_t: torch.Tensor, s_t: torch.Tensor) -> dict[str, torch.Tensor]:
        h = torch.cat([z_t, s_t], dim=-1)

        h = self.feature_extractor(h)

        loc = self.loc(h.view(-1, 256, 1, 1))

        return {"loc": F.sigmoid(loc), "scale": 1.0}


class Reward(dists.Normal):
    def __init__(self, belief_size: int, state_size: int) -> None:
        super().__init__(var=["r_t"], cond_var=["z_t", "s_t"], name="p")

        self.feature_extractor = nn.Sequential(
            nn.Linear(belief_size + state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.loc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, z_t: torch.Tensor, s_t: torch.Tensor) -> dict[str, torch.Tensor]:
        h = torch.cat([z_t, s_t], dim=-1)

        h = self.feature_extractor(h)

        loc = self.loc(h)

        return {"loc": loc, "scale": 0.1}


class Transition(dists.Deterministic):
    def __init__(self, state_size: int, action_size: int, belief_size: int) -> None:
        super().__init__(var=["z_tp1"], cond_var=["z_t", "s_t", "a_t", "t_t"], name="p")

        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)

        self.act_fn = nn.ReLU()

    def forward(self, s_t: torch.Tensor, a_t: torch.Tensor, z_t: torch.Tensor, t_t: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.act_fn(self.fc_embed_state_action(torch.cat([s_t, a_t], dim=1) * t_t))
        z_tp1 = self.rnn(hidden, z_t * t_t)

        return {"z_tp1": z_tp1}


class Stochastic(dists.Normal):
    def __init__(self, state_size: int, belief_size: int) -> None:
        super().__init__(var=["s_t"], cond_var=["z_t"], name="p")

        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, state_size),
            nn.ReLU(),
            nn.Linear(state_size, state_size),
            nn.ReLU(),
        )

        self.loc_scale = nn.Sequential(
            nn.Linear(state_size, belief_size),
            nn.ReLU(),
            nn.Linear(belief_size, belief_size),
            nn.ReLU(),
            nn.Linear(belief_size, belief_size * 2),
        )

    def forward(self, z_t: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.feature_extractor(z_t)

        loc_scale = self.loc_scale(h)
        loc, scale = torch.chunk(loc_scale, 2, dim=-1)

        return {"loc": loc, "scale": F.softplus(scale) + 0.001}


class Actor(dists.Normal):
    def __init__(self, action_size: int, belief_size: int, state_size: int, name: str) -> None:
        super().__init__(var=["a_t"], cond_var=["z_t", "s_t"], name=name)

        self.feature_extractor = nn.Sequential(
            nn.Linear(belief_size + state_size, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )

        self.loc_scale = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_size * 2),
        )

    def forward(self, z_t: torch.Tensor, s_t: torch.Tensor) -> dict[str, torch.Tensor]:
        h = torch.cat([z_t, s_t], dim=-1)

        h = self.feature_extractor(h)

        loc_scale = self.loc_scale(h)
        loc, scale = torch.chunk(loc_scale, 2, dim=-1)

        loc_0 = torch.tanh(loc[:, 0])
        loc_1 = torch.sigmoid(loc[:, 1])
        loc_2 = torch.sigmoid(loc[:, 2])

        loc = torch.stack([loc_0, loc_1, loc_2], dim=-1)
        scale = torch.clamp(F.softplus(scale) + 0.1, 0.1, 10.0)

        return {"loc": loc, "scale": scale}


class Critic(dists.Normal):
    def __init__(self, belief_size: int, state_size: int) -> None:
        super().__init__(var=["v_t"], cond_var=["z_t", "s_t"], name="p")

        self.feature_extractor = nn.Sequential(
            nn.Linear(belief_size + state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.loc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, z_t: torch.Tensor, s_t: torch.Tensor) -> dict[str, torch.Tensor]:
        h = torch.cat([z_t, s_t], dim=-1)

        h = self.feature_extractor(h)

        loc = self.loc(h)

        return {"loc": loc, "scale": 1.0}
