import torch
from pixyz import distributions as dist
from pixyz.losses import Expectation as E
from pixyz.losses import IterativeLoss, LogProb, MaxLoss
from pixyz.losses import KullbackLeibler as KL
from pixyz.models import Model
from pixyz.utils import epsilon
from torch import nn


class RecurrentStateSpaceModel(Model):
    def __init__(self, state_size: int, belief_size: int, device: str):
        self.encoder = Encoder(state_size).to(device)
        self.decoder = Decoder().to(device)
        self.reward_decoder = Reward().to(device)
        self.discount = Discount().to(device)
        self.transition = Transition(belief_size).to(device)
        self.stochastic = Stochastic(state_size).to(device)

        model_modules = [self.encoder, self.decoder, self.reward_decoder, self.transition, self.stochastic, self.discount]

        free_nats = 3.0
        rec_obs = -E(self.encoder * self.transition, LogProb(self.decoder) + LogProb(self.reward_decoder) + LogProb(self.discount))
        kl = MaxLoss(KL(self.encoder, self.stochastic), free_nats)
        step_loss = rec_obs + kl
        loss = IterativeLoss(step_loss=step_loss, series_var=["o_t", "a_t", "e_t", "r_t", "d_t"], update_value={"h_tp1": "h_t"}).mean()

        super().__init__(loss=loss, distributions=model_modules, optimizer=torch.optim.Adam, optimizer_params={"lr": 1e-3}, clip_grad_norm=100.0)

    def predict(self, o_t: torch.Tensor, a_t: torch.Tensor, h_t: torch.Tensor, e_t: torch.Tensor) -> dict[str, torch.Tensor]:
        s_t = self.encoder.sample({"o_t": o_t, "h_t": h_t})["s_t"]
        o_t = self.decoder.sample_mean({"h_t": h_t, "s_t": s_t})
        r_t = self.reward_decoder.sample_mean({"h_t": h_t, "s_t": s_t})
        d_t = self.discount.sample_mean({"h_t": h_t, "s_t": s_t})

        h_tp1 = self.transition.sample({"s_t": s_t, "h_t": h_t, "a_t": a_t, "e_t": e_t})["h_tp1"]

        return {
            "s_t": s_t,
            "o_t": o_t,
            "r_t": r_t,
            "d_t": d_t,
            "h_tp1": h_tp1,
        }


class Transition(dist.Deterministic):
    def __init__(self, belief_size: int, hidden_size: int = 200):
        super().__init__(var=["h_tp1"], cond_var=["s_t", "h_t", "a_t", "e_t"], name="f")

        self.fc_embed_state_action = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.SiLU(),
            nn.LazyLinear(belief_size),
            nn.SiLU(),
        )

        self.rnn = nn.GRUCell(belief_size, belief_size)

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor, a_t: torch.Tensor, e_t: torch.Tensor) -> dict[str, torch.Tensor]:
        # Compute belief (deterministic hidden state)
        e_t = e_t == 1.0
        hidden = self.fc_embed_state_action(torch.cat([s_t * e_t, a_t], dim=-1))

        h_tp1 = self.rnn(hidden, h_t * e_t)

        return {"h_tp1": h_tp1}


class Stochastic(dist.Normal):
    def __init__(self, state_size: int, hidden_size: int = 200):
        super().__init__(var=["s_t"], cond_var=["h_t"], name="p")

        self.backbone = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.SiLU(),
            nn.LazyLinear(hidden_size),
            nn.SiLU(),
            nn.LazyLinear(hidden_size),
            nn.SiLU(),
            nn.LazyLinear(hidden_size),
            nn.SiLU(),
        )

        self._loc = nn.Sequential(
            nn.LazyLinear(state_size),
        )

        self._scale = nn.Sequential(
            nn.LazyLinear(state_size),
            nn.Softplus(),
        )

    def forward(self, h_t: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(h_t)
        loc = self._loc(h)
        scale = self._scale(h) + epsilon()

        return {"loc": loc, "scale": scale}


class Encoder(dist.Normal):
    def __init__(self, state_size: int):
        super().__init__(var=["s_t"], cond_var=["h_t", "o_t"], name="q")

        self.backbone = nn.Sequential(
            nn.LazyConv2d(16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.LazyConv2d(32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.LazyConv2d(64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.LazyConv2d(128, kernel_size=4, stride=2, padding=1),
            nn.Flatten(),
            nn.LazyLinear(1024),
        )

        self._loc = nn.Sequential(
            nn.LazyLinear(200),
            nn.SiLU(),
            nn.LazyLinear(state_size),
        )

        self._scale = nn.Sequential(
            nn.LazyLinear(200),
            nn.SiLU(),
            nn.LazyLinear(state_size),
            nn.Softplus(),
        )

    def forward(self, o_t: torch.Tensor, h_t: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(o_t)
        h = torch.cat([h, h_t], dim=-1)

        loc = self._loc(h)
        scale = self._scale(h) + epsilon()

        return {"loc": loc, "scale": scale}


class Decoder(dist.Normal):
    def __init__(self, embedding_size: int = 1024, activation_function: str = "SiLU"):
        super().__init__(var=["o_t"], cond_var=["h_t", "s_t"], name="p")

        act_fn = getattr(nn, activation_function)
        self.embedding_size = embedding_size

        self.backbone = nn.Sequential(
            nn.LazyLinear(embedding_size),
            act_fn(),
        )

        self._loc = nn.Sequential(
            nn.LazyConvTranspose2d(128, kernel_size=4, stride=2, padding=1),
            act_fn(),
            nn.LazyConvTranspose2d(32, kernel_size=4, stride=2, padding=1),
            act_fn(),
            nn.LazyConvTranspose2d(8, kernel_size=4, stride=2, padding=1),
            act_fn(),
            nn.LazyConvTranspose2d(3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> dict[str, torch.Tensor | float]:
        h = torch.cat([h_t, s_t], dim=-1)

        h = self.backbone(h).reshape(-1, self.embedding_size // (4 * 4), 4, 4)
        loc = self._loc(h)

        return {"loc": loc, "scale": 1.0}


class Reward(dist.Normal):
    def __init__(self, activation_function: str = "SiLU"):
        super().__init__(var=["r_t"], cond_var=["h_t", "s_t"], name="p")

        act_fn = getattr(nn, activation_function)

        self.backbone = nn.Sequential(
            nn.LazyLinear(200),
            act_fn(),
            nn.LazyLinear(200),
            act_fn(),
        )

        self._loc = nn.Sequential(
            nn.LazyLinear(1),
        )

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> dict[str, torch.Tensor]:
        h = torch.concatenate([h_t, s_t], dim=-1)
        h = self.backbone(h)

        loc = self._loc(h)

        return {"loc": loc, "scale": 1.0}


class Discount(dist.Bernoulli):
    def __init__(self):
        super().__init__(var=["d_t"], cond_var=["h_t", "s_t"], name="p")

        self.backbone = nn.Sequential(
            nn.LazyLinear(200),
            nn.SiLU(),
            nn.LazyLinear(200),
            nn.SiLU(),
        )

        self._probs = nn.Sequential(
            nn.LazyLinear(1),
            nn.Sigmoid(),
        )

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> dict[str, torch.Tensor]:
        h = torch.concatenate([h_t, s_t], dim=-1)
        h = self.backbone(h)

        probs = self._probs(h)
        probs = probs.clamp(0.0, 0.999)

        return {"probs": probs}


class Critic(dist.Normal):
    def __init__(self):
        super().__init__(var=["v_t"], cond_var=["h_t", "s_t"], name="")

        self.backbone = nn.Sequential(
            nn.LazyLinear(200),
            nn.SiLU(),
            nn.LazyLinear(200),
            nn.SiLU(),
        )

        self._loc = nn.Sequential(
            nn.LazyLinear(1),
        )

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> dict[str, torch.Tensor]:
        h = torch.concatenate([h_t, s_t], dim=-1)
        h = self.backbone(h)

        loc = self._loc(h)
        return {"loc": loc, "scale": 1.0}


class Actor(dist.Normal):
    def __init__(self, action_size: int, name: str):
        super().__init__(var=["a_t"], cond_var=["h_t", "s_t"], name=name)

        self.backbone = nn.Sequential(
            nn.LazyLinear(200),
            nn.SiLU(),
            nn.LazyLinear(200),
            nn.SiLU(),
        )

        self._loc = nn.Sequential(
            nn.LazyLinear(action_size),
            nn.Tanh(),
        )

        self._scale = nn.Sequential(
            nn.LazyLinear(action_size),
            nn.Softplus(),
        )

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> dict[str, torch.Tensor]:
        h = torch.concatenate([h_t, s_t], dim=-1)
        h = self.backbone(h)

        loc = self._loc(h)
        scale = self._scale(h) + epsilon()

        return {"loc": loc, "scale": scale}

    def sample_action(self, h_t: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
        action = self.sample({"h_t": h_t, "s_t": s_t}, reparam=True)["a_t"]
        return torch.tanh(action)
