"""Losses for training models."""

from typing import Any

import sympy
import torch
from pixyz.distributions import Distribution
from pixyz.losses.losses import Loss, LossSelfOperator, MaxLoss, MinLoss, Parameter
from torch import nn


class PPOClipLoss(Loss):
    """PPO clip loss.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> from pixyz.losses import PPOClipLoss
    ...
    >>> class P(Normal):
    ...
    ...     def __init__(self):
    ...         super().__init__(var=["z"], cond_var=["x"], name="p")
    ...
    ...     def forward(self, x):
    ...         return {"loc": x, "scale": torch.nn.functional.softplus(x)}
    ...
    >>> p = P()
    >>> ppo_clip_loss = PPOClipLoss(p, 0.2)
    >>> x = torch.zeros(1, 128)
    >>> z = torch.zeros(1, 128)
    >>> ppo_clip_loss.eval({"z": z, "x": x})
    tensor(0.)  # Expected output
    """

    def __init__(
        self, p: Distribution, q: Distribution, clip: float, adv_var: str = "A"
    ) -> None:
        super().__init__([*p.cond_var, *p.var, adv_var])

        self.p = p
        self.q = q
        self.clip = clip
        self.adv_var = adv_var

        surrogate1 = RatioLoss(p, q) * Parameter(adv_var)
        surrogate2 = ClipLoss(RatioLoss(p, q), 1 - clip, 1 + clip) * Parameter(adv_var)
        self.loss = -MinLoss(surrogate1, surrogate2)

    @property
    def _symbol(self) -> sympy.Symbol:
        return sympy.Symbol(
            f"-min(\\frac{{{self.p.prob_text}}}{{{self.q.prob_text}}}, clip(\\frac{{{self.p.prob_text}}}{{{self.q.prob_text}}}, 1-{self.clip}, 1+{self.clip}))"
        )

    def forward(
        self, x_dict: dict[str, torch.Tensor], **kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        clip_loss, x_dict = self.loss(x_dict, **kwargs)

        return clip_loss, {}


class ValueClipLoss(Loss):
    """Value clip loss.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> from pixyz.losses import ValueClipLoss
    ...
    >>> class P(Normal):
    ...
    ...     def __init__(self):
    ...         super().__init__(var=["z"], cond_var=["x"], name="p")
    ...
    ...     def forward(self, x):
    ...         return {"loc": x, "scale": torch.nn.functional.softplus(x)}
    ...
    >>> p = P()
    >>> value_clip_loss = ValueClipLoss(p, 0.2)
    >>> x = torch.zeros(1, 128)
    >>> z = torch.zeros(1, 128)
    >>> value_clip_loss.eval({"z": z, "x": x})
    tensor(0.)  # Expected output
    """

    def __init__(self, p: Distribution, vtarget_var: str, clip: float) -> None:
        super().__init__([*p.cond_var, *p.var, vtarget_var])

        self.p = p
        self.clip = clip
        self.vtarget_var = vtarget_var

    @property
    def _symbol(self) -> sympy.Symbol:
        return sympy.Symbol(
            f"max(|{self.p.prob_text} - {self.vtarget_var}|^2, |clip({self.p.prob_text}, -{self.clip}, {self.clip}) - {self.vtarget_var}|^2)"
        )

    def forward(
        self, x_dict: dict[str, torch.Tensor], **kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        old_value_prediction = self.p.sample(x_dict)[self.p.var[0]].detach()
        value_prediction = self.p.sample(x_dict)[self.p.var[0]]

        value_prediction_clipped = old_value_prediction + torch.clamp(
            value_prediction - old_value_prediction, -self.clip, self.clip
        )
        loss = torch.max(
            torch.square(x_dict[self.vtarget_var] - value_prediction),
            torch.square(x_dict[self.vtarget_var] - value_prediction_clipped),
        )

        return loss, {}


class RatioLoss(Loss):
    """Compute the ratio of two distributions of the same type.

    Examples
    --------
    >>> import torch
    >>> from torch.nn import functional as F
    >>> from pixyz.distributions import Normal
    ...
    >>> # Set distributions
    >>> class P(Normal):
    ...     def __init__(self):
    ...         super().__init__(var=["z"],cond_var=["x"],name="p")
    ...     def forward(self, x):
    ...         return {"loc": x, "scale": F.softplus(x)}
    >>>
    >>> class Q(Normal):
    ...     def __init__(self):
    ...         super().__init__(var=["z"],cond_var=["x"],name="q")
    ...     def forward(self, x):
    ...         return {"loc": x, "scale": F.softplus(x)}
    >>>
    >>> ratio_loss = RatioLoss(P(), Q()).mean()
    >>>
    >>> x = torch.zeros(1, 128)
    >>> z = torch.zeros(1, 128)
    >>>
    >>> ratio_loss.eval({"z": z, "x": x})
    tensor(1.)

    """

    def __init__(
        self,
        p: Distribution,
        q: Distribution,
        sum_features: bool = False,
        feature_dims: int | None = None,
    ) -> None:
        super().__init__(p.var + p.input_var + q.var + q.input_var)

        self.sum_features = sum_features
        self.feature_dims = feature_dims

        if p.name == q.name:
            msg = "The two distributions are of different types. Make the two distributions of the same type."
            raise ValueError(msg)

        self.p = p
        self.q = q

    @property
    def _symbol(self) -> sympy.Symbol:
        return sympy.Symbol(f"\\frac{{{self.p.prob_text}}}{{{self.q.prob_text}}}")

    def forward(
        self, x_dict: dict[str, Any], **kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[None, None]]:
        p_log_prob = (
            self.p.log_prob(
                sum_features=self.sum_features, feature_dims=self.feature_dims, **kwargs
            )
            .eval(x_dict)
            .sum(dim=-1)
        )
        q_log_prob = (
            self.q.log_prob(
                sum_features=self.sum_features, feature_dims=self.feature_dims, **kwargs
            )
            .eval(x_dict)
            .sum(dim=-1)
        )

        ratio = torch.exp(p_log_prob - q_log_prob.detach()).reshape(-1, 1)

        return ratio, {}


class ClipLoss(LossSelfOperator):
    """Cut out the error within a certain range.

    Examples
    --------
    >>> import torch
    >>> from torch.nn import functional as F
    >>> from pixyz.distributions import Normal
    >>> from pixyz.losses import LogProb
    ...
    >>> class P(Normal):
    ...     def __init__(self):
    ...         super().__init__(var=["z"],cond_var=["x"],name="p")
    ...     def forward(self, x):
    ...         return {"loc": x, "scale": F.softplus(x)}
    >>>
    >>> p = P()
    >>>
    >>> clip_loss = ClipLoss(LogProb(p), 0.9, 1.1)
    >>>
    >>> x = torch.zeros(1, 128)
    >>> z = torch.zeros(1, 128)
    >>>
    >>> clip_loss.eval({"z": z, "x": x})
    tensor([0.9000])

    """

    def __init__(self, loss1: Loss | LossSelfOperator, min: float, max: float) -> None:
        super().__init__(loss1)

        self.min = min
        self.max = max

    @property
    def _symbol(self) -> sympy.Symbol:
        return sympy.Symbol(f"clip({self.loss1.loss_text}, {self.min}, {self.max})")

    def forward(
        self, x_dict: dict[str, torch.Tensor], **kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        loss, x_dict = self.loss1(x_dict, **kwargs)
        loss = torch.clamp(loss, self.min, self.max)

        return loss, x_dict


class MSELoss(Loss):
    """Mean Square Error.

    Examples
    --------
    >>> import torch
    >>> from torch.nn import functional as F
    >>> from pixyz.distributions import Normal
    >>>
    >>> class P(Normal):
    ...     def __init__(self):
    ...         super().__init__(var=["z"],cond_var=["x"],name="p")
    ...     def forward(self, x):
    ...         return {"loc": x, "scale": F.softplus(x)}
    >>>
    >>> mse_loss = MSELoss(P(), "y")
    >>>
    >>> x = torch.rand(1, 128)
    >>> y = torch.rand(1, 128)
    >>>
    >>> mse_loss.eval({"x": x, "y": y}).shape
    torch.Size([])

    """

    def __init__(self, p: Distribution, var: str, reduction: str = "mean") -> None:
        """Initialize the loss."""
        super().__init__([*p.cond_var, var])

        self.p = p
        self.var = var

        self.mse = nn.MSELoss(reduction=reduction)

    @property
    def _symbol(self) -> sympy.Symbol:
        """Return the symbol of the loss."""
        return sympy.Symbol(f"MSE({self.p.prob_text}, {self.var})")

    def forward(
        self, x_dict: dict[str, torch.Tensor], **kwargs: dict[str, bool | torch.Size]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass."""
        pred = self.p.sample(x_dict, **kwargs)[self.p.var[0]].squeeze()
        loss = self.mse(pred, x_dict[self.var].squeeze())
        return loss, {}
