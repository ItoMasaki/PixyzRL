"""Losses for training models."""

from typing import Any

import sympy
import torch
from pixyz.distributions import Distribution
from pixyz.losses.losses import Loss, LossSelfOperator, MinLoss, NegLoss, Parameter
from torch import nn


def ppo(actor: Distribution, actor_old: Distribution, clip_param: float = 0.2) -> NegLoss:
    """Proximal Policy Optimization."""
    surr1 = RatioLoss(actor, actor_old) * Parameter("A")
    surr2 = ClipLoss(RatioLoss(actor, actor_old), 1 - clip_param, 1 + clip_param) * Parameter("A")
    return -MinLoss(surr1, surr2)


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
    ...         self.model_loc = torch.nn.Linear(128, 64)
    ...         self.model_scale = torch.nn.Linear(128, 64)
    ...     def forward(self, x):
    ...         return {"loc": self.model_loc(x), "scale": F.softplus(self.model_scale(x))}
    >>>
    >>> class Q(Normal):
    ...     def __init__(self):
    ...         super().__init__(var=["z"],cond_var=["x"],name="q")
    ...         self.model_loc = torch.nn.Linear(128, 64)
    ...         self.model_scale = torch.nn.Linear(128, 64)
    ...     def forward(self, x):
    ...         return {"loc": self.model_loc(x), "scale": F.softplus(self.model_scale(x))}
    >>>
    >>> p = P()
    >>> q = Q()
    >>>
    >>> ratio_loss = RatioLoss(p, q).mean()
    >>>
    >>> x = torch.zeros(1, 128)
    >>> z = torch.zeros(1, 64)
    >>>
    >>> ratio_loss.eval({"z": z, "x": x})
    tensor(0.9940, grad_fn=<MeanBackward0>)

    """

    def __init__(self, p: Distribution, q: Distribution, sum_features: bool = False, feature_dims: int | None = None) -> None:
        super().__init__(p.var + p.input_var + q.var + q.input_var)

        self.sum_features = sum_features
        self.feature_dims = feature_dims

        if p.name == q.name:
            msg = "The two distributions are of different types. Make the two distributions of the same type."
            raise ValueError(msg)

        self.p = p
        self.q = q

        self.q.requires_grad = False

    @property
    def _symbol(self):
        return sympy.Symbol(f"\\frac{{{self.p.prob_text}}}{{{self.q.prob_text}}}")

    def forward(self, x_dict: dict[str, Any], **kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict[None, None]]:
        p_log_prob = self.p.log_prob(sum_features=self.sum_features, feature_dims=self.feature_dims, **kwargs).eval(x_dict).sum(dim=-1)
        q_log_prob = self.q.log_prob(sum_features=self.sum_features, feature_dims=self.feature_dims, **kwargs).eval(x_dict).sum(dim=-1)

        ratio = torch.exp(p_log_prob - q_log_prob.detach())

        return ratio, {}


class ClipLoss(LossSelfOperator):
    """Cut out the error within a certain range.

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
    ...         self.model_loc = torch.nn.Linear(128, 64)
    ...         self.model_scale = torch.nn.Linear(128, 64)
    ...     def forward(self, x):
    ...         return {"loc": self.model_loc(x), "scale": F.softplus(self.model_scale(x))}
    >>>
    >>> p = P()
    >>>
    >>> clip_loss = ClipLoss(LogProb(p), 0.9, 1.1)
    >>>
    >>> x = torch.zeros(1, 128)
    >>> z = torch.zeros(1, 64)
    >>>
    >>> clip_loss.eval({"z": z, "x": x})
    tensor([1.], grad_fn=<ClampBackward1>)

    """

    def __init__(self, loss1: Loss | LossSelfOperator, min: float, max: float) -> None:
        super().__init__(loss1)

        self.min = min
        self.max = max

    @property
    def _symbol(self):
        return sympy.Symbol(f"clip({self.loss1.loss_text}, {self.min}, {self.max})")

    def forward(self, x_dict: dict[str, torch.Tensor], **kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
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
    >>> mse_loss = MSELoss("x", "y")
    >>>
    >>> x = torch.rand(1, 128)
    >>> y = torch.rand(1, 128)
    >>>
    >>> mse_loss.eval({"x": x, "y": y})
    tensor(0.1752)

    """

    def __init__(self, p: Distribution, var: str, reduction: str = "mean") -> None:
        """Initialize the loss."""
        super().__init__([*p.cond_var, var])

        self.p = p
        self.var = var

        self.MSELoss = nn.MSELoss(reduction=reduction)

    @property
    def _symbol(self) -> sympy.Symbol:
        """Return the symbol of the loss."""
        return sympy.Symbol(f"MSE({self.p.prob_text}, {self.var2})")

    def forward(self, x_dict: dict[str, torch.Tensor], **kwargs: dict[str, bool | torch.Size]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass."""
        loss = self.MSELoss(self.p.sample(x_dict, **kwargs)[self.p.var[0]].squeeze(), x_dict[self.var].squeeze()).mean()

        return loss, {}
