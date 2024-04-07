import sympy

import torch
from torch import nn

from pixyz.losses.losses import Loss, LossSelfOperator
from pixyz.losses import Entropy, MinLoss, ValueLoss, Parameter


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

    def __init__(self, p, q, sum_features=False, feature_dims=None):
        super().__init__(p.var + p.input_var + q.var + q.input_var)

        self.sum_features = sum_features
        self.feature_dims = feature_dims

        if p.name == q.name:
            raise ValueError("The two distributions are of different types. Make the two distributions of the same type.")

        self.p = p
        self.q = q
        
        self.q.requires_grad = False

    @property
    def _symbol(self):
        return sympy.Symbol(f"\\frac{{{self.p.prob_text}}}{{{self.q.prob_text}}}")

    def forward(self, x_dict={}, **kwargs):
        p_log_prob = self.p.log_prob(sum_features=self.sum_features, feature_dims=self.feature_dims, **kwargs).eval(x_dict)
        q_log_prob = self.q.log_prob(sum_features=self.sum_features, feature_dims=self.feature_dims, **kwargs).eval(x_dict)

        ratio = torch.exp(p_log_prob - q_log_prob.detach())

        return ratio, {}


class ClipLoss(LossSelfOperator):
    """Cut out the error within a certain range

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

    def __init__(self, loss1, min, max):
        super().__init__(loss1)

        self.min = min
        self.max = max

    @property
    def _symbol(self):
        return sympy.Symbol(f"clip({self.loss1.loss_text}, {self.min}, {self.max})")

    def forward(self, x_dict={}, **kwargs):
        loss, x_dict = self.loss1(x_dict, **kwargs)
        loss = torch.clamp(loss, self.min, self.max)

        return loss, x_dict


class MSELoss(Loss):
    """Mean Square Error

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

    def __init__(self, var1, var2):
        super().__init__([var1, var2])

        self.var1 = var1
        self.var2 = var2

        self.MSELoss = nn.MSELoss(reduction="none")

    @property
    def _symbol(self):
        return sympy.Symbol(f"MSE({self.var1},{self.var2})")

    def forward(self, x_dict={}, **kwargs):

        loss = self.MSELoss(x_dict[self.var1], x_dict[self.var2])

        return loss, {}
