from pixyz.distributions import Normal, ProductOfNormal
from pixyz.losses import KullbackLeibler

import torch
from torch.nn import functional as F


class P_1(Normal):
    def __init__(self):
        super().__init__(var=["z"], cond_var=[], name="p")
        self.loc = torch.nn.Parameter(torch.zeros(64))
        self.scale = torch.nn.Parameter(torch.ones(64))

    def forward(self):
        return {"loc": self.loc, "scale": F.softplus(self.scale)}
    

class P_2(Normal):
    def __init__(self):
        super().__init__(var=["z"], cond_var=[], name="p")
        self.loc = torch.nn.Parameter(torch.zeros(64))
        self.scale = torch.nn.Parameter(torch.ones(64))

    def forward(self):
        return {"loc": self.loc, "scale": F.softplus(self.scale)}


class Q(Normal):
    def __init__(self):
        super().__init__(var=["z"], cond_var=[], name="q")
        self.loc = torch.nn.Parameter(torch.zeros(64))
        self.scale = torch.nn.Parameter(torch.ones(64))

    def forward(self):
        return {"loc": self.loc, "scale": F.softplus(self.scale)}
    

poe = ProductOfNormal([P_1(), P_2()])
kl = KullbackLeibler(poe, Q())
print(kl({}))