import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .manifold import PVManifold


def gamma(x: torch.Tensor, kappa: float) -> torch.Tensor:
    x_squared = (x * x).sum(dim=-1, keepdim=True)
    kappa_squared = kappa**2
    return torch.sqrt(1.0 + x_squared * kappa_squared)


def _lift(u: torch.Tensor, kappa: float) -> torch.Tensor:
    g = gamma(u, kappa) / kappa
    return torch.cat([g, u], dim=-1)


def _drop(U: torch.Tensor, kappa: float) -> torch.Tensor:
    del kappa
    return U[..., 1:]


class PVLinear(nn.Module):
    """Fully connected layer on the PV manifold via Lorentz lifting."""

    def __init__(
        self,
        manifold: PVManifold,
        in_features: int,
        out_features: int,
        bias: bool = True,
        manifold_out: PVManifold | None = None,
        p_drop: float = 0.0,
    ):
        super().__init__()
        self.manifold_in = manifold
        self.manifold_out = manifold_out or manifold
        self.k_in = self.manifold_in.c
        self.k_out = self.manifold_out.c
        self.p_drop = p_drop

        self.linear = nn.Linear(in_features + 1, out_features, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        X = _lift(u, self.k_in)
        W = F.dropout(self.linear.weight, self.p_drop, self.training)
        Y_space = F.linear(X, W, self.linear.bias)
        t = torch.sqrt(torch.clamp_min((Y_space**2).sum(-1, keepdim=True) + 1.0 / (self.k_out**2), 1e-9))
        Y = torch.cat([t, Y_space], dim=-1)
        return _drop(Y, self.k_out)


class PVLinearLFC(nn.Module):
    """PV variant of the Lorentz Fully Connected (LFC) layer with learnable gate."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kappa: float,
        psi: nn.Module | None = None,
        dropout_p: float = 0.0,
        learnable_lambda: bool = True,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kappa = float(kappa)
        self.eps = eps
        self.psi = psi if psi is not None else nn.Identity()
        self.dropout_p = float(dropout_p)

        self.W = nn.Linear(in_features, out_features, bias=True)
        nn.init.xavier_uniform_(self.W.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.W.bias)

        self.v = nn.Parameter(torch.randn(in_features) * (1.0 / math.sqrt(in_features)))
        self.b_gate = nn.Parameter(torch.zeros(()))

        if learnable_lambda:
            self.lambda_raw = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("lambda_raw", torch.tensor(float("nan")))
        self.learnable_lambda = learnable_lambda

        self.dropout = nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()

    def _lambda(self):
        if self.learnable_lambda:
            return F.softplus(self.lambda_raw) + 1e-6
        return 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_psi = self.psi(x)

        Ww = self.W
        if isinstance(self.dropout, nn.Dropout) and self.training:
            weight = self.dropout(Ww.weight)
            z = F.linear(x_psi, weight, Ww.bias)
        else:
            z = Ww(x_psi)

        z_norm = z.norm(dim=-1, keepdim=True)
        z_hat = z / (z_norm + self.eps)

        gate = self._lambda() * torch.sigmoid(x @ self.v + self.b_gate)
        gate = gate.unsqueeze(-1)

        y_space = gate * z_hat
        return y_space
import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .manifold import PVManifold

TINY = 1e-15


def gamma(x: torch.Tensor, kappa: float) -> torch.Tensor:
    x_squared = (x * x).sum(dim=-1, keepdim=True)
    kappa_squared = kappa**2
    return torch.sqrt(1.0 + x_squared * kappa_squared)


def _lift(u: torch.Tensor, kappa: float) -> torch.Tensor:
    g = gamma(u, kappa) / kappa
    return torch.cat([g, u], dim=-1)


def _drop(U: torch.Tensor, kappa: float) -> torch.Tensor:
    del kappa
    return U[..., 1:]


class PVLinear(nn.Module):
    """
    Fully connected layer on the PV manifold via Lorentz lifting.
    """

    def __init__(
        self,
        manifold: PVManifold,
        in_features: int,
        out_features: int,
        bias: bool = True,
        manifold_out: PVManifold | None = None,
        p_drop: float = 0.0,
    ):
        super().__init__()
        self.manifold_in = manifold
        self.manifold_out = manifold_out or manifold
        self.k_in = self.manifold_in.c
        self.k_out = self.manifold_out.c
        self.p_drop = p_drop

        self.linear = nn.Linear(in_features + 1, out_features, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        X = _lift(u, self.k_in)
        W = F.dropout(self.linear.weight, self.p_drop, self.training)
        Y_space = F.linear(X, W, self.linear.bias)
        t = torch.sqrt(torch.clamp_min((Y_space**2).sum(-1, keepdim=True) + 1.0 / (self.k_out**2), 1e-9))
        Y = torch.cat([t, Y_space], dim=-1)
        return _drop(Y, self.k_out)


class PVLinearLFC(nn.Module):
    """
    PV variant of the Lorentz Fully Connected (LFC) layer with learnable gate.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kappa: float,
        psi: nn.Module | None = None,
        dropout_p: float = 0.0,
        learnable_lambda: bool = True,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kappa = float(kappa)
        self.eps = eps
        self.psi = psi if psi is not None else nn.Identity()
        self.dropout_p = float(dropout_p)

        self.W = nn.Linear(in_features, out_features, bias=True)
        nn.init.xavier_uniform_(self.W.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.W.bias)

        self.v = nn.Parameter(torch.randn(in_features) * (1.0 / math.sqrt(in_features)))
        self.b_gate = nn.Parameter(torch.zeros(()))

        if learnable_lambda:
            self.lambda_raw = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("lambda_raw", torch.tensor(float("nan")))
        self.learnable_lambda = learnable_lambda

        self.dropout = nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()

    def _lambda(self):
        if self.learnable_lambda:
            return F.softplus(self.lambda_raw) + 1e-6
        return 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_psi = self.psi(x)

        Ww = self.W
        if isinstance(self.dropout, nn.Dropout) and self.training:
            weight = self.dropout(Ww.weight)
            z = F.linear(x_psi, weight, Ww.bias)
        else:
            z = Ww(x_psi)

        z_norm = z.norm(dim=-1, keepdim=True)
        z_hat = z / (z_norm + self.eps)

        gate = self._lambda() * torch.sigmoid(x @ self.v + self.b_gate)
        gate = gate.unsqueeze(-1)

        y_space = gate * z_hat
        return y_space

