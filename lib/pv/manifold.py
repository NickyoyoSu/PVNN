import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


PV_SAFETY_LIMIT = 10.0
TINY = 1e-15
EPS = {torch.float32: 1e-6, torch.float64: 1e-12}
_JIT_DISABLED = os.getenv("PVNN_DISABLE_JIT", "0") == "1"
_script = (lambda fn: fn) if _JIT_DISABLED else torch.jit.script


def _eps(x: torch.Tensor) -> float:
    return EPS.get(x.dtype, 1e-12)


@_script
def _pv_beta(x: torch.Tensor, c: torch.Tensor, tiny: float) -> torch.Tensor:
    x2 = (x * x).sum(dim=-1, keepdim=True)
    denom = torch.sqrt(1.0 + c * x2)
    return 1.0 / denom.clamp_min(tiny)


@_script
def _pv_exp0(v: torch.Tensor, c: torch.Tensor, eps: float, safety_limit: float) -> torch.Tensor:
    s = torch.rsqrt(c)
    r = torch.norm(v, p=2, dim=-1, keepdim=True)
    arg = torch.clamp(r / s, max=safety_limit)
    coef = torch.sinh(arg) / arg.clamp_min(eps)
    return coef * v


@_script
def _pv_log0(y: torch.Tensor, c: torch.Tensor, eps: float, safety_limit: float) -> torch.Tensor:
    s = torch.rsqrt(c)
    y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)
    arg = torch.clamp(y_norm / s, max=safety_limit)
    coef = torch.asinh(arg) / arg.clamp_min(eps)
    return coef * y


@_script
def _pv_dist0(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    s = torch.rsqrt(c)
    y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)
    return s * torch.asinh(y_norm / s)


@_script
def _pv_gyro_add(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    tiny: float,
) -> torch.Tensor:
    b_x = _pv_beta(x, c, tiny)
    b_y = _pv_beta(y, c, tiny)
    xy = (x * y).sum(dim=-1, keepdim=True)
    coef = (b_x / (1.0 + b_x)) * (xy * c) + (1.0 - b_y) / b_y
    return x + y + coef * x


@_script
def _pv_gyro_neg(x: torch.Tensor) -> torch.Tensor:
    return -x


@_script
def _pv_dist(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    tiny: float,
) -> torch.Tensor:
    s = torch.rsqrt(c)
    z = _pv_gyro_add(_pv_gyro_neg(x), y, c, tiny)
    b_z = _pv_beta(z, c, tiny)
    p = (b_z / (1.0 + b_z)) * z
    r = torch.norm(p, p=2, dim=-1, keepdim=True)
    arg = torch.clamp(r / s, max=1.0 - 1e-7)
    return 2.0 * s * torch.atanh(arg)


@_script
def _pv_log_map(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    tiny: float,
) -> torch.Tensor:
    z_pv = _pv_gyro_add(_pv_gyro_neg(x), y, c, tiny)
    r_z = torch.norm(z_pv, p=2, dim=-1, keepdim=True).clamp_min(tiny)
    d_xy = _pv_dist(x, y, c, tiny)
    sigma = d_xy / r_z
    sigma = torch.where(d_xy < tiny, torch.ones_like(sigma), sigma)
    b_x = _pv_beta(x, c, tiny)
    tau_coef = (c * b_x / (1.0 + b_x)) * sigma
    dot_x_z = (x * z_pv).sum(dim=-1, keepdim=True)
    return sigma * z_pv + tau_coef * dot_x_z * x


@_script
def _pv_exp_map(
    x: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    tiny: float,
    arg_clip: float,
) -> torch.Tensor:
    s2 = 1.0 / c
    x_norm2 = (x * x).sum(dim=-1, keepdim=True)
    v_norm2 = (v * v).sum(dim=-1, keepdim=True)
    dot_x_v = (x * v).sum(dim=-1, keepdim=True)

    g_pvv = (v_norm2 - (dot_x_v * dot_x_v) / (s2 + x_norm2)).clamp_min(tiny)
    r_g = torch.sqrt(g_pvv)

    b_x = _pv_beta(x, c, tiny)
    coef1 = b_x / (1.0 + b_x)
    coef2 = -(c * (b_x ** 3)) / ((1.0 + b_x).clamp_min(tiny) ** 2)
    dpi_v = coef1 * v + coef2 * dot_x_v * x

    lambda_x = (1.0 + b_x) / b_x.clamp_min(tiny)
    sqrtc = torch.sqrt(c)
    z_arg = torch.clamp(sqrtc * r_g, -arg_clip, arg_clip)
    sinhc = torch.sinh(z_arg) / z_arg.clamp_min(tiny)
    w = lambda_x * sinhc * dpi_v
    return _pv_gyro_add(x, w, c, tiny)


@_script
def _pv_pt_0_to_y(
    y: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    tiny: float,
) -> torch.Tensor:
    b = _pv_beta(y, c, tiny)
    coef = c * b / (1.0 + b)
    dot = (y * v).sum(dim=-1, keepdim=True)
    return v + coef * dot * y


@_script
def _pv_pt_y_to_0(
    y: torch.Tensor,
    w: torch.Tensor,
    c: torch.Tensor,
    tiny: float,
) -> torch.Tensor:
    b = _pv_beta(y, c, tiny)
    coef = c * b / (1.0 + b)
    dot = (y * w).sum(dim=-1, keepdim=True)
    return w - coef * dot * y


@_script
def _pv_proj_tan(v: torch.Tensor, max_norm: float, eps: float) -> torch.Tensor:
    if max_norm <= 0.0:
        return v
    r = torch.norm(v, p=2, dim=-1, keepdim=True).clamp_min(eps)
    return torch.clamp(max_norm / r, max=1.0) * v


@_script
def _pv_proj(x: torch.Tensor, max_norm: float, eps: float) -> torch.Tensor:
    if max_norm <= 0.0:
        return torch.nan_to_num(x)
    norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return torch.nan_to_num(scale * x)


@_script
def _pv_hyperbolic_distance(
    x: torch.Tensor,
    w: torch.Tensor,
    sigma: torch.Tensor,
    c: torch.Tensor,
    eps: float,
    safety_limit: float,
    tiny: float,
) -> torch.Tensor:
    v = _pv_log0(x, c, eps, safety_limit)
    signed = (v * w).sum(dim=-1, keepdim=True) - sigma
    return signed.abs() / torch.sqrt(c.clamp_min(tiny))


def beta(x: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    c_tensor = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    return _pv_beta(x, c_tensor, TINY)


def _sech(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.cosh(x)


class PVManifold(nn.Module):
    """PV manifold with positive curvature scale c and JIT-backed core ops."""

    def __init__(self, c: float, learnable: bool = False, min_c: float = 1e-8):
        super().__init__()
        if c <= 0:
            raise ValueError(f"c must be positive, got {c}")
        self.min_c = float(min_c)
        if learnable:
            init = torch.tensor(float(c), dtype=torch.float32)
            self.raw_c = nn.Parameter(torch.log(torch.expm1(init)))
            self.register_buffer("_c_const", torch.tensor(0.0, dtype=torch.float32))
        else:
            self.register_buffer("_c_const", torch.tensor(float(c), dtype=torch.float32))
            self.raw_c = None

    @property
    def c_tensor(self) -> torch.Tensor:
        if self.raw_c is None:
            return self._c_const
        return F.softplus(self.raw_c) + self.min_c

    @property
    def c(self) -> float:
        return float(self.c_tensor.detach().item())

    @property
    def s(self) -> torch.Tensor:
        return torch.rsqrt(self.c_tensor)

    @property
    def sqrtc(self) -> torch.Tensor:
        return torch.sqrt(self.c_tensor)

    def exp0(self, v: torch.Tensor) -> torch.Tensor:
        return _pv_exp0(v, self.c_tensor.to(dtype=v.dtype, device=v.device), _eps(v), PV_SAFETY_LIMIT)

    def log0(self, y: torch.Tensor) -> torch.Tensor:
        return _pv_log0(y, self.c_tensor.to(dtype=y.dtype, device=y.device), _eps(y), PV_SAFETY_LIMIT)

    def dist0(self, y: torch.Tensor) -> torch.Tensor:
        return _pv_dist0(y, self.c_tensor.to(dtype=y.dtype, device=y.device))

    def pt_0_to_y(self, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return _pv_pt_0_to_y(y, v, self.c_tensor.to(dtype=y.dtype, device=y.device), TINY)

    def pt_y_to_0(self, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return _pv_pt_y_to_0(y, w, self.c_tensor.to(dtype=y.dtype, device=y.device), TINY)

    def gyro_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _pv_gyro_add(x, y, self.c_tensor.to(dtype=x.dtype, device=x.device), TINY)

    def gyro_neg(self, x: torch.Tensor) -> torch.Tensor:
        return _pv_gyro_neg(x)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _pv_dist(x, y, self.c_tensor.to(dtype=x.dtype, device=x.device), TINY)

    def proj(self, x: torch.Tensor, c: float | None = None, max_norm: float | None = None) -> torch.Tensor:
        if max_norm is None:
            return torch.nan_to_num(x)
        return _pv_proj(x, float(max_norm), _eps(x))

    def proj_tan0(self, v: torch.Tensor, c: float | None = None, max_norm: float = 20.0) -> torch.Tensor:
        if max_norm is None:
            return v
        return _pv_proj_tan(v, float(max_norm), _eps(v))

    def expmap0(self, v: torch.Tensor, c: float | None = None) -> torch.Tensor:
        return self.exp0(v)

    def logmap0(self, y: torch.Tensor, c: float | None = None) -> torch.Tensor:
        return self.log0(y)

    def hyperbolic_distance(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        sigma: torch.Tensor,
        c: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if c is None:
            c_tensor = self.c_tensor.to(dtype=x.dtype, device=x.device)
        else:
            c_tensor = torch.as_tensor(c, dtype=x.dtype, device=x.device).clamp_min(self.min_c)
        return _pv_hyperbolic_distance(x, w, sigma, c_tensor, _eps(x), PV_SAFETY_LIMIT, TINY)

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return _pv_exp_map(x, v, self.c_tensor.to(dtype=x.dtype, device=x.device), TINY, 20.0)

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _pv_log_map(x, y, self.c_tensor.to(dtype=x.dtype, device=x.device), TINY)

    def proj_tan(self, v: torch.Tensor, c: float | None = None, max_norm: float = 20.0) -> torch.Tensor:
        return self.proj_tan0(v, c=c, max_norm=max_norm)


__all__ = ["PVManifold", "PV_SAFETY_LIMIT", "TINY", "_sech", "beta"]
