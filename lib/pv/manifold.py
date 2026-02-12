import math
import os
import torch


PV_DEBUG = os.environ.get("PV_DEBUG") == "1"
PV_SAFETY_LIMIT = 10.0
TINY = 1e-15
EPS = {torch.float32: 1e-6, torch.float64: 1e-12}


def _eps(x: torch.Tensor) -> float:
    return EPS.get(x.dtype, 1e-12)


def beta(x: torch.Tensor, c: float) -> torch.Tensor:
    denom = torch.sqrt(1.0 + c * (x * x).sum(dim=-1, keepdim=True))
    return 1.0 / denom.clamp_min(TINY)


def _sech(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.cosh(x)


def gamma(x: torch.Tensor, c: float) -> torch.Tensor:
    denom = torch.sqrt(1.0 - c * (x * x).sum(dim=-1, keepdim=True))
    return 1.0 / denom.clamp_min(TINY)


def _sinhc(z: torch.Tensor) -> torch.Tensor:
    eps = _eps(z)
    z2 = z * z
    y = torch.sinh(z) / z.clamp_min(eps)
    y0 = 1.0 + z2 / 6.0
    return torch.where(z.abs() < eps, y0, y)


def acosh(z: torch.Tensor, eps: float = TINY) -> torch.Tensor:
    z = torch.clamp(z, min=1.0 + eps)
    return torch.log(z + torch.sqrt(z * z - 1))


def sigma(u: torch.Tensor, v: torch.Tensor, kappa: float, eps: float = TINY) -> torch.Tensor:
    dot = (u * v).sum(-1, keepdim=True)
    base = (v * v).sum(-1, keepdim=True) - dot ** 2 / (1 + (u * u).sum(-1, keepdim=True))
    return torch.sqrt(torch.clamp(base, min=eps)) / kappa


def tanh(x):
    return torch.tanh(x)


def artanh(x):
    return torch.atanh(x)


class PVManifold:
    """PV model with curvature K=-c < 0; use only c and s=1/sqrt(c)."""

    def __init__(self, c: float):
        assert c > 0, "c must be positive."
        self.c = float(c)
        self.s = 1.0 / math.sqrt(self.c)   # s = 1/√c
        self.sqrtc = math.sqrt(self.c)     # sqrt(c), used by exp_map and related ops

    # ----- Exp/Log at the origin -----
    def exp0(self, v: torch.Tensor) -> torch.Tensor:
        # Exp_0(v) = (1/√c) sinh(√c ||v||) v/||v||  == s * sinh(||v||/s) * v/||v||
        r = v.norm(dim=-1, keepdim=True)
        arg = (r / self.s).clamp(max=PV_SAFETY_LIMIT)
        coef = torch.sinh(arg) / arg.clamp_min(_eps(v))
        y = coef * v
        if PV_DEBUG and (torch.isnan(y).any() or torch.isinf(y).any()):
            with torch.no_grad():
                def _stat(t):
                    t2 = torch.nan_to_num(t)
                    return float(t2.min().item()), float(t2.max().item()), float(t2.mean().item())
                print("[PV.exp0][diag] y NaN/Inf -> r stats(min,max,mean)=", _stat(r),
                      " r/s max=", float((r / self.s).abs().max().item()))
        return y

    def log0(self, y: torch.Tensor) -> torch.Tensor:
        # Log_0(y) = (1/√c) asinh(√c ||y||) y/||y|| == s * asinh(||y||/s) * y/||y||
        s = y.norm(dim=-1, keepdim=True)
        arg = (s / self.s).clamp(max=PV_SAFETY_LIMIT)
        coef = torch.asinh(arg) / arg.clamp_min(_eps(y))
        v = coef * y
        if PV_DEBUG and (torch.isnan(v).any() or torch.isinf(v).any()):
            with torch.no_grad():
                def _stat(t):
                    t2 = torch.nan_to_num(t)
                    return float(t2.min().item()), float(t2.max().item()), float(t2.mean().item())
                print("[PV.log0][diag] v NaN/Inf -> ||y|| stats=", _stat(s),
                      " asinh(arg) arg_max=", float((s / self.s).abs().max().item()))
        return v

    def dist0(self, y: torch.Tensor) -> torch.Tensor:
        # d(0,y) = s * asinh(||y||/s)
        s = y.norm(dim=-1, keepdim=True)
        return self.s * torch.asinh(s / self.s)

    # ----- Parallel transport along geodesic 0 -> y -----
    def pt_0_to_y(self, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b = beta(y, self.c)                                # (...,1)
        coef = self.c * b / (1.0 + b)                      # c * β/(1+β)
        dot = (y * v).sum(dim=-1, keepdim=True)
        return v + coef * dot * y

    def pt_y_to_0(self, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        b = beta(y, self.c)
        coef = self.c * b / (1.0 + b)
        dot = (y * w).sum(dim=-1, keepdim=True)
        return w - coef * dot * y

    # ----- PV gyroaddition (closed form; your screenshot Eq. (2)) -----
    def gyro_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x ⊕ y = x + y + { β_x/(1+β_x) * <x,y>/s^2 + (1-β_y)/β_y } x
        b_x = beta(x, self.c)                               # (...,1)
        b_y = beta(y, self.c)                               # (...,1)
        s2  = self.s * self.s                               # = 1/c
        xy  = (x * y).sum(dim=-1, keepdim=True)
        coef = (b_x / (1.0 + b_x)) * (xy / s2) + (1.0 - b_y) / b_y
        return x + y + coef * x

    def gyro_neg(self, x: torch.Tensor) -> torch.Tensor:
        return -x

    # ----- Geodesic distance via PB projection π((−x)⊕y) -----
    def _pi_ball(self, u: torch.Tensor) -> torch.Tensor:
        b = beta(u, self.c)
        return (b / (1.0 + b)) * u

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # d(x,y) = 2s * artanh( ||π( (−x)⊕y )|| / s )
        z = self.gyro_add(self.gyro_neg(x), y)
        p = self._pi_ball(z)
        r = p.norm(dim=-1, keepdim=True)
        arg = torch.clamp(r / self.s, max=1 - 1e-15)
        d = 2.0 * self.s * torch.atanh(arg)
        if torch.isnan(d).any() or torch.isinf(d).any():
            with torch.no_grad():
                print("[PV.dist][diag] d NaN/Inf; ||z|| max=", float(z.norm(dim=-1).max().item()),
                      " ||p||/s max=", float((r/self.s).max().item()))
        return d

    # ----- projections (PV is unconstrained; add mild clipping only) -----
    def proj(self, x: torch.Tensor, c: float | None = None, max_norm: float | None = None) -> torch.Tensor:
        if max_norm is None:
            return torch.nan_to_num(x)
        r = x.norm(dim=-1, keepdim=True).clamp_min(_eps(x))
        return torch.nan_to_num(torch.clamp(max_norm / r, max=1.0) * x)

    def proj_tan0(self, v: torch.Tensor, c: float | None = None, max_norm: float = 20.0) -> torch.Tensor:
        if max_norm is None:
            return v
        r = v.norm(dim=-1, keepdim=True).clamp_min(_eps(v))
        return torch.clamp(max_norm / r, max=1.0) * v

    # ----- Compatibility aliases (legacy API) -----
    def expmap0(self, v: torch.Tensor, c: float | None = None) -> torch.Tensor:
        return self.exp0(v)

    def logmap0(self, y: torch.Tensor, c: float | None = None) -> torch.Tensor:
        return self.log0(y)

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Thm 4.3 / D.4.1 (Eq 643, 608, 634)
        w = Exp_p(v) ominus p = λ_x * _sinhc(√c * ||v||_p) * dπ_p[v] [cite: 634, 643]
        where λ_x = (1+β_x)/β_x [cite: 705]
        """
        # 1. Compute g_p(v,v) and its norm ||v||_p = r_g
        s2 = self.s * self.s
        x_norm2 = (x * x).sum(dim=-1, keepdim=True)
        v_norm2 = (v * v).sum(dim=-1, keepdim=True)
        dot_x_v = (x * v).sum(dim=-1, keepdim=True)

        # g_p(v,v) = ||v||^2 - <x,v>^2 / (s^2 + ||x||^2) [cite: 68, 538]
        g_pvv = (v_norm2 - (dot_x_v * dot_x_v) / (s2 + x_norm2)).clamp_min(TINY)
        r_g = torch.sqrt(g_pvv)  # ||v||_p

        # 2. Evaluate dπ_p[v] (Eq 608)
        # *** Fix: ensure b_x uses the correct beta_x [cite: 112]
        b_x = beta(x, self.c)  # (..., 1) [cite: 112]

        coef1 = b_x / (1.0 + b_x)
        # dπ_p[v] formulation [cite: 608]
        coef2 = -(self.c * (b_x ** 3)) / ((1.0 + b_x).clamp_min(TINY) ** 2)
        dpi_v = coef1 * v + coef2 * dot_x_v * x

        # 3. Compute w = Exp_p(v) ominus p using the simplified expression
        # λ_x = (1+β_x)/β_x [cite: 705]
        lambda_x = (1.0 + b_x) / b_x.clamp_min(TINY)

        # Use _sinhc(z) = sinh(z)/z; clamp z=√c·r_g to avoid overflow
        z_arg = self.sqrtc * r_g
        z_arg = torch.clamp(z_arg, -20.0, 20.0)
        # w_coef = λ_x * _sinhc(√c * r_g) [cite: 643]
        w_coef = lambda_x * _sinhc(z_arg)
        w = w_coef * dpi_v

        # 4. Final gyro-addition
        y = self.gyro_add(x, w)
        if torch.isnan(y).any() or torch.isinf(y).any():
            with torch.no_grad():
                def _stat(t):
                    t2 = torch.nan_to_num(t)
                    return float(t2.min().item()), float(t2.max().item()), float(t2.mean().item())
                print("[PV.exp_map][diag] y NaN/Inf; g_pvv min=", float(g_pvv.min().item()),
                      " r_g max=", float(r_g.max().item()), " b_x min/max=",
                      float(b_x.min().item()), float(b_x.max().item()),
                      " dpi_v max=", float(dpi_v.abs().max().item()),
                      " w_coef max=", float(w_coef.abs().max().item()))
        return y

    # ----- Core log map at x (general) [cite: 110, 671-709] -----
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compact reconstruction based on [cite: 707]:
        d_pq = dist(p, q)
        σ = d_pq / ||Z_pv||
        τ_term = (c * β_p / (1+β_p)) * σ * <p, Z_pv>
        Log_p(q) = σ * Z_pv + τ_term * p
        """
        # 1. Compute Z_pv = (-x) ⊕ y [cite: 709]
        z_pv = self.gyro_add(self.gyro_neg(x), y)
        r_z_pv_sq = (z_pv * z_pv).sum(dim=-1, keepdim=True)
        r_z_pv = torch.sqrt(r_z_pv_sq.clamp_min(TINY))

        # 2. Compute distance d(x,y); self.dist is numerically stable internally
        d_xy = self.dist(x, y)

        # 3. Compute σ = d(x,y) / ||Z_pv|| [cite: 114, 707]
        sigma_val = d_xy / r_z_pv.clamp_min(TINY)

        # Handle the degenerate case x==y (sigma -> 1)
        is_close = (r_z_pv_sq < TINY)
        sigma_val = torch.where(is_close, torch.ones_like(sigma_val), sigma_val)

        # 4. Compute τ_term [cite: 115, 707]
        # *** Fix: ensure b_x uses the correct beta_x [cite: 112]
        b_x = beta(x, self.c)

        # τ_coef = (c * β_p / (1+β_p)) * σ [cite: 707]
        tau_coef = (self.c * b_x / (1.0 + b_x).clamp_min(TINY)) * sigma_val
        dot_x_z = (x * z_pv).sum(dim=-1, keepdim=True)
        tau_term = tau_coef * dot_x_z

        # 5. Assemble Log_p(q) = σ * Z_pv + (τ_term) * p [cite: 707]
        out = sigma_val * z_pv + tau_term * x
        if PV_DEBUG and (torch.isnan(out).any() or torch.isinf(out).any()):
            with torch.no_grad():
                print("[PV.log_map][diag] NaN/Inf -> ||z_pv|| max=", float(r_z_pv.max().item()),
                      " d(x,y) max=", float(d_xy.max().item()),
                      " sigma max=", float(sigma_val.max().item()),
                      " b_x min/max=", float(b_x.min().item()), float(b_x.max().item()))
        return out

    def proj_tan(self, v: torch.Tensor, c: float | None = None, max_norm: float = 20.0) -> torch.Tensor:
        return self.proj_tan0(v, c=c, max_norm=max_norm)


__all__ = [
    "PVManifold",
    "PV_DEBUG",
    "PV_SAFETY_LIMIT",
    "TINY",
    "_sech",
]

