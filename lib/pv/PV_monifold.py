# pv_manifold.py  (c-only / s-only version; PV-add in closed form)
from __future__ import annotations
import math
import typing as T
import torch
import torch.nn as nn
import torch.nn.functional as F

TINY = 1e-15
EPS = {torch.float32: 1e-6, torch.float64: 1e-12}
def _eps(x: torch.Tensor) -> float: return EPS.get(x.dtype, 1e-12)

# ---------- core factors ----------
def beta(x: torch.Tensor, c: float) -> torch.Tensor:
    denom = torch.sqrt(1.0 + c * (x * x).sum(dim=-1, keepdim=True))
    return 1.0 / denom.clamp_min(TINY)

def gamma(x: torch.Tensor, c: float) -> torch.Tensor:
    denom = torch.sqrt(1.0 - c * (x * x).sum(dim=-1, keepdim=True))
    return 1.0 / denom.clamp_min(TINY)

# ---------- small-angle helpers ----------
def _sinhc(z: torch.Tensor) -> torch.Tensor:
    eps = _eps(z); z2 = z*z
    y = torch.sinh(z) / z.clamp_min(eps)
    y0 = 1.0 + z2/6.0
    return torch.where(z.abs() < eps, y0, y)

def _sech(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.cosh(x)

def _asinhc(z: torch.Tensor) -> torch.Tensor:
    eps = _eps(z); z2 = z*z
    y = torch.asinh(z) / z.clamp_min(eps)
    y0 = 1.0 - z2/6.0
    return torch.where(z.abs() < eps, y0, y)

# =====================================================================
#                           Proper-Velocity space
# =====================================================================
class PVManifold:
    """PV model with curvature K=-c < 0; use only c and s=1/sqrt(c)."""
    def __init__(self, c: float):
        assert c > 0, "c must be positive."
        self.c = float(c)
        self.s = 1.0 / math.sqrt(self.c)   # s = 1/√c
        self.sqrtc = math.sqrt(self.c)     # sqrt(c) for exp_map etc.

    # ----- Exp/Log at the origin -----
    def exp0(self, v: torch.Tensor) -> torch.Tensor:
        # Exp_0(v) = (1/√c) sinh(√c ||v||) v/||v||  == s * sinh(||v||/s) * v/||v||
        r = v.norm(dim=-1, keepdim=True)
        coef = torch.sinh(r / self.s) / (r / self.s).clamp_min(_eps(v))
        y = coef * v
        # Print diagnostic only on anomaly
        if torch.isnan(y).any() or torch.isinf(y).any():
            with torch.no_grad():
                def _stat(t):
                    t2 = torch.nan_to_num(t)
                    return float(t2.min().item()), float(t2.max().item()), float(t2.mean().item())
                print("[PV.exp0][diag] y NaN/Inf; r stats(min,max,mean)=", _stat(r),
                      " r/s max=", float((r/self.s).abs().max().item()))
        return y

    def log0(self, y: torch.Tensor) -> torch.Tensor:
        # Log_0(y) = (1/√c) asinh(√c ||y||) y/||y|| == s * asinh(||y||/s) * y/||y||
        s = y.norm(dim=-1, keepdim=True)
        coef = torch.asinh(s / self.s) / (s / self.s).clamp_min(_eps(y))
        v = coef * y
        if torch.isnan(v).any() or torch.isinf(v).any():
            with torch.no_grad():
                def _stat(t):
                    t2 = torch.nan_to_num(t)
                    return float(t2.min().item()), float(t2.max().item()), float(t2.mean().item())
                print("[PV.log0][diag] v NaN/Inf; ||y|| stats=", _stat(s),
                      " asinh(arg) arg_max=", float((s/self.s).abs().max().item()))
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
        if max_norm is None: return torch.nan_to_num(x)
        r = x.norm(dim=-1, keepdim=True).clamp_min(_eps(x))
        return torch.nan_to_num(torch.clamp(max_norm / r, max=1.0) * x)

    def proj_tan0(self, v: torch.Tensor, c: float | None = None, max_norm: float = 20.0) -> torch.Tensor:
        if max_norm is None: return v
        r = v.norm(dim=-1, keepdim=True).clamp_min(_eps(v))
        return torch.clamp(max_norm / r, max=1.0) * v

    # ----- Compatibility aliases (legacy API) -----

    # Accept optional c to match older call sites but ignore it internally
    def expmap0(self, v: torch.Tensor, c: float | None = None) -> torch.Tensor:
        return self.exp0(v)

    def logmap0(self, y: torch.Tensor, c: float | None = None) -> torch.Tensor:
        return self.log0(y)
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Thm 4.3 / D.4.1 (Eq 643, 608, 634)
        w = Exp_p(v) ominus p = lambda_x * _sinhc(sqrt(c) * ||v||_p) * dpi_p[v] [cite: 634, 643]
        where lambda_x = (1+beta_x)/beta_x [cite: 705]
        """
        # 1. Compute g_p(v,v) and norm ||v||_p = r_g
        s2 = self.s * self.s
        x_norm2 = (x * x).sum(dim=-1, keepdim=True)
        v_norm2 = (v * v).sum(dim=-1, keepdim=True)
        dot_x_v = (x * v).sum(dim=-1, keepdim=True)

        # g_p(v,v) = ||v||^2 - <x,v>^2 / (s^2 + ||x||^2) [cite: 68, 538]
        g_pvv = (v_norm2 - (dot_x_v * dot_x_v) / (s2 + x_norm2)).clamp_min(TINY)
        r_g = torch.sqrt(g_pvv)  # ||v||_p
        
        # 2. Compute dpi_p[v] (Eq 608)
        # *** Fix: b_x is correct beta_x [cite: 112]
        b_x = beta(x, self.c)  # (..., 1) [cite: 112]

        coef1 = b_x / (1.0 + b_x)
        # dpi_p[v] formula [cite: 608]
        coef2 = -(self.c * (b_x ** 3)) / ((1.0 + b_x).clamp_min(TINY) ** 2)
        dpi_v = coef1 * v + coef2 * dot_x_v * x 

        # 3. Compute w = Exp_p(v) ominus p (simplified formula above)
        # lambda_x = (1+beta_x)/beta_x [cite: 705]
        lambda_x = (1.0 + b_x) / b_x.clamp_min(TINY)

        # _sinhc(z) = sinh(z)/z; clamp z=sqrt(c)*r_g to avoid overflow
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

    # ----- [Core] Log at x (general) [cite: 110, 671-709] -----
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

        # 2. Compute distance d(x,y)
        d_xy = self.dist(x, y)  # self.dist is numerically stable

        # 3. Compute sigma = d(x,y) / ||Z_pv|| [cite: 114, 707]
        sigma = d_xy / r_z_pv.clamp_min(TINY)
        
        # Handle x=y case (sigma -> 1)
        is_close = (r_z_pv_sq < TINY)
        sigma = torch.where(is_close, torch.ones_like(sigma), sigma)

        # 4. Compute tau_term [cite: 115, 707]
        # *** Fix: b_x is correct beta_x [cite: 112]
        b_x = beta(x, self.c)
        
        # τ_coef = (c * β_p / (1+β_p)) * σ [cite: 707]
        tau_coef = (self.c * b_x / (1.0 + b_x).clamp_min(TINY)) * sigma
        dot_x_z = (x * z_pv).sum(dim=-1, keepdim=True)
        tau_term = tau_coef * dot_x_z

        # 5. Assemble Log_p(q) = sigma * Z_pv + (tau_term) * p [cite: 707]
        out = sigma * z_pv + tau_term * x
        if torch.isnan(out).any() or torch.isinf(out).any():
            with torch.no_grad():
                print("[PV.log_map][diag] out NaN/Inf; ||z_pv|| max=", float(r_z_pv.max().item()),
                      " d(x,y) max=", float(d_xy.max().item()),
                      " sigma max=", float(sigma.max().item()),
                      " b_x min/max=", float(b_x.min().item()), float(b_x.max().item()))
        return out
    #def proj_tan0(self, v: torch.Tensor, c: float | None = None, max_norm: float = 20.0) -> torch.Tensor:
    #    return self.proj_tan(v, max_norm=max_norm)

    # Provide proj_tan alias for forward-compatibility
    def proj_tan(self, v: torch.Tensor, c: float | None = None, max_norm: float = 20.0) -> torch.Tensor:
        return self.proj_tan0(v, c=c, max_norm=max_norm)

# =====================================================================
#                        PV Multinomial Logistic Regression
# =====================================================================
class PVManifoldMLR(nn.Module):
    """
    PV unidirectional MLR aligned with HNN++ (PB) idea, but fully PV-closed-form.

    Params per class k:
        z_k ∈ R^d  (origin normal / direction)
        r_k ∈ R    (bias along geodesic in direction z_k)

    Score v_k(y) uses the PV-only closed form (★) above.
    """
    def __init__(self, c: float, in_features: int, num_classes: int):
        super().__init__()
        assert c > 0.0
        self.c = float(c)
        self.sqrtc = math.sqrt(self.c)
        self.d = in_features
        self.K = num_classes

        # trainables: z_k (K,d) and r_k (K,1)
        self.z = nn.Parameter(torch.empty(self.K, self.d))
        self.r = nn.Parameter(torch.empty(self.K, 1))
        self.reset_parameters()

    def reset_parameters(self):
        # small, isotropic init for stability; r near 0
        nn.init.normal_(self.z, mean=0.0, std=1e-2)
        nn.init.uniform_(self.r, a=-1e-3, b=1e-3)

    @torch.no_grad()
    def _norm_z(self) -> torch.Tensor:
        # ||z_k||, shape [K,1]
        return self.z.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: [B, d] in PV coordinates.
        returns logits: [B, K]
        """
        B, d = y.shape
        assert d == self.d

        c = self.c
        sqrtc = self.sqrtc

        # 0) Param self-check and minimal self-heal (only when NaN/Inf present)
        if torch.isnan(self.z).any() or torch.isinf(self.z).any():
            with torch.no_grad():
                self.z.data = torch.nan_to_num(self.z.data, nan=0.0)
        if torch.isnan(self.r).any() or torch.isinf(self.r).any():
            with torch.no_grad():
                self.r.data = torch.nan_to_num(self.r.data, nan=0.0)

        # precompute ||y|| and sqrt(1 + c ||y||^2), shape [B,1]
        y2 = (y * y).sum(dim=-1, keepdim=True)
        root_1_cy2 = torch.sqrt(1.0 + c * y2)

        # per-class norms ||z_k||, shapes [K,1]
        z = torch.nan_to_num(self.z)
        rz = self._norm_z()  # [K,1]

        # inner products <y, z_k>, shape [B,K]
        yz = y @ z.t()                        # [B,K]

        # broadcast scalars per class
        r_clean = torch.nan_to_num(self.r.squeeze(-1), nan=0.0)
        sr = sqrtc * r_clean                  # [K]    = sqrt(c)*r_k
        # Minimal invasive stability: clean NaN/Inf and clamp magnitude to avoid sinh/sech overflow
        if torch.isnan(sr).any() or torch.isinf(sr).any():
            with torch.no_grad():
                num_nan = int(torch.isnan(sr).sum().item())
                num_inf = int(torch.isinf(sr).sum().item())
                print("[PVMLR][param] sr contains NaN/Inf -> nan:", num_nan, " inf:", num_inf)
            sr = torch.nan_to_num(sr, nan=0.0, posinf=10.0, neginf=-10.0)
        sr = sr.clamp(-10.0, 10.0)
        sech_sr = _sech(sr)                   # [K]
        sinh_sr = torch.sinh(sr)              # [K]

        # term A: (2 - sech(sqrt(c) r_k)) <y, z_k>
        A = (2.0 - sech_sr).unsqueeze(0) * yz                     # [B,K]

        # term B: (sinh(sqrt(c) r_k)/sqrt(c)) ||z_k|| * sqrt(1 + c||y||^2)
        B = (sinh_sr / sqrtc).unsqueeze(0) * rz.t() * root_1_cy2  # [B,K]

        # bracket [...] in (★)
        bracket = A - B                                           # [B,K]

        # outer coefficients in (★)
        coef_outer = (rz.t() / sqrtc)                             # [1,K]
        coef_inner = (sqrtc / rz.t())                             # [1,K]

        # v_k(y) debug print (only on anomaly)
        arg = coef_inner * bracket
        # Minimal invasive: clamp asinh argument to avoid NaN
        arg = torch.clamp(arg, -1e6, 1e6)
        if torch.isnan(arg).any() or torch.isinf(arg).any() or (rz.min() <= 0):
            with torch.no_grad():
                def _stat(t):
                    t2 = torch.nan_to_num(t)
                    return (float(t2.min().item()), float(t2.max().item()), float(t2.mean().item()))
                print("[PVMLR][diag] arg stats=", _stat(arg), " | yz=", _stat(yz), " | rz=", _stat(rz),
                      " | root_1_cy2 max=", float(root_1_cy2.max().item()))
        v = coef_outer * torch.asinh(arg)        # [B,K]
        if torch.isnan(v).any() or torch.isinf(v).any():
            with torch.no_grad():
                print("[PVMLR][diag] v NaN/Inf; A/B max=", float(A.abs().max().item()), float(B.abs().max().item()),
                      " coef_outer max=", float(coef_outer.abs().max().item()))
        return v

# =====================================================================
#                           PV Fully Connected (FC)
# =====================================================================
class PVFC(nn.Module):
    """
    PV FC: y_k = (1/sqrt(c)) * sinh( sqrt(c) * v_k(x) ),  where v_k is the signed distance from PV_MLR.
    """
    def __init__(self, c: float, in_features: int, out_features: int, use_bias: bool = True, inner_act: str = 'none'):
        super().__init__()
        self.c = float(c)
        self.sqrtc = math.sqrt(self.c)
        self.mlr = PVManifoldMLR(c, in_features, out_features)
        self.use_bias = use_bias
        self.M = PVManifold(self.c)
        self.bias = nn.Parameter(torch.zeros(out_features)) if use_bias else None
        self.inner_act = inner_act.lower() if isinstance(inner_act, str) else 'none'

    def _activate_v(self, v: torch.Tensor) -> torch.Tensor:
        if self.inner_act == 'relu':
            return F.relu(v)
        if self.inner_act == 'tanh':
            return torch.tanh(v)
        if self.inner_act == 'softplus':
            return F.softplus(v, beta=1, threshold=20.)
        return v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.mlr(x)  # [B, m], signed distances in PV (units of length)
        # Apply inner_act to v_k (per user semantics)
        v = self._activate_v(v)
        z = self.sqrtc * v
        # Minimal invasive stability: clamp sinh input to avoid overflow
        z = torch.clamp(z, -20.0, 20.0)
        # Debug print: only when NaN/Inf or extreme magnitude
        if torch.isnan(v).any() or torch.isinf(v).any() or torch.isnan(z).any() or torch.isinf(z).any() or (z.abs().max() > 1e6):
            with torch.no_grad():
                def _stat(t):
                    t2 = torch.nan_to_num(t)
                    return (float(t2.min().item()), float(t2.max().item()), float(t2.mean().item()))
                print("[PVFC][diag] v stats min/max/mean=", _stat(v), " | z stats=", _stat(z))
        y = (1.0 / self.sqrtc) * torch.sinh(z)
        # Euclidean bias (tangent space) -> exp to manifold then gyro-add for PVFC bias
        if self.use_bias and self.bias is not None:
            b_tan = self.bias.unsqueeze(0).to(x.device, dtype=x.dtype)
            b_hyp = self.M.exp0(b_tan)
            y = self.M.gyro_add(y, b_hyp)
        return y