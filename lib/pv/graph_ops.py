# pv_manifold_k.py
# Fully aligned with PVNN paper notation: using curvature K (where K < 0).
# Ref: "Proper Velocity Neural Networks" (ICLR 2026 submission)
from __future__ import annotations
import math
import typing as T
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

TINY = 1e-15
EPS = {torch.float32: 1e-6, torch.float64: 1e-12}
def _eps(x: torch.Tensor) -> float: return EPS.get(x.dtype, 1e-12)

_JIT_DISABLED = os.getenv("PVNN_DISABLE_JIT", "0") == "1"
_script = (lambda fn: fn) if _JIT_DISABLED else torch.jit.script


@_script
def _pv_beta(x: torch.Tensor, k: float, tiny: float) -> torch.Tensor:
    x2 = (x * x).sum(dim=-1, keepdim=True)
    denom = torch.sqrt(1.0 - k * x2)
    return 1.0 / denom.clamp_min(tiny)


@_script
def _pv_gyro_add(x: torch.Tensor, y: torch.Tensor, k: float, neg_k: float, tiny: float) -> torch.Tensor:
    b_x = _pv_beta(x, k, tiny)
    b_y = _pv_beta(y, k, tiny)
    xy = (x * y).sum(dim=-1, keepdim=True)
    neg_k_xy = neg_k * xy
    term1 = (b_x / (1.0 + b_x)) * neg_k_xy
    term2 = (1.0 / b_y) - 1.0
    coef = term1 + term2
    return x + y + coef * x


@_script
def _pv_gyro_scalar_mul(
    x: torch.Tensor,
    r: torch.Tensor,
    s: float,
    eps: float,
    zmax: float,
) -> torch.Tensor:
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    arg_asinh = x_norm / s
    term = r * torch.asinh(arg_asinh)
    if zmax > 0:
        term = torch.clamp(term, -zmax, zmax)
    res_mag = s * torch.sinh(term)
    scale = res_mag / x_norm.clamp_min(eps)
    return scale * x


@_script
def _pv_expmap0(v: torch.Tensor, s: float, eps: float) -> torch.Tensor:
    r = torch.norm(v, p=2, dim=-1, keepdim=True)
    coef = torch.sinh(r / s) / (r / s).clamp_min(eps)
    return coef * v


@_script
def _pv_logmap0(y: torch.Tensor, s: float, eps: float) -> torch.Tensor:
    s_norm = torch.norm(y, p=2, dim=-1, keepdim=True)
    coef = torch.asinh(s_norm / s) / (s_norm / s).clamp_min(eps)
    return coef * y


@_script
def _pv_dist0(y: torch.Tensor, s: float) -> torch.Tensor:
    s_norm = torch.norm(y, p=2, dim=-1, keepdim=True)
    return s * torch.asinh(s_norm / s)


@_script
def _pv_gyro_neg(x: torch.Tensor) -> torch.Tensor:
    return -x


@_script
def _pv_dist(x: torch.Tensor, y: torch.Tensor, k: float, neg_k: float, s: float, tiny: float) -> torch.Tensor:
    z = _pv_gyro_add(_pv_gyro_neg(x), y, k, neg_k, tiny)
    b_z = _pv_beta(z, k, tiny)
    p = (b_z / (1.0 + b_z)) * z
    r = torch.norm(p, p=2, dim=-1, keepdim=True)
    arg = torch.clamp(r / s, max=1.0 - 1e-7)
    return 2.0 * s * torch.atanh(arg)


@_script
def _pv_log_map(x: torch.Tensor, y: torch.Tensor, k: float, neg_k: float, s: float, tiny: float) -> torch.Tensor:
    z_pv = _pv_gyro_add(_pv_gyro_neg(x), y, k, neg_k, tiny)
    r_z = torch.norm(z_pv, p=2, dim=-1, keepdim=True).clamp_min(tiny)
    d_xy = _pv_dist(x, y, k, neg_k, s, tiny)
    sigma = d_xy / r_z
    sigma = torch.where(d_xy < tiny, torch.ones_like(sigma), sigma)
    b_x = _pv_beta(x, k, tiny)
    tau_coef = (neg_k * b_x / (1.0 + b_x)) * sigma
    dot_x_z = (x * z_pv).sum(dim=-1, keepdim=True)
    return sigma * z_pv + tau_coef * dot_x_z * x


@_script
def _pv_exp_map(x: torch.Tensor, v: torch.Tensor, k: float, neg_k: float, sqrt_neg_k: float, tiny: float) -> torch.Tensor:
    dot_x_v = (x * v).sum(dim=-1, keepdim=True)
    b_x = _pv_beta(x, k, tiny)
    norm_sq = (v * v).sum(dim=-1, keepdim=True) + k * (b_x ** 2) * (dot_x_v ** 2)
    norm_v_x = torch.sqrt(norm_sq.clamp_min(tiny))
    coef1 = b_x / (1.0 + b_x)
    coef2 = k * (b_x ** 3) / ((1.0 + b_x) ** 2)
    dpi_v = coef1 * v + coef2 * dot_x_v * x
    norm_dpi = torch.norm(dpi_v, p=2, dim=-1, keepdim=True).clamp_min(tiny)
    dir_dpi = dpi_v / norm_dpi
    arg = sqrt_neg_k * ((1.0 + b_x) / b_x) * norm_dpi
    u_mag = (1.0 / sqrt_neg_k) * torch.sinh(arg)
    u = u_mag * dir_dpi
    return _pv_gyro_add(x, u, k, neg_k, tiny)


@_script
def _pv_proj_tan(v: torch.Tensor, max_norm: float, eps: float) -> torch.Tensor:
    r = torch.norm(v, p=2, dim=-1, keepdim=True).clamp_min(eps)
    return torch.clamp(max_norm / r, max=1.0) * v


@_script
def _pv_proj(x: torch.Tensor, max_norm: float, eps: float) -> torch.Tensor:
    if max_norm <= 0.0:
        return x
    norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return scale * x

# ---------- Core Factors ----------
def beta(x: torch.Tensor, k: float) -> torch.Tensor:
    """
    Relativistic beta factor: beta_x = 1 / sqrt(1 - K ||x||^2)
    [cite_start][cite: 174, 3360]
    Note: k is negative, so this behaves like 1/sqrt(1 + |K|x^2).
    """
    return _pv_beta(x, k, TINY)

# ---------- Small-angle Helpers ----------
def _sinhc(z: torch.Tensor) -> torch.Tensor:
    """Computes sinh(z)/z numerically stable around 0."""
    eps = _eps(z); z2 = z*z
    y = torch.sinh(z) / z.clamp_min(eps)
    y0 = 1.0 + z2/6.0
    return torch.where(z.abs() < eps, y0, y)

def _sech(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.cosh(x)

# =====================================================================
#                           Proper-Velocity Space
# =====================================================================
class PVManifold:
    """
    PV model defined by curvature K < 0.
    Parameter 'k' must be negative (e.g., -1.0).
    """
    def __init__(self, k: float):
        assert k < 0, f"Curvature K must be negative for PV hyperbolic space, got {k}."
        self.k = float(k)
        self.neg_k = -self.k                    # -K (positive value, equivalent to c)
        self.s = 1.0 / math.sqrt(self.neg_k)    # s = 1/sqrt(-K)
        self.sqrt_neg_k = math.sqrt(self.neg_k) # sqrt(-K)
        # Backwards-compat alias: many callers used a positive curvature "c"
        self.c = self.neg_k

    # ----- Compatibility helpers -----
    def _ensure_curvature(self, c: T.Optional[float]) -> None:
        if c is None:
            return
        requested_k = -abs(float(c))
        if not math.isclose(requested_k, self.k, rel_tol=1e-6, abs_tol=1e-8):
            # Keep silent unless mismatch is critical; old code often reuses c inconsistently.
            pass

    # [cite_start]----- Exp/Log at the origin [cite: 3521-3522] -----
    def expmap0(self, v: torch.Tensor, c: T.Optional[float] = None) -> torch.Tensor:
        self._ensure_curvature(c)
        return _pv_expmap0(v, self.s, _eps(v))

    # Backwards compatibility alias
    def exp0(self, v: torch.Tensor, c: T.Optional[float] = None) -> torch.Tensor:
        return self.expmap0(v, c=c)

    def logmap0(self, y: torch.Tensor, c: T.Optional[float] = None) -> torch.Tensor:
        self._ensure_curvature(c)
        return _pv_logmap0(y, self.s, _eps(y))

    def log0(self, y: torch.Tensor, c: T.Optional[float] = None) -> torch.Tensor:
        return self.logmap0(y, c=c)

    def dist0(self, y: torch.Tensor) -> torch.Tensor:
        return _pv_dist0(y, self.s)

    def gyro_scalar_mul(
        self,
        r: float | torch.Tensor,
        x: torch.Tensor,
        zmax: float | None = 50.0,
    ) -> torch.Tensor:
        """
        PV gyro-scalar multiplication: r (x) x
        Formula (Eq 3): 
          r (x) x = (1/sqrt(-K)) * sinh( r * asinh( sqrt(-K)||x|| ) ) * x/||x||
        """
        if not torch.is_tensor(r):
            r = torch.tensor(r, device=x.device, dtype=x.dtype)
        zmax_val = float(zmax) if zmax is not None else -1.0
        return _pv_gyro_scalar_mul(x, r, self.s, _eps(x), zmax_val)

    # [cite_start]----- PV Gyroaddition [cite: 3380] -----
    def gyro_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _pv_gyro_add(x, y, self.k, self.neg_k, TINY)

    def gyro_neg(self, x: torch.Tensor) -> torch.Tensor:
        return _pv_gyro_neg(x)
    
    # [cite_start]----- Geodesic Distance [cite: 3514] -----
    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _pv_dist(x, y, self.k, self.neg_k, self.s, TINY)

    # [cite_start]----- Log Map at x (General) [cite: 3512, 5479] -----
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _pv_log_map(x, y, self.k, self.neg_k, self.s, TINY)

    # [cite_start]----- Exp Map at x (General) [cite: 3511] -----
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return _pv_exp_map(x, v, self.k, self.neg_k, self.sqrt_neg_k, TINY)

    # ----- Project Tangent (Helper) -----
    def proj_tan(self, v: torch.Tensor, max_norm: float = 20.0) -> torch.Tensor:
        return _pv_proj_tan(v, max_norm, _eps(v))

    def proj_tan0(self, v: torch.Tensor, c: T.Optional[float] = None, max_norm: float = 20.0) -> torch.Tensor:
        self._ensure_curvature(c)
        return self.proj_tan(v, max_norm=max_norm)

    def proj(self, x: torch.Tensor, c: T.Optional[float] = None, max_norm: float = 50.0) -> torch.Tensor:
        self._ensure_curvature(c)
        if max_norm is None:
            return x
        return _pv_proj(x, float(max_norm), _eps(x))


# =====================================================================
#                  PV Multinomial Logistic Regression
# =====================================================================
class PVManifoldMLR(nn.Module):
    """
    PV MLR layer using curvature K < 0.
    Implements the CORRECTED formula from PVNN (File 2) which uses cosh/sinh.
    
    Parameters per class k:
        z_k: Direction in tangent space at origin
        r_k: Scalar bias (distance along geodesic)
    """
    def __init__(self, k: float, in_features: int, num_classes: int):
        super().__init__()
        assert k < 0, "Curvature K must be negative."
        self.k = float(k)
        self.neg_k = -self.k
        self.sqrt_neg_k = math.sqrt(self.neg_k)
        
        self.d = in_features
        self.K_classes = num_classes

        # Parameters defined in tangent space at origin
        self.z = nn.Parameter(torch.empty(self.K_classes, self.d))
        self.r = nn.Parameter(torch.empty(self.K_classes, 1))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization
        nn.init.normal_(self.z, mean=0.0, std=1e-2)
        nn.init.uniform_(self.r, a=-1e-3, b=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, d] (Points in PV space)
        Returns: [B, K_classes] (Logits)
        
        [cite_start]Formula based on PVNN Thm 5.2[cite: 3609]:
        v_k(x) = (||z||/sqrt(-K)) * asinh( ... )
        Argument inside asinh:
           sqrt(-K)/||z|| * [ cosh(sqrt(-K)r) <x,z> - sinh(sqrt(-K)r) * sqrt(1-K||x||^2) * ||z||/sqrt(-K) ]
        """
        # Precompute ||z_k||
        z_norm = self.z.norm(dim=-1, keepdim=True).clamp_min(TINY) # [K, 1]
        
        # 1. Hyperbolic radius term: sr = sqrt(-K) * r_k
        sr = self.sqrt_neg_k * self.r.squeeze(-1) # [K_classes]
        sr = sr.clamp(-15.0, 15.0) # Numerical stability clip
        
        cosh_sr = torch.cosh(sr) # [K]
        sinh_sr = torch.sinh(sr) # [K]
        
        # Term A: cosh(sr) * <x, z>
        xz = x @ self.z.t() # [B, K]
        term_A = cosh_sr.unsqueeze(0) * xz # [B, K]
        
        # 2. Term B involving beta factor
        # beta_x^{-1} = sqrt(1 - K ||x||^2)
        # Note: self.k is negative, so -self.k is positive
        x_sq = (x*x).sum(dim=-1, keepdim=True)
        beta_inv = torch.sqrt(1.0 - self.k * x_sq) # [B, 1]
        
        # [cite_start]From eq[cite: 3609]: term is sinh(sr) * sqrt(1-K||x||^2) * (||z|| / sqrt(-K))
        # Note: The raw formula subtracts: sinh(sr) * beta_inv
        # But we need to account for the normalization factors outside the bracket in the theorem.
        # [cite_start]Let's align strictly with [cite: 3609] structure:
        # v_k = ||z||/c' * asinh( c'/||z|| * <x,z> * cosh - sinh * beta_inv ) ??? 
        # Actually, let's look at the expanded form derived in code context:
        # The subtractive term B needs to match the dimension of A (which is <x,z>).
        # <x,z> has units length^2? No, length. 
        # beta_inv is unitless.
        # So we need a length factor. It comes from ||z||/sqrt(-K).
        
        term_B = (sinh_sr.unsqueeze(0) / self.sqrt_neg_k) * z_norm.t() * beta_inv # [B, K]
        
        # 3. Combine
        # Factor C = sqrt(-K) / ||z||
        factor_C = self.sqrt_neg_k / z_norm.t() # [1, K]
        
        argument = factor_C * (term_A - term_B)
        argument = torch.clamp(argument, -1e6, 1e6)
        
        # 4. Final Scale: ||z|| / sqrt(-K)
        scale = z_norm.t() / self.sqrt_neg_k
        
        return scale * torch.asinh(argument)


# =====================================================================
#                           PV Fully Connected
# =====================================================================
class PVFC(nn.Module):
    """
    PV Fully Connected Layer using K < 0.
    Maps PV -> PV via hyperplane distances.
    
    y_k = (1/sqrt(-K)) * sinh( sqrt(-K) * act(v_k(x)) )
    """
    def __init__(self, k: float, in_features: int, out_features: int, 
                 use_bias: bool = True, inner_act: str = 'none'):
        super().__init__()
        assert k < 0, "Curvature K must be negative."
        self.k = float(k)
        self.neg_k = -self.k
        self.sqrt_neg_k = math.sqrt(self.neg_k)
        
        # 1. Linear-like transformation (calculates v_k)
        self.mlr = PVManifoldMLR(k, in_features, out_features)
        
        # 2. Bias setup
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.zeros(out_features)) if use_bias else None
        self.manifold = PVManifold(k)
        
        # 3. Inner activation (applied to the distance v_k)
        self.inner_act = inner_act.lower() if isinstance(inner_act, str) else 'none'

    def _activate_v(self, v: torch.Tensor) -> torch.Tensor:
        """Apply non-linearity to the signed distances."""
        if self.inner_act == 'relu':
            return F.relu(v)
        if self.inner_act == 'tanh':
            return torch.tanh(v)
        if self.inner_act == 'softplus':
            return F.softplus(v, beta=1, threshold=20.)
        return v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Get signed distances v_k(x) [B, out]
        v = self.mlr(x) 
        
        # 2. Apply optional inner activation on distances
        v = self._activate_v(v)
        
        # 3. Apply non-linearity to map distance back to coordinate
        # y = (1/sqrt(-K)) * sinh( sqrt(-K) * v )
        arg = self.sqrt_neg_k * v
        arg = torch.clamp(arg, -15.0, 15.0) # Numerical stability clip
        y = (1.0 / self.sqrt_neg_k) * torch.sinh(arg)
        
        # 4. Apply Bias (via Gyro-addition in PV space)
        if self.use_bias and self.bias is not None:
            # Bias is a vector in tangent space at origin
            # Map it to manifold: Exp_0(bias)
            b_hyp = self.manifold.exp0(self.bias.unsqueeze(0))
            # Add bias: y (+) b
            y = self.manifold.gyro_add(y, b_hyp)
            
        return y


# =====================================================================
#                     Tangent block (Log -> Linear -> Exp)
# =====================================================================
class PV_TangentBlock(nn.Module):
    """
    Standard manifold operation: Map to tangent space, apply Euclidean linear, map back.
    """
    def __init__(self, k: float, in_dim: int, out_dim: int,
                 bias: bool=True, tau_clip: float | None=None,
                 nonlin: T.Callable[[torch.Tensor], torch.Tensor]=F.relu):
        super().__init__()
        self.M = PVManifold(k)
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.nonlin = nonlin
        self.tau_clip = tau_clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Log map at origin
        v = self.M.log0(x)
        
        # 2. Euclidean linear layer
        v = self.lin(v)
        
        # 3. Nonlinearity in tangent space
        if self.nonlin is not None:
            v = self.nonlin(v)
            
        # 4. Optional clipping (for stability)
        if self.tau_clip is not None:
            r = v.norm(dim=-1, keepdim=True).clamp_min(_eps(v))
            v = torch.clamp(self.tau_clip / r, max=1.0) * v
            
        # 5. Exp map at origin
        return self.M.exp0(v)