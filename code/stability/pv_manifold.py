# pv_manifold_k.py
# Fully aligned with PVNN paper notation: using curvature K (where K < 0).
# Ref: "Proper Velocity Neural Networks" (ICLR 2026 submission)
from __future__ import annotations
import math
import typing as T
import torch
import torch.nn as nn
import torch.nn.functional as F

TINY = 1e-15
EPS = {torch.float32: 1e-6, torch.float64: 1e-12}
def _eps(x: torch.Tensor) -> float: return EPS.get(x.dtype, 1e-12)

# ---------- Core Factors ----------
def beta(x: torch.Tensor, k: float) -> torch.Tensor:
    """
    Relativistic beta factor: beta_x = 1 / sqrt(1 - K ||x||^2)
    [cite_start][cite: 174, 3360]
    Note: k is negative, so this behaves like 1/sqrt(1 + |K|x^2).
    """
    # x^2 sum
    x2 = (x * x).sum(dim=-1, keepdim=True)
    # 1 - k * x^2
    denom = torch.sqrt(1.0 - k * x2)
    return 1.0 / denom.clamp_min(TINY)

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
        # Exp_0(v) = (1/sqrt(-K)) * sinh(sqrt(-K)||v||) * v/||v||
        r = v.norm(dim=-1, keepdim=True)
        # s * sinh(r/s) is equivalent to (1/sqrt(-K)) * sinh(sqrt(-K)*r)
        coef = torch.sinh(r / self.s) / (r / self.s).clamp_min(_eps(v))
        return coef * v

    # Backwards compatibility alias
    def exp0(self, v: torch.Tensor, c: T.Optional[float] = None) -> torch.Tensor:
        return self.expmap0(v, c=c)

    def logmap0(self, y: torch.Tensor, c: T.Optional[float] = None) -> torch.Tensor:
        self._ensure_curvature(c)
        # Log_0(y) = (1/sqrt(-K)) * asinh(sqrt(-K)||y||) * y/||y||
        s_norm = y.norm(dim=-1, keepdim=True)
        coef = torch.asinh(s_norm / self.s) / (s_norm / self.s).clamp_min(_eps(y))
        return coef * y

    def log0(self, y: torch.Tensor, c: T.Optional[float] = None) -> torch.Tensor:
        return self.logmap0(y, c=c)

    def dist0(self, y: torch.Tensor) -> torch.Tensor:
        # d(0,y) = (1/sqrt(-K)) * asinh(sqrt(-K)||y||)
        s_norm = y.norm(dim=-1, keepdim=True)
        return self.s * torch.asinh(s_norm / self.s)

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
        # Ensure r is a tensor for broadcasting
        if not torch.is_tensor(r):
            r = torch.tensor(r, device=x.device, dtype=x.dtype)
            
        x_norm = x.norm(dim=-1, keepdim=True)
        
        # Argument for sinh: r * asinh( ||x|| / s )
        # Note: s = 1/sqrt(-K), so ||x||/s = sqrt(-K)||x||
        arg_asinh = x_norm / self.s
        term = r * torch.asinh(arg_asinh)
        
        # Optional clamp for numerical safety
        if zmax is not None:
            zmax = float(zmax)
            term = torch.clamp(term, -zmax, zmax)
        
        # Result magnitude: s * sinh(...)
        res_mag = self.s * torch.sinh(term)
        
        # Apply direction: res_mag * (x / x_norm)
        # Use safe division pattern: (res_mag / x_norm) * x
        scale = res_mag / x_norm.clamp_min(_eps(x))
        
        return scale * x

    # [cite_start]----- PV Gyroaddition [cite: 3380] -----
    def gyro_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Eq (2): x (+) y = x + y + { (1-beta_y)/beta_y - K * beta_x/(1+beta_x) * <x,y> } x
        b_x = beta(x, self.k)
        b_y = beta(y, self.k)
        xy  = (x * y).sum(dim=-1, keepdim=True)
        
        # Term: - K * <x,y> (Note: self.k is negative, so -self.k is positive)
        # Using variable name 'neg_k_xy' for clarity: (-K)*<x,y>
        neg_k_xy = self.neg_k * xy 
        
        # Coefficient calculation
        # Term 1: beta_x / (1+beta_x) * (-K <x,y>)
        term1 = (b_x / (1.0 + b_x)) * neg_k_xy
        # Term 2: (1 - beta_y) / beta_y  =  1/beta_y - 1
        term2 = (1.0 / b_y) - 1.0
        
        coef = term1 + term2
        return x + y + coef * x

    def gyro_neg(self, x: torch.Tensor) -> torch.Tensor:
        return -x
    
    # [cite_start]----- Geodesic Distance [cite: 3514] -----
    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Thm 4.3: d(x,y) = (2/sqrt(-K)) * atanh( sqrt(-K) || pi( (-x)+y ) || )
        # where pi(z) maps PV to Poincare ball.
        z = self.gyro_add(self.gyro_neg(x), y)
        
        # [cite_start]pi(z) = (beta_z / (1+beta_z)) * z  [cite: 3431]
        b_z = beta(z, self.k)
        p = (b_z / (1.0 + b_z)) * z
        
        r = p.norm(dim=-1, keepdim=True)
        # argument for atanh is sqrt(-K) * ||p|| = r / s
        arg = torch.clamp(r / self.s, max=1.0 - 1e-7)
        return 2.0 * self.s * torch.atanh(arg)

    # [cite_start]----- Log Map at x (General) [cite: 3512, 5479] -----
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Log_x(y) using Thm 4.3 formulas.
        """
        # 1. Z = (-x) (+) y
        z_pv = self.gyro_add(self.gyro_neg(x), y)
        r_z = z_pv.norm(dim=-1, keepdim=True).clamp_min(TINY)

        # 2. Distance d(x,y)
        d_xy = self.dist(x, y)

        # 3. sigma = d(x,y) / ||Z||
        sigma = d_xy / r_z
        # Handle x ~= y case
        sigma = torch.where(d_xy < TINY, torch.ones_like(sigma), sigma)

        # 4. tau term
        # From PVNN derivation: tau = ( -K * beta_x / (1+beta_x) ) * sigma * <x, Z>
        b_x = beta(x, self.k)
        
        # Coeff: -K * beta / (1+beta)
        tau_coef = (self.neg_k * b_x / (1.0 + b_x)) * sigma
        dot_x_z = (x * z_pv).sum(dim=-1, keepdim=True)
        
        return sigma * z_pv + tau_coef * dot_x_z * x

    # [cite_start]----- Exp Map at x (General) [cite: 3511] -----
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exp_x(v) using Thm 4.3 / Lem 4.1
        """
        # [cite_start]1. PV metric norm ||v||_x [cite: 3381]
        # g_x(u,v) = <u,v> + K * beta_x^2 * <x,u>*<x,v>  (Eq 1)
        # Note self.k is negative, so +K term is subtractive (correct for PV).
        dot_x_v = (x * v).sum(dim=-1, keepdim=True)
        b_x = beta(x, self.k)
        
        norm_sq = (v*v).sum(dim=-1, keepdim=True) + self.k * (b_x**2) * (dot_x_v**2)
        # Numerical guard: metric should be positive definite
        norm_v_x = torch.sqrt(norm_sq.clamp_min(TINY))

        # [cite_start]2. Differential d_pi_x(v) [cite: 3436]
        # d_pi(v) = beta/(1+beta) v + K * beta^3/(1+beta)^2 * <x,v> x
        # Note: K is used directly (negative)
        coef1 = b_x / (1.0 + b_x)
        coef2 = self.k * (b_x**3) / ((1.0 + b_x)**2)
        dpi_v = coef1 * v + coef2 * dot_x_v * x
        
        norm_dpi = dpi_v.norm(dim=-1, keepdim=True).clamp_min(TINY)
        dir_dpi = dpi_v / norm_dpi
        
        # [cite_start]3. Apply formula from Thm 4.3 [cite: 3511]
        # w = Exp_0( ... ) logic 
        # Arg for sinh: sqrt(-K) * (1+beta)/beta * ||dpi(v)||
        # Note: formula says sinh( sqrt(-K)*(1+beta)/beta * ||d_pi(v)|| )
        arg = self.sqrt_neg_k * ((1.0 + b_x)/b_x) * norm_dpi
        
        # Displacement vector magnitude: (1/sqrt(-K)) * sinh(...)
        u_mag = (1.0 / self.sqrt_neg_k) * torch.sinh(arg)
        u = u_mag * dir_dpi
        
        # Final result: x (+) u
        return self.gyro_add(x, u)

    # ----- Project Tangent (Helper) -----
    def proj_tan(self, v: torch.Tensor, max_norm: float = 20.0) -> torch.Tensor:
        r = v.norm(dim=-1, keepdim=True).clamp_min(_eps(v))
        return torch.clamp(max_norm / r, max=1.0) * v

    def proj_tan0(self, v: torch.Tensor, c: T.Optional[float] = None, max_norm: float = 20.0) -> torch.Tensor:
        self._ensure_curvature(c)
        return self.proj_tan(v, max_norm=max_norm)

    def proj(self, x: torch.Tensor, c: T.Optional[float] = None, max_norm: float = 50.0) -> torch.Tensor:
        """
        Lightweight projection used for compatibility with older code paths.
        Simply rescales points whose norm exceeds `max_norm`.
        """
        self._ensure_curvature(c)
        if max_norm is None or max_norm <= 0:
            return x
        norm = x.norm(dim=-1, keepdim=True).clamp_min(_eps(x))
        scale = torch.clamp(max_norm / norm, max=1.0)
        return scale * x


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