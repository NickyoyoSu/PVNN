import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# From PV_monifold.py
PV_DEBUG = os.environ.get("PV_DEBUG") == "1"
PV_SAFETY_LIMIT = 10.0
TINY = 1e-15
EPS = {torch.float32: 1e-6, torch.float64: 1e-12}
def _eps(x: torch.Tensor) -> float: return EPS.get(x.dtype, 1e-12)

# ---------- core factors ----------
def beta(x: torch.Tensor, c: float) -> torch.Tensor:
    denom = torch.sqrt(1.0 + c * (x * x).sum(dim=-1, keepdim=True))
    return 1.0 / denom.clamp_min(TINY)

def _sech(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.cosh(x)


def gamma(x: torch.Tensor, c: float) -> torch.Tensor:
    denom = torch.sqrt(1.0 - c * (x * x).sum(dim=-1, keepdim=True))
    return 1.0 / denom.clamp_min(TINY)

def _sinhc(z: torch.Tensor) -> torch.Tensor:
    eps = _eps(z); z2 = z*z
    y = torch.sinh(z) / z.clamp_min(eps)
    y0 = 1.0 + z2/6.0
    return torch.where(z.abs() < eps, y0, y)

def acosh(z: torch.Tensor, eps: float = TINY) -> torch.Tensor:
    z = torch.clamp(z, min=1. + eps)
    return torch.log(z + torch.sqrt(z * z - 1))

def sigma(u: torch.Tensor, v: torch.Tensor, kappa: float, eps: float = TINY) -> torch.Tensor:
    dot = (u * v).sum(-1, keepdim=True)
    base = (v * v).sum(-1, keepdim=True) - dot ** 2 / (1 + (u * u).sum(-1, keepdim=True))
    return torch.sqrt(torch.clamp(base, min=eps)) / kappa

# Additional helpers
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
        sigma = d_xy / r_z_pv.clamp_min(TINY)

        # Handle the degenerate case x==y (sigma -> 1)
        is_close = (r_z_pv_sq < TINY)
        sigma = torch.where(is_close, torch.ones_like(sigma), sigma)

        # 4. Compute τ_term [cite: 115, 707]
        # *** Fix: ensure b_x uses the correct beta_x [cite: 112]
        b_x = beta(x, self.c)

        # τ_coef = (c * β_p / (1+β_p)) * σ [cite: 707]
        tau_coef = (self.c * b_x / (1.0 + b_x).clamp_min(TINY)) * sigma
        dot_x_z = (x * z_pv).sum(dim=-1, keepdim=True)
        tau_term = tau_coef * dot_x_z

        # 5. Assemble Log_p(q) = σ * Z_pv + (τ_term) * p [cite: 707]
        out = sigma * z_pv + tau_term * x
        if PV_DEBUG and (torch.isnan(out).any() or torch.isinf(out).any()):
            with torch.no_grad():
                print("[PV.log_map][diag] NaN/Inf -> ||z_pv|| max=", float(r_z_pv.max().item()),
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

        # 0) Minimal self-healing for NaN/Inf parameters
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
        # Clamp sr to keep sinh/sech stable and scrub NaN/Inf
        if torch.isnan(sr).any() or torch.isinf(sr).any():
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

        # Debug printing for v_k(y) (only triggered when anomalies occur)
        arg = coef_inner * bracket
        # Clamp asinh arguments to keep outputs finite
        arg = torch.clamp(arg, -1e6, 1e6)
        if torch.isnan(arg).any() or torch.isinf(arg).any() or (rz.min() <= 0):
            if PV_DEBUG:
                with torch.no_grad():
                    def _stat(t):
                        t2 = torch.nan_to_num(t)
                        return float(t2.min().item()), float(t2.max().item()), float(t2.mean().item())
                    print("[PVMLR][diag] arg NaN/Inf -> arg stats=", _stat(arg),
                          " yz stats=", _stat(yz), " rz min=", float(rz.min().item()),
                          " root_1_cy2 max=", float(root_1_cy2.max().item()))
            arg = torch.nan_to_num(arg, nan=0.0, posinf=1e6, neginf=-1e6)
        v = coef_outer * torch.asinh(arg)        # [B,K]
        if torch.isnan(v).any() or torch.isinf(v).any():
            v = torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
        return v


# From Lorentz_monifold.py
def arcosh(x):
    z = torch.sqrt(torch.clamp_min(x.pow(2) - 1.0, 1e-7))
    return torch.log(x + z)

class LorentzManifold(nn.Module):
    def __init__(self, k=-1.0, learnable=False):  # default to negative curvature
        super().__init__()
        k = abs(k)
        if learnable:
            self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32))
        else:
            self.k = torch.tensor(k, dtype=torch.float32)
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def expmap0(self, u, c=None):
        c_val = c if c is not None else self.k
        K = 1. / c_val
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * torch.cosh(theta)
        res[:, 1:] = sqrtK * torch.sinh(theta) * x / x_norm
        return self.proj(res)

    def proj(self, x):
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.sum(y**2, dim=-1, keepdim=True)

        time_component = torch.sqrt(1.0 / self.k + y_sqnorm)
        return torch.cat([time_component, y], dim=-1)

    def to_lorentz(self, x):
        x_norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
        time_component = torch.sqrt(1.0 / self.k + x_norm_sq)
        x_lorentz = torch.cat([time_component, x], dim=1)
        return self.proj(x_lorentz)

class LorentzManifoldMLR(nn.Module):
    def __init__(self, manifold, in_features, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold
        self.a = nn.Parameter(torch.zeros(num_classes))
        self.z = nn.Parameter(F.pad(torch.zeros(num_classes, in_features-1), (1,0), value=1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_classes))
        stdv = 1. / math.sqrt(self.z.size(1))
        nn.init.uniform_(self.z, -stdv, stdv)
        nn.init.uniform_(self.a, -stdv, stdv)
        if bias:
            nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x):
        # Ensure the input is expressed in Lorentz coordinates
        x_lorentz = self.manifold.to_lorentz(x)

        # Hyperplane parameterization
        c_tensor = torch.tensor(self.manifold.c, device=x.device, dtype=x.dtype)
        sqrt_mK = 1/torch.sqrt(c_tensor)
        norm_z = torch.norm(self.z, dim=-1)
        w_t = torch.sinh(sqrt_mK*self.a)*norm_z
        w_s = torch.cosh(sqrt_mK*self.a.view(-1,1))*self.z
        beta = torch.sqrt(torch.clamp(-w_t**2+torch.norm(w_s, dim=-1)**2, min=1e-15))

        # Compute class-wise distances in a single pass
        alpha = -w_t*x_lorentz[:,0:1] + torch.matmul(x_lorentz[:,1:], w_s.transpose(0,1))
        d = torch.sqrt(c_tensor)*torch.abs(torch.asinh(sqrt_mK*alpha/beta.view(1,-1)))

        # Signed distance
        logits = torch.sign(alpha)*beta.view(1,-1)*d

        # Apply bias if available
        if self.has_bias:
            logits = logits + self.bias

        return logits

# From HNN_layers.py
class PoincareBall(nn.Module):
    def __init__(self, k=-1.0, learnable=False):  # default to negative curvature
        super().__init__()
        k = abs(k)  # always operate on |k|
        if learnable:
            self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32))
        else:
            self.k = torch.tensor(k, dtype=torch.float32)
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def expmap0(self, u):
        sqrt_k = torch.sqrt(self.k)
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = torch.tanh(sqrt_k * u_norm) * u / (sqrt_k * u_norm)
        return gamma_1

    def projx(self, x):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / torch.sqrt(self.k)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

class HyperbolicMLR(nn.Module):
    def __init__(self, manifold, in_features, num_classes):
        super().__init__()
        self.manifold = manifold
        self.weight = nn.Parameter(torch.empty(in_features, num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        std = (in_features ** -0.5)
        nn.init.normal_(self.weight, 0, std)
        nn.init.zeros_(self.bias)


    def forward(self, x):
        print("[DEBUG] HyperbolicMLR input x has nan:", torch.isnan(x).any().item(), "range:", x.min().item(), x.max().item())
        x = torch.clamp(x, -1e3, 1e3)
        x = self.manifold.projx(x)
        print("[DEBUG] HyperbolicMLR after projx x has nan:", torch.isnan(x).any().item(), "range:", x.min().item(), x.max().item())
        weight_t = self.weight.t()
        weight_norm = weight_t.norm(dim=1)
        print("[DEBUG] HyperbolicMLR weight_norm has nan:", torch.isnan(weight_norm).any().item(), "range:", weight_norm.min().item(), weight_norm.max().item())
        weight_unit = weight_t / weight_norm.unsqueeze(1).clamp_min(1e-15)
        print("[DEBUG] HyperbolicMLR weight_unit has nan:", torch.isnan(weight_unit).any().item(), "range:", weight_unit.min().item(), weight_unit.max().item())
        # Support both PoincaréBall (k<0) and Lorentz (k>0) by taking |k|
        k = self.manifold.k
        if torch.is_tensor(k):
            c = torch.abs(k).to(x.device, x.dtype).clamp_min(1e-15)
        else:
            try:
                c = abs(float(k))
            except Exception:
                c = 1.0
        rc = torch.sqrt(c)
        print("[DEBUG] HyperbolicMLR rc:", rc.item() if torch.numel(rc)==1 else rc)
        bias_clamped = torch.clamp(self.bias, -10.0, 10.0)
        drcr = 2.0 * rc * bias_clamped
        drcr = torch.clamp(drcr, -15.0, 15.0)
        print("[DEBUG] HyperbolicMLR drcr has nan:", torch.isnan(drcr).any().item(), "range:", drcr.min().item(), drcr.max().item())
        rcx = rc * x
        print("[DEBUG] HyperbolicMLR rcx has nan:", torch.isnan(rcx).any().item(), "range:", rcx.min().item(), rcx.max().item())
        cx2 = rcx.pow(2).sum(dim=1, keepdim=True)
        cx2 = torch.clamp(cx2, max=0.999)
        print("[DEBUG] HyperbolicMLR cx2 has nan:", torch.isnan(cx2).any().item(), "range:", cx2.min().item(), cx2.max().item())
        prod = torch.matmul(rcx, weight_unit.t())
        print("[DEBUG] HyperbolicMLR prod has nan:", torch.isnan(prod).any().item(), "range:", prod.min().item(), prod.max().item())
        numer = 2.0 * prod * torch.cosh(drcr) - (1.0 + cx2) * torch.sinh(drcr)
        print("[DEBUG] HyperbolicMLR numer has nan:", torch.isnan(numer).any().item(), "range:", numer.min().item(), numer.max().item())
        denom = torch.clamp_min(1.0 - cx2, 1e-15)
        print("[DEBUG] HyperbolicMLR denom has nan:", torch.isnan(denom).any().item(), "range:", denom.min().item(), denom.max().item())
        asinh_input = numer / denom
        asinh_input = torch.clamp(asinh_input, -15.0, 15.0)
        print("[DEBUG] HyperbolicMLR asinh_input has nan:", torch.isnan(asinh_input).any().item(), "range:", asinh_input.min().item(), asinh_input.max().item())
        dist = 2.0 * weight_norm / rc * torch.asinh(asinh_input)
        print("[DEBUG] HyperbolicMLR dist has nan:", torch.isnan(dist).any().item(), "range:", dist.min().item(), dist.max().item())
        result = dist  # Can flip the sign depending on logits definition
        result = torch.clamp(result, -1e3, 1e3)
        result = torch.nan_to_num(result, nan=0.0)
        
        return result


def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    """Unidirectional hyperbolic logistic regression helper."""
    if not torch.is_tensor(c):
        c = torch.tensor(c, device=x.device, dtype=x.dtype)
    c = torch.abs(c)  # Ensure positive curvature magnitude

    # Parameter preprocessing
    rc = torch.sqrt(c)
    r = torch.clamp(r, -10.0, 10.0)
    drcr = 2. * rc * r
    drcr = torch.clamp(drcr, -15.0, 15.0)

    # Input transform
    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)
    cx2 = torch.clamp(cx2, max=0.999)
    
    prod = torch.matmul(rcx, z_unit)
    
    # Hyperplane distance score
    numer = 2. * prod * torch.cosh(drcr) - (1. + cx2) * torch.sinh(drcr)
    denom = torch.clamp_min(1. - cx2, 1e-15)
    asinh_input = numer / denom
    asinh_input = torch.clamp(asinh_input, -15.0, 15.0)
    result = 2 * z_norm / rc * torch.asinh(asinh_input)
    
    result = torch.clamp(result, -1e3, 1e3)
    result = torch.nan_to_num(result, nan=0.0)
    
    return result

    
class HNNPlusPlusMLR(nn.Module):
    """Unidirectional hyperbolic classifier head (HNN++)."""

    def __init__(self, manifold, feat_dim, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        # Ensure curvature is represented as a tensor
        c = manifold.k if hasattr(manifold, 'k') else manifold.c
        c = abs(c)
        if isinstance(c, float):
            c_tensor = torch.tensor(c)
        else:
            c_tensor = c

        # Weight init after enforcing positive curvature
        weight = torch.empty(feat_dim, num_classes).normal_(
            mean=0, std=(feat_dim) ** -0.5 / torch.sqrt(c_tensor))

        self.weight_g = nn.Parameter(weight.norm(dim=0))  # magnitude scalar r
        self.weight_v = nn.Parameter(weight)              # direction vector p
        self.bias = nn.Parameter(torch.empty(num_classes), requires_grad=bias)  # distance shift d

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        weight_unit = self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15)

        # Use the updated unidirectional_poincare_mlr (handles negative c already)
        return unidirectional_poincare_mlr(
            x, self.weight_g, weight_unit, self.bias, self.manifold.k)  # pass k instead of c



class KleinManifold(torch.nn.Module):
    """
    Klein Manifold wrapper.
    Provides a unified nn.Module interface for Klein manifold operations.

    """
    def __init__(self, c=-1.0, k=None, learnable=False):  # keep k-compatibility
        super().__init__()
        # Support explicit k; fall back to c when not provided
        if k is not None:
            c = k
        self.k = abs(c)  # operate on |k|
        if learnable:
            self.k = nn.Parameter(torch.tensor(self.k, dtype=torch.float32))
        else:
            self.k = torch.tensor(self.k, dtype=torch.float32)
        self.min_norm = 1e-15
        self.name = "Klein"
        self.c = c
    def proj_tan0(self, u, c=None):
        """
        Tangent space at the origin is Euclidean, so only sanitize numerics.
        """
        # Quietly scrub NaN/Inf without spamming logs
        u_safe = torch.nan_to_num(u, nan=0.0)
        return torch.clamp(u_safe, min=-1e6, max=1e6)

    def proj_tan(self, u, x, c=None):
        """Project onto the tangent space at point x in the Klein model."""
        k_val = c if c is not None else self.k

        # Tangent-space projection formula
        # 1) squared norm of x
        x_sq_norm = torch.sum(x * x, dim=-1, keepdim=True)

        # 2) inner product
        x_u_inner = torch.sum(x * u, dim=-1, keepdim=True)

        # 3) assemble projection
        factor = k_val * x_u_inner / (1 - k_val * x_sq_norm)
        proj_u = u - factor * x
        
        return proj_u
        
    def proj(self, x, c=None):
        """Project onto the Klein disk."""
        k_val = c if c is not None else self.k
        eps = 1e-10

        # Norm computation
        x_norm = torch.norm(x, dim=-1, keepdim=True)

        # Detect anomalies
        if torch.isnan(x_norm).any() or torch.isinf(x_norm).any():
            print("Warning: NaN/Inf detected in Klein proj")
            x_norm = torch.where(torch.isnan(x_norm) | torch.isinf(x_norm),
                            torch.ones_like(x_norm) * 0.01, x_norm)

        # Safe projection
        max_norm = (1.0 / math.sqrt(k_val)) - eps
        mask = x_norm > max_norm
        x_safe = torch.where(mask, x * (max_norm / x_norm), x)
        
        return x_safe


    def expmap0(self, u):
        sqrt_k = torch.sqrt(self.k)
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = torch.tanh(sqrt_k * u_norm) * u / (sqrt_k * u_norm)
        return gamma_1

    def logmap0(self, p):
        """Extra-stable logmap0 implementation."""
        # Detect and repair NaN entries before proceeding
        if torch.isnan(p).any():
            print("Warning: logmap0 input contains NaN; attempting repair")
            p = torch.nan_to_num(p, nan=0.0)

        sqrt_k = torch.sqrt(self.k)
        p_norm = p.norm(dim=-1, p=2, keepdim=True)

        # Safely handle tiny norms
        p_norm = p_norm.clamp_min(self.min_norm)

        # Clamp artanh argument
        arg = sqrt_k * p_norm
        arg = torch.clamp(arg, -0.99, 0.99)

        # Evaluate logmap
        scale = 1. / sqrt_k * torch.atanh(arg) / p_norm
        result = scale * p

        # Final clipping
        return torch.clamp(result, -50.0, 50.0)
    
    def mobius_matvec(self, m, x, c=None):
        k_val = c if c is not None else self.k
        
        # Improvement 1: guard against all-zero rows
        row_norms = torch.norm(m, dim=1)
        zero_rows = row_norms < 1e-8
        if zero_rows.any():
            # Inject a tiny stabilizer
            stabilizer = torch.zeros_like(m)
            zero_indices = zero_rows.nonzero().squeeze(-1)
            stabilizer[zero_indices, 0] = 1e-6
            m = m + stabilizer  # avoid later singularities

        # Improvement 2: use the stable logmap0
        x_tan = self.logmap0(x, c=k_val)

        # Improvement 3: clamp intermediate results
        mx = x_tan @ m.transpose(-1, -2)
        mx = torch.clamp(mx, -1e3, 1e3)  # avoid extreme values

        # Stable expmap0
        result = self.expmap0(mx, c=k_val)

        # Project back into the Klein disk
        return self.proj(result, c=k_val)
        


    def scalar_mul(self, r, x, c=None):
        """
        Einstein scalar multiplication implemented directly in the Klein model.
        r: scalar multiplier
        x: point on the Klein model
        """
        k_val = c if c is not None else self.k

        # Handle x=0 or r=0 explicitly
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        mask = (x_norm < self.min_norm) | (torch.abs(r) < self.min_norm)
        if mask.any():
            return torch.zeros_like(x) if mask.all() else x.clone() * 0

        # Standard implementation
        sqrt_k = torch.sqrt(k_val)
        x_norm_clamp = torch.clamp_min(x_norm, self.min_norm)

        # Compute tanh(r * atanh(||x||))
        atanh_norm = torch.atanh(sqrt_k * x_norm_clamp) / sqrt_k
        new_norm = torch.tanh(r * atanh_norm) / sqrt_k

        # Result = new norm * unit vector
        result = new_norm * x / x_norm_clamp
        return self.proj(result, c=k_val)  # keep inside the Klein disk


    def add(self, x, y, c=None, dim=-1):
        """Einstein addition implemented directly on the Klein model."""
        k_val = c if c is not None else self.k
        x2 = x.pow(2).sum(dim=dim, keepdim=True)  # ||x||^2
        xy = (x * y).sum(dim=dim, keepdim=True)   # <x, y>
        gamma_x = 1 / torch.sqrt(1 - k_val * x2)  # Lorentz factor

        # Einstein addition formula
        num = x + (1 / gamma_x) * y + k_val * (gamma_x / (1 + gamma_x)) * xy * x
        denom = 1 + k_val * xy

        result = num / denom.clamp_min(self.min_norm)
        return self.proj(result, c=k_val)  # keep inside the Klein disk
    
    def mobius_add(self, x, y, c=None, dim=-1):
        """
        Möbius addition is equivalent to Einstein addition on the Klein model.
        This alias keeps the API consistent with other manifolds.
        """
        return self.add(x, y, c, dim)
   

    def dist(self, x, y, c=None):
        """Compute the distance between two Klein points."""
        k_val = c if c is not None else self.k

        # Squared norms
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)

        # Inner product
        xy_inner = torch.sum(x * y, dim=-1, keepdim=True)

        # Klein distance formula
        numerator = 2 * k_val * xy_inner + (1 - k_val * x_norm_sq) * (1 - k_val * y_norm_sq)
        denominator = torch.sqrt((1 - k_val * x_norm_sq) * (1 - k_val * y_norm_sq))
        cosh_term = numerator / denominator.clamp_min(self.min_norm)

        # Use acosh to invert the hyperbolic cosine
        cosh_term = torch.clamp(cosh_term, min=1.0 + 1e-7)
        sqrt_k = torch.sqrt(k_val)
        return torch.acosh(cosh_term) / sqrt_k



class KleinManifoldMLR(nn.Module):
    """
    Multinomial Logistic Regression on the Klein model
    (Ganea et al. 2018; Klein formulation  [oai_citation:1‡arXiv](https://arxiv.org/abs/2410.16813?utm_source=chatgpt.com) [oai_citation:2‡CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2020/papers/Khrulkov_Hyperbolic_Image_Embeddings_CVPR_2020_paper.pdf?utm_source=chatgpt.com))
    """
    def __init__(self, manifold, in_features, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold          # provides proj/logmap0
        self.W = nn.Parameter(torch.empty(num_classes, in_features))
        if bias:
            self.b = nn.Parameter(torch.empty(num_classes))
        else:
            self.register_parameter('b', None)

        # He-uniform initialization
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.W, -bound, bound)
        if bias:
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        # Klein projection & tangent mapping
        x = self.manifold.proj(x)                # (...,d)
        x_tan = self.manifold.logmap0(x)         # curvature is positive now
        
        # Vectorized MLR: logits = x·Wᵀ + b
        logits = x_tan @ self.W.t()                       # (...,C)
        if self.b is not None:
            logits = logits + self.b
        
        # Clamp for stability
        logits = torch.clamp(logits, -1e3, 1e3)
        logits = torch.nan_to_num(logits, nan=0.0)
        
        return logits 
# ... existing code ...

class KleinManifoldMLR(nn.Module):
    """
    Multinomial Logistic Regression on the Klein model
    (Ganea et al. 2018; Klein formulation  [oai_citation:1‡arXiv](https://arxiv.org/abs/2410.16813?utm_source=chatgpt.com) [oai_citation:2‡CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2020/papers/Khrulkov_Hyperbolic_Image_Embeddings_CVPR_2020_paper.pdf?utm_source=chatgpt.com))
    """
    def __init__(self, manifold, in_features, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold          # provides proj/logmap0
        self.W = nn.Parameter(torch.empty(num_classes, in_features))
        if bias:
            self.b = nn.Parameter(torch.empty(num_classes))
        else:
            self.register_parameter('b', None)

        # He-uniform initialization
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.W, -bound, bound)
        if bias:
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        # Klein projection & tangent mapping
        x = self.manifold.proj(x)                # (...,d)
        x_tan = self.manifold.logmap0(x)         # curvature is positive now
        
        # Vectorized MLR: logits = x·Wᵀ + b
        logits = x_tan @ self.W.t()                       # (...,C)
        if self.b is not None:
            logits = logits + self.b
        
        # Clamp for stability
        logits = torch.clamp(logits, -1e3, 1e3)
        logits = torch.nan_to_num(logits, nan=0.0)
        
        return logits 


class EuclideanMLR(nn.Module):
    """
    Multiclass logistic regression head in Euclidean space.
    Standard linear classifier used when no manifold constraints are needed.
    """
    def __init__(self, manifold, in_features, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold  # keep API symmetry even though no manifold ops are required
        self.in_features = in_features
        self.num_classes = num_classes

        # Standard linear layer parameters
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_classes))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Use Xavier initialization."""
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input features [batch_size, in_features]
        Returns:
            logits: classification logits [batch_size, num_classes]
        """
        # Standard Euclidean linear transform: logits = x @ Wᵀ + b
        logits = F.linear(x, self.weight, self.bias)

        # Clamp for numerical stability
        logits = torch.clamp(logits, -1e3, 1e3)
        logits = torch.nan_to_num(logits, nan=0.0)

        return logits

class EuclideanMLR(nn.Module):
    """
    Multiclass logistic regression head in Euclidean space.
    Standard linear classifier used when no manifold constraints are needed.
    """
    def __init__(self, manifold, in_features, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold  # keep API symmetry even though no manifold ops are required
        self.in_features = in_features
        self.num_classes = num_classes

        # Standard linear layer parameters
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_classes))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Use Xavier initialization."""
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input features [batch_size, in_features]
        Returns:
            logits: classification logits [batch_size, num_classes]
        """
        # Standard Euclidean linear transform: logits = x @ Wᵀ + b
        logits = F.linear(x, self.weight, self.bias)

        # Clamp for numerical stability
        logits = torch.clamp(logits, -1e3, 1e3)
        logits = torch.nan_to_num(logits, nan=0.0)

        return logits 