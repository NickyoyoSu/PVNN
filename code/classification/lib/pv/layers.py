import math
import torch
import torch.nn as nn

from .manifold import PV_DEBUG, TINY, _sech


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


__all__ = ["PVManifoldMLR"]

