import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as T

from .PV_monifold import PVManifold, PVFC
from .gyrobn_pv import PVGyroBN1d
from . import debug as pv_debug


class PVConv1d(nn.Module):
    """
    "Aggregate-then-Transform" style PV convolution layer.

    Implements PVFC(Concat(x_i)) pattern.
    Uses PV manifold as R^n; no Log0/Exp0 wrapper.

    1. Extract C_in-dim vectors x_i from k*k receptive field.
    2. Concat all k x_i in Euclidean space to form k*C_in-dim vector x_concat.
    3. Treat x_concat as point on PV space R^{k*C_in}.
    4. Use PVFC (Thm 5.3) to map x_concat from PV^{k*C_in} to PV^{C_out}.
    """
    def __init__(self, 
                 c: float, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int = 1, 
                 padding: int = 0, 
                 bias: bool = True,
                 inner_act: str = 'none'):
        
        super().__init__()
        self.manifold = PVManifold(c)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.num_kernel_positions = self.kernel_size
        
        # Core design: single PVFC layer
        # Input dim k * C_in (concatenated vector), output dim C_out
        self.pvfc_transform = PVFC(
            c=c,
            in_features=self.num_kernel_positions * self.in_channels,
            out_features=self.out_channels,
            use_bias=bias,
            inner_act=inner_act
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x shape: [B, C_in, L_in] (PyTorch 1D conv standard format)
        """
        B, C_in, L_in = x.shape
        if C_in != self.in_channels:
            raise ValueError(f"Input channels ({C_in}) do not match declared in_channels ({self.in_channels})")

        # --- 1. Extract patches (Euclidean) ---
        # F.unfold needs 4D input; treat 1D as 2D with H=1
        x_4d = x.unsqueeze(2)  # [B, C_in, 1, L_in]
        
        patches = F.unfold(x_4d, 
                           kernel_size=(1, self.kernel_size), 
                           padding=(0, self.padding), 
                           stride=(1, self.stride))
        # patches shape: [B, C_in * k, L_out]

        L_out = patches.shape[2]  # actual L_out

        # --- 2. Euclidean aggregation (Concat) ---
        # [B, C_in * k, L_out] -> [B, L_out, C_in * k]
        patches_permuted = patches.permute(0, 2, 1).contiguous()
        
        # -> [B*L_out, C_in * k] (flatten B and L_out)
        # This is x_concat, a large vector in R^{k*C_in}
        x_concat = patches_permuted.view(B * L_out, -1)

        # --- 3. Apply PVFC transform (on manifold) ---
        # x_concat treated as point on PV^{k*C_in}
        # x_out_flat shape: [B*L_out, C_out]
        x_out_flat = self.pvfc_transform(x_concat)
            
        # --- 4. Restore sequence shape ---
        # [B*L_out, C_out] -> [B, L_out, C_out]
        x_out = x_out_flat.view(B, L_out, self.out_channels)

        # [B, L_out, C_out] -> [B, C_out, L_out] (restore PyTorch NCL)
        x_out = x_out.permute(0, 2, 1).contiguous()
        
        return x_out


class PVReLU(nn.Module):
    """
    ReLU on PV manifold: logmap0 -> ReLU -> expmap0
    """
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.manifold.logmap0(x, c=getattr(self.manifold, 'c', None))
        v = F.relu(v)
        y = self.manifold.expmap0(v, c=getattr(self.manifold, 'c', None))
        return y


class PVAct1d(nn.Module):
    """
    Optional activation on PV manifold: logmap0 -> act -> expmap0

    Supports:
    - relu
    - tanh
    - softplus
    - none / identity
    """
    def __init__(self, manifold, act: str = "relu"):
        super().__init__()
        self.manifold = manifold
        self.act = (act or "none").lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pv_debug.report_if_nonfinite("PVAct1d.input", x)
        if self.act in ("none", "identity", "linear"):
            return x
        v = self.manifold.logmap0(x, c=getattr(self.manifold, 'c', None))
        pv_debug.report_if_nonfinite("PVAct1d.logmap0", v)
        if self.act == "relu":
            v = F.relu(v)
        elif self.act == "tanh":
            v = torch.tanh(v)
        elif self.act == "softplus":
            v = F.softplus(v, beta=1.0, threshold=20.0)
        else:
            raise ValueError(f"Unsupported PV activation: {self.act}")
        y = self.manifold.expmap0(v, c=getattr(self.manifold, 'c', None))
        pv_debug.report_if_nonfinite("PVAct1d.output", y)
        return y


class PVBatchNorm1d(nn.Module):
    """
    PV BN with compatible interface:
    - use_gyrobn=True: use PVGyroBN1d from gyrobn_pv, inner manifold from PV_monifold.PVManifold(c)
    - use_gyrobn=False: tangent-space Euclidean BN
    Interface compatible with blocks_1d extended params (unused ignored)
    """
    def __init__(self,
                 manifold,
                 num_features: int,
                 use_gyrobn: bool = True,
                 print_stats: bool = False,
                 clamp_factor: float = 3.0,
                 use_euclid_stats: bool = True,
                 **kwargs):
        super().__init__()
        self.manifold = manifold
        self.use_gyrobn = use_gyrobn
        if use_gyrobn:
            c_val = float(getattr(manifold, 'c', getattr(manifold, 'curvature', 1.0)))
            inner_manifold = PVManifold(c=c_val)
            self.impl = PVGyroBN1d(inner_manifold, num_features)
        else:
            self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gyrobn:
            return self.impl(x)
        # Tangent-space BN
        B, C, L = x.shape
        xt = self.manifold.logmap0(x.permute(0, 2, 1).contiguous().view(-1, C), c=getattr(self.manifold, 'c', None))
        xt = self.bn(xt.view(B, L, C).permute(0, 2, 1))
        y = self.manifold.expmap0(xt.permute(0, 2, 1).contiguous().view(-1, C), c=getattr(self.manifold, 'c', None)).view(B, L, C).permute(0, 2, 1)
        return y