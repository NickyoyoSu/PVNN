import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as T

from .PV_monifold import PVManifold, PVFC
from .gyrobn_pv import PVGyroBN1d
from . import debug as pv_debug


class PVConv2d(nn.Module):
    """
    A proper hyperbolic 2D convolution layer for PV manifold.

    Follows "Transform-then-Aggregate" paradigm:
    1. Create independent PVFC(in, out) for each (i,j) in k*k kernel.
    2. For each patch of x, feed C_in-dim vector at (i,j) to PVFC_{i,j}, get C_out-dim hyperbolic vector.
    3. Log0 the k*k C_out-dim vectors to tangent space.
    4. Sum them in tangent space (aggregate).
    5. Apply activation (phi) and bias (b) in tangent space.
    6. Exp0 result back to PV manifold.
    """
    def __init__(self, 
                 c: float, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int = 1, 
                 padding: int = 0, 
                 bias: bool = True, 
                 activation: T.Callable = F.relu):
        """
        Initialize PV convolution layer.

        Args:
            c (float): Manifold curvature (c > 0).
            in_channels (int): Input channels.
            out_channels (int): Output channels (number of filters).
            kernel_size (int): Kernel size (k x k).
            stride (int): Stride.
            padding (int): Padding.
            bias (bool): Whether to add Euclidean bias after tangent-space aggregation.
            activation (Callable): Activation in tangent space (e.g. F.relu).
        """
        super().__init__()
        self.manifold = PVManifold(c)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Unified kernel_size handling
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
            
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        
        # Core design: create k*k independent PVFC layers, each C_in -> C_out.
        self.k_h, self.k_w = self.kernel_size
        self.num_kernels = self.k_h * self.k_w
        
        self.kernels = nn.ModuleList()
        for _ in range(self.num_kernels):
            # No bias inside PVFC; we add shared bias after aggregation
            self.kernels.append(PVFC(c, in_channels, out_channels, use_bias=False))

        # Shared Euclidean bias, added after tangent-space aggregation
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x shape: [B, C_in, H, W]
        """
        B, C_in, H, W = x.shape
        if C_in != self.in_channels:
            raise ValueError(f"Input channels ({C_in}) do not match declared in_channels ({self.in_channels})")

        # --- 1. Extract patches ---
        # Use F.unfold for efficient receptive field extraction
        patches = F.unfold(x, 
                           kernel_size=self.kernel_size, 
                           padding=self.padding, 
                           stride=self.stride)
        # patches shape: [B, C_in * k*k, L]
        # L is output pixel count (H' * W')

        # --- 2. Reshape patches for kernel application ---
        # [B, C_in * k*k, L] -> [B, C_in, k*k, L]
        patches = patches.view(B, C_in, self.num_kernels, -1)
        
        # -> [B, L, k*k, C_in] (move L near batch dim)
        patches = patches.permute(0, 3, 2, 1).contiguous()

        # -> [B*L, k*k, C_in] (flatten B and L for batch processing)
        # We have B*L "patches", each with k*k C_in-dim vectors
        L = patches.shape[1]  # actual L (H_out * W_out)
        patches = patches.view(B * L, self.num_kernels, C_in)

        # --- 3. Apply PVFC kernels, Log0, and aggregate ---
        v_patches_list = []
        for i in range(self.num_kernels):
            # Get C_in-dim vector at position i for all patches
            # x_i shape: [B*L, C_in]
            x_i = patches[:, i, :]

            # Step 1: Feature transform (on PV manifold)
            # h_i shape: [B*L, C_out]
            h_i = self.kernels[i](x_i)

            # Step 2: Map to tangent space (for aggregation)
            v_i = self.manifold.log0(h_i)
            v_patches_list.append(v_i)
        
        # Step 3: Aggregate (in tangent space)
        # v_stack shape: [k*k, B*L, C_out]
        v_stack = torch.stack(v_patches_list, dim=0)
        # v_agg shape: [B*L, C_out]
        v_agg = v_stack.sum(dim=0)

        # --- 4. Apply bias and activation (in tangent space) ---
        if self.bias is not None:
            v_agg = v_agg + self.bias  # broadcast [B*L, C_out] + [C_out]
        
        if self.activation is not None:
            v_agg = self.activation(v_agg)
            
        # --- 5. Map back to PV manifold ---
        # x_out_flat shape: [B*L, C_out]
        x_out_flat = self.manifold.exp0(v_agg)

        # --- 6. Restore image shape ---
        # Compute H_out, W_out
        H_out = (H + 2 * self.padding - self.k_h) // self.stride + 1
        W_out = (W + 2 * self.padding - self.k_w) // self.stride + 1
        
        # [B*L, C_out] -> [B, L, C_out]
        x_out = x_out_flat.view(B, L, self.out_channels)
        
        # [B, L, C_out] -> [B, H_out, W_out, C_out]
        x_out = x_out.view(B, H_out, W_out, self.out_channels)
        
        # [B, H_out, W_out, C_out] -> [B, C_out, H_out, W_out] (restore PyTorch NCHW)
        x_out = x_out.permute(0, 3, 1, 2).contiguous()
        
        return x_out



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


class PVFullyConnected(nn.Module):
    """
    PV fully connected (stays on manifold): PVFC wrapper
    """
    def __init__(self, manifold, in_features: int, out_features: int, use_bias: bool = True, inner_act: str = 'none'):
        super().__init__()
        c_val = float(getattr(manifold, 'c', getattr(manifold, 'curvature', 1.0)))
        self.fc = PVFC(c=c_val, in_features=in_features, out_features=out_features, use_bias=use_bias, inner_act=inner_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


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