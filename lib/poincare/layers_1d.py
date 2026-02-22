import math
import torch
import torch.nn as nn

from lib.geoopt.manifolds.stereographic.manifold import PoincareBall
from lib.poincare.layers.PMLR import UnidirectionalPoincareMLR


def _beta_fn(a: float, b: float) -> float:
    return math.gamma(a) * math.gamma(b) / math.gamma(a + b)


class PoincareBatchNorm1d(nn.Module):
    def __init__(self, manifold: PoincareBall, num_features: int):
        super().__init__()
        self.manifold = manifold
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) on Poincaré ball
        B, C, L = x.shape
        xt = self.manifold.logmap0(x.permute(0, 2, 1).contiguous().view(-1, C))
        xt = self.bn(xt.view(B, L, C).permute(0, 2, 1))
        y = self.manifold.expmap0(xt.permute(0, 2, 1).contiguous().view(-1, C)).view(B, L, C).permute(0, 2, 1)
        return y


class PoincareReLU(nn.Module):
    def __init__(self, manifold: PoincareBall):
        super().__init__()
        self.manifold = manifold
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        xt = self.manifold.logmap0(x.permute(0, 2, 1).contiguous().view(-1, C))
        xt = self.relu(xt)
        y = self.manifold.expmap0(xt).view(B, L, C).permute(0, 2, 1)
        return y


class PoincareConv1d(nn.Module):
    def __init__(
        self,
        manifold: PoincareBall,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        bias: bool = True,
        id_init: bool = True,
    ):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.has_bias = bias
        self.id_init = id_init

        in_features = self.in_channels * self.kernel_size

        # Parameters to mimic Poincaré MLR linear kernel
        self.weights = nn.Parameter(torch.empty(in_features, self.out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))

        # Beta constants for scalar transform
        self.beta_ni = _beta_fn(self.in_channels / 2.0, 0.5)
        self.beta_n = _beta_fn((self.in_channels * self.kernel_size) / 2.0, 0.5)

        # Internal MLR
        self.mlr = UnidirectionalPoincareMLR(
            feat_dim=in_features, num_outcome=self.out_channels, bias=bias, ball=self.manifold
        )

        self.reset_parameters()

    def reset_parameters(self):
        in_features = self.in_channels * self.kernel_size
        if self.id_init:
            # Use identity-like init (0.5 factor), handling rectangular case
            self.weights.data.zero_()
            diag = min(in_features, self.out_channels)
            self.weights.data[:diag, :diag] = 0.5 * torch.eye(diag)
        else:
            nn.init.normal_(
                self.weights, mean=0.0, std=(2.0 * self.in_channels * self.kernel_size * self.out_channels) ** -0.5
            )
        if self.has_bias:
            nn.init.zeros_(self.bias)

        # Initialize internal MLR to small values as well
        with torch.no_grad():
            self.mlr.weight_v.data.normal_(mean=0.0, std=(self.in_channels * self.kernel_size) ** -0.5)
            self.mlr.weight_g.data.copy_(self.mlr.weight_v.data.norm(dim=0))
            self.mlr.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) on Poincaré ball
        B, C, L = x.shape
        if self.padding > 0:
            pad = x.new_zeros((B, C, self.padding))
            x = torch.cat([pad, x, pad], dim=-1)
            L = x.shape[-1]

        K = self.kernel_size
        L_out = (L - K) // self.stride + 1

        # Extract patches: (B, C, L) -> (B, C, L_out, K) -> (B, L_out, C*K)
        patches = x.unfold(dimension=-1, size=K, step=self.stride)  # (B, C, L_out, K)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(B * L_out, C * K)

        # Scalar transform for concatenation
        scale = (self.beta_n / self.beta_ni)
        patches = patches * scale

        # Map to Poincaré ball from origin
        patches_ball = self.manifold.expmap0(patches)

        # Apply Poincaré MLR (internal)
        y = self.mlr(patches_ball)
        if self.has_bias:
            y = y + self.bias

        # Map Euclidean output back to ball using stereographic formula
        c = self.manifold.c
        rc = c.sqrt()
        y = (rc * y).sinh() / rc
        denom = 1 + (1 + c * y.pow(2).sum(dim=-1, keepdim=True)).sqrt()
        y = y / denom

        # Reshape to (B, out_channels, L_out)
        y = y.view(B, L_out, self.out_channels).permute(0, 2, 1).contiguous()
        return y

