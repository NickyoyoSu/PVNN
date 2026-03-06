import torch.nn as nn
from .layers_1d import (
    PVBatchNorm1d,
    PVAct1d,
    PVConv1d,
)
from .manifold import PVManifold


def get_pv_convolution_block(manifold: PVManifold,
                             channels_sizes,
                             kernel_size: int = 9,
                             padding: int = 4,
                             act: str = "none",
                             act2: str | None = None):
    """
    Expects channels_sizes = [c_in, c_mid, c_mid, c_out]
    """
    if act2 is None:
        act2 = act
    return nn.Sequential(
        PVConv1d(c=getattr(manifold, 'c', getattr(manifold, 'curvature', 1.0)),
                 in_channels=channels_sizes[1],
                 out_channels=channels_sizes[1],
                 kernel_size=kernel_size,
                 padding=padding),
        PVBatchNorm1d(manifold=manifold, num_features=channels_sizes[1], use_gyrobn=False, clamp_factor=3.0, use_euclid_stats=True),
        PVAct1d(manifold=manifold, act=act),
        PVConv1d(c=getattr(manifold, 'c', getattr(manifold, 'curvature', 1.0)),
                 in_channels=channels_sizes[1],
                 out_channels=channels_sizes[1],
                 kernel_size=kernel_size,
                 padding=padding),
        PVAct1d(manifold=manifold, act=act2),
        
    )
       
