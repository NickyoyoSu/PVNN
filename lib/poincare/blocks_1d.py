import torch.nn as nn
from lib.poincare.layers_1d import PoincareConv1d, PoincareBatchNorm1d, PoincareReLU


def get_poincare_convolution_block(manifold, channels_sizes, kernel_size: int = 9, padding: int = 4):
    return nn.Sequential(
        PoincareConv1d(
            manifold=manifold,
            in_channels=channels_sizes[0],
            out_channels=channels_sizes[1],
            kernel_size=kernel_size,
            padding=padding,
        ),
        PoincareBatchNorm1d(manifold=manifold, num_features=channels_sizes[1]),
        PoincareReLU(manifold=manifold),
        PoincareConv1d(
            manifold=manifold,
            in_channels=channels_sizes[2],
            out_channels=channels_sizes[3],
            kernel_size=kernel_size,
            padding=padding,
        ),
        PoincareBatchNorm1d(manifold=manifold, num_features=channels_sizes[3]),
    )

