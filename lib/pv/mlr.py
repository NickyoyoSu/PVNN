import torch
import torch.nn as nn

from .PV_monifold import PVManifoldMLR as _PVManifoldMLR


def _c_from_manifold(manifold) -> float:
    """Get positive curvature c from manifold for PV_monifold.PVManifoldMLR."""
    if hasattr(manifold, "c"):
        try:
            return float(manifold.c)
        except Exception:
            pass
    if hasattr(manifold, "c_tensor"):
        try:
            return float(manifold.c_tensor.detach().item())
        except Exception:
            pass
    if hasattr(manifold, "k"):
        try:
            k_val = float(manifold.k)
            return -k_val if k_val < 0 else abs(k_val)
        except Exception:
            pass
    return 1.0


class PVManifoldMLR(nn.Module):
    """PV MLR head compatible with PV_monifold (z, r) closed-form."""
    def __init__(self, manifold, in_features, num_classes):
        super().__init__()
        c_val = _c_from_manifold(manifold)
        self.impl = _PVManifoldMLR(c=c_val, in_features=in_features, num_classes=num_classes)

    def forward(self, x):
        return self.impl(x)
