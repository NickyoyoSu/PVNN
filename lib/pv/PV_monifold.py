from .graph_ops import PVFC as _GraphPVFC
from .graph_ops import PVManifoldMLR as _GraphPVManifoldMLR
from .manifold import PVManifold, TINY, _sech, beta


def _as_negative_curvature(c: float) -> float:
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    return -abs(float(c))


class PVManifoldMLR(_GraphPVManifoldMLR):
    def __init__(self, c: float, in_features: int, num_classes: int):
        super().__init__(_as_negative_curvature(c), in_features, num_classes)


class PVFC(_GraphPVFC):
    def __init__(
        self,
        c: float,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        inner_act: str = "none",
    ):
        super().__init__(
            _as_negative_curvature(c),
            in_features,
            out_features,
            use_bias=use_bias,
            inner_act=inner_act,
        )


__all__ = ["PVManifold", "PVManifoldMLR", "PVFC", "TINY", "_sech", "beta"]
