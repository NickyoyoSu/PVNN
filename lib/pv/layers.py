from .graph_ops import PVManifoldMLR as _GraphPVManifoldMLR


class PVManifoldMLR(_GraphPVManifoldMLR):
    def __init__(self, c: float, in_features: int, num_classes: int):
        if c <= 0:
            raise ValueError(f"c must be positive, got {c}")
        super().__init__(-abs(float(c)), in_features, num_classes)


__all__ = ["PVManifoldMLR"]
