"""Compatibility layer for legacy imports.

New code should import from domain modules directly:
- `lib.poincare.hnn_manifold`
- `lib.klein.manifold`
- `lib.Euclidean.mlr`
- `lib.pv.graph_ops` / `lib.pv.manifold` / `lib.pv.layers`
- `lib.lorentz.graph_manifold`
"""

from lib.poincare.hnn_manifold import PoincareBall, HyperbolicMLR, HNNPlusPlusMLR
from lib.klein.manifold import KleinManifold, KleinManifoldMLR
from lib.Euclidean.mlr import EuclideanMLR
from lib.lorentz.graph_manifold import LorentzManifold, LorentzManifoldMLR
from lib.pv.graph_ops import PVManifold, PVManifoldMLR

__all__ = [
    "PoincareBall",
    "HyperbolicMLR",
    "HNNPlusPlusMLR",
    "KleinManifold",
    "KleinManifoldMLR",
    "EuclideanMLR",
    "LorentzManifold",
    "LorentzManifoldMLR",
    "PVManifold",
    "PVManifoldMLR",
]
