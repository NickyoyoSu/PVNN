import torch


def report_if_nonfinite(tag: str, tensor: torch.Tensor) -> None:
    """Print minimal diagnostics when tensor contains NaN/Inf."""
    if tensor is None:
        return
    if torch.isfinite(tensor).all():
        return
    t = torch.nan_to_num(tensor.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    print(
        f"[PV DEBUG] {tag}: non-finite detected, "
        f"shape={tuple(tensor.shape)}, "
        f"min={float(t.min().item()):.6f}, "
        f"max={float(t.max().item()):.6f}, "
        f"mean={float(t.mean().item()):.6f}",
    )
import torch


def report_if_nonfinite(tag: str, tensor: torch.Tensor) -> None:
    """Print lightweight diagnostics when tensor contains NaN/Inf."""
    if tensor is None:
        return
    if torch.isfinite(tensor).all():
        return
    t = torch.nan_to_num(tensor.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    print(
        f"[PV DEBUG] {tag}: non-finite detected, "
        f"shape={tuple(tensor.shape)}, "
        f"min={float(t.min().item()):.6f}, "
        f"max={float(t.max().item()):.6f}, "
        f"mean={float(t.mean().item()):.6f}"
    )
