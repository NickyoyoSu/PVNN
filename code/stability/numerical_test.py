import math
import os
import sys
import time
import argparse
from contextlib import nullcontext

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

# ===== Local imports =====
from pv_manifold import PVManifold
def make_pv_manifold(kappa: float) -> PVManifold:
    """Create PVManifold using negative curvature K=-|kappa| (historical c>0)."""
    return PVManifold(k=-abs(float(kappa)))
from hnn_layers import PoincareBall, mobius_scalar_mul

# Lorentz (hyperboloid) model

from lorentz_manifold import LorentzManifold
L_AVAILABLE = True



# =================== Utilities ===================

def summarize(name: str, y: torch.Tensor):
    y_ = torch.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
    any_nan = bool(torch.isnan(y).any().item())
    any_inf = bool(torch.isinf(y).any().item())
    print(f"[{name}] "
          f"min={float(y_.min().item()):.3e} "
          f"max={float(y_.max().item()):.3e} "
          f"mean={float(y_.mean().item()):.3e} "
          f"std={float(y_.std(unbiased=False).item()):.3e} "
          f"nan={any_nan} inf={any_inf}")

def has_nan_or_inf(t: torch.Tensor) -> bool:
    return bool(torch.isnan(t).any().item()) or bool(torch.isinf(t).any().item())

def failure_rate(t: torch.Tensor) -> float:
    """Return fraction of samples containing NaN/Inf."""
    if t.numel() == 0:
        return 0.0
    bad = torch.isnan(t) | torch.isinf(t)
    if bad.ndim == 1:
        per_sample = bad
    else:
        per_sample = bad.view(bad.shape[0], -1).any(dim=1)
    return float(per_sample.float().mean().item())

def count_nan_inf(t: torch.Tensor):
    return int(torch.isnan(t).any().item()), int(torch.isinf(t).any().item())

def poincare_saturation_rate(x_pb: torch.Tensor, kappa: float, thr: float = 0.999) -> float:
    # Near-boundary saturation rate (||x|| close to 1/sqrt(kappa))
    sqrt_c = math.sqrt(kappa)
    max_ball = (1.0 / sqrt_c) * (1.0 - 1e-6)
    norm = torch.linalg.norm(x_pb, dim=-1)
    return float((norm >= max_ball * thr).float().mean().item())

def lorentz_constraint_violation(x_l: torch.Tensor, kappa: float, atol: float = 1e-5) -> float:
    """
    Constraint violation rate: x0^2 - ||x||^2 = 1/kappa
    Samples containing NaN / Inf also count as violations.
    """
    x0, xs = x_l[..., :1], x_l[..., 1:]
    lhs = x0.pow(2) - (xs.pow(2)).sum(dim=-1, keepdim=True)
    target = 1.0 / kappa
    diff = lhs - target

    # Whether each sample violates the constraint or contains NaN/Inf in any coordinate
    bad = torch.isnan(diff) | torch.isinf(diff) | (diff.abs() > atol)
    if bad.ndim == 1:
        per_sample = bad
    else:
        per_sample = bad.view(bad.shape[0], -1).any(dim=1)
    return float(per_sample.float().mean().item())

def lorentz_renorm(x_l: torch.Tensor, kappa: float) -> torch.Tensor:
    # Project (x0, s) back onto the hyperboloid: x0 = sqrt(1/k + ||s||^2), keeping s unchanged
    x0, s = x_l[..., :1], x_l[..., 1:]
    new_x0 = torch.sqrt(1.0 / kappa + (s * s).sum(dim=-1, keepdim=True))
    return torch.cat([new_x0, s], dim=-1)

def make_unit_dirs(batch, d, device, dtype):
    dirs = torch.randn(batch, d, device=device, dtype=dtype)
    return dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-9)

def make_common_vs(batch, d, radii, device, dtype):
    """
    Sample shared tangent-space directions with varying norms (log-spaced),
    used to build comparable inputs across manifolds via exp0.
    """
    dirs = make_unit_dirs(batch, d, device, dtype)
    radii = radii.to(device=device, dtype=dtype)
    return dirs * radii.unsqueeze(-1)

def lift_to_lorentz_tangent(v):
    """
    Embed a d-dimensional tangent vector into Lorentz tangent space (d+1) by
    prepending a zero time component.
    """
    zeros = torch.zeros(v.shape[0], 1, device=v.device, dtype=v.dtype)
    return torch.cat([zeros, v], dim=-1)

def make_lorentz_batch_from_dirs(dirs, radii, kappa, dtype, device):
    # Given the space component s, build (x0, s) on the hyperboloid
    s = dirs * radii.unsqueeze(-1)
    x0 = torch.sqrt(1.0 / kappa + (s * s).sum(dim=-1, keepdim=True))
    return torch.cat([x0.to(dtype), s.to(dtype)], dim=-1).to(device)


# =================== Guarded PV scalar multiplication ===================

@torch.no_grad()
def pv_scalar_mul_guarded(X: torch.Tensor, r: torch.Tensor, kappa: float, zmax: float = 20.0) -> torch.Tensor:
    """
    Guarded PV scalar multiplication: z = clamp(r * asinh(kappa*||X||), ±zmax), Y = sinh(z)/(kappa||X||) * X; r⊗0=0.
    """
    eps = torch.finfo(X.dtype).eps
    norm = torch.linalg.norm(X, dim=-1, keepdim=True).clamp_min(eps)
    a = torch.asinh(kappa * norm)
    z = torch.clamp(r * a, min=-zmax, max=zmax)
    coef = torch.sinh(z) / (kappa * norm)
    Y = coef * X
    Y = torch.where(norm < 10 * eps, torch.zeros_like(Y), Y)
    return Y


def pv_scalar_mul_guarded_diff(X: torch.Tensor, r: torch.Tensor, kappa: float, zmax: float = 20.0) -> torch.Tensor:
    """
    Differentiable variant of pv_scalar_mul_guarded for gradient sweeps.
    """
    eps = torch.finfo(X.dtype).eps
    norm = torch.linalg.norm(X, dim=-1, keepdim=True).clamp_min(eps)
    a = torch.asinh(kappa * norm)
    z = r * a
    if zmax is not None:
        z = torch.clamp(z, min=-zmax, max=zmax)
    coef = torch.sinh(z) / (kappa * norm)
    Y = coef * X
    Y = torch.where(norm < 10 * eps, torch.zeros_like(Y), Y)
    return Y


# =================== Gradient curve (∥∇X f∥ vs radius) ===================

def grad_sweep(op_fn, X_seed, radii=None):
    """
    Given op_fn(X)->Y, compute ∥∇_X ||Y||∥ as a function of the radius
    """
    device, dtype = X_seed.device, X_seed.dtype
    radii = radii if radii is not None else torch.logspace(0, 4, steps=32, base=10., device=device, dtype=dtype)
    grads = []
    for r in radii:
        x = X_seed.clone()
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9) * r
        x.requires_grad_(True)
        out = op_fn(x)
        loss = out.norm(dim=-1).mean()
        loss.backward()
        g = x.grad.norm(dim=-1).mean().detach().item()
        grads.append(g)
    return radii.detach().cpu().numpy().tolist(), grads


# =================== Precision / timing / memory benchmark ===================

def run_precision_sweep(op_name, build_inputs_fn, op_fns, kappa=0.5, device=None):
    """
    Compare FP32 / FP16 / BF16; record time, memory, NaN/Inf, Poincaré saturation rate and Lorentz constraint violation rate
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    dtypes = [torch.float32] + ([torch.float16, torch.bfloat16] if device.type == 'cuda' else [])
    results = []
    for dt in dtypes:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        inp = build_inputs_fn(dt, device)
        for name, fn in op_fns.items():
            if fn is None:
                continue
            ctx = (torch.autocast(device_type='cuda', dtype=dt) 
                   if (device.type == 'cuda' and dt in (torch.float16, torch.bfloat16)) else nullcontext())
            start = time.perf_counter()
            with ctx:
                out = fn(inp[name])
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_ms = 1000.0 * (time.perf_counter() - start)
            n_nan, n_inf = count_nan_inf(out)
            rec = dict(model=name, op=op_name, dtype=str(dt).split('.')[-1], time_ms=t_ms, nan=bool(n_nan), inf=bool(n_inf))
            if device.type == 'cuda':
                rec["mem_bytes"] = int(torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
            # Additional event counts
            if name == "Poincare":
                rec["sat_rate"] = poincare_saturation_rate(out, kappa)
            if name == "Lorentz":
                rec["lorentz_viol"] = lorentz_constraint_violation(out, kappa)
            results.append(rec)
    return results


# =================== Scalar multiplication / addition experiments ===================

@torch.no_grad()
def run_mul_once(dtype=torch.float64, d=16, kappa=0.5, batch=4096, r_scale=50.0, device=torch.device('cpu')):
    torch.set_default_dtype(dtype)

    dirs = make_unit_dirs(batch, d, device, dtype)

    # PV: large radii
    radii_pv = torch.logspace(0, 3, steps=batch, base=10.0, dtype=dtype, device=device)
    X_pv = dirs * radii_pv.unsqueeze(-1)

    # Poincaré: fixed fraction of the ball radius (fairer, avoids near-boundary saturation)
    sqrt_c = math.sqrt(kappa)
    ball_radius = 1.0 / sqrt_c
    pb_frac = 0.6  # 60% of the ball radius
    X_pb = dirs * (pb_frac * ball_radius)

    # Lorentz: build (x0, s)
    X_lx = make_lorentz_batch_from_dirs(dirs, radii_pv, kappa, dtype, device) if L_AVAILABLE else None

    r = torch.tensor(r_scale, dtype=dtype, device=device)

    M = make_pv_manifold(kappa)
    B = PoincareBall()
    L = LorentzManifold(c=kappa) if L_AVAILABLE else None

    # PV scalar multiplication (wider clamp to avoid early saturation and show the dynamic range)
    Y_pv = pv_scalar_mul_guarded(X_pv, r, kappa, zmax=30.0)

    # Poincaré scalar multiplication
    Y_pb = mobius_scalar_mul(X_pb, r, c=kappa)

    # Lorentz scalar "multiplication" = exp0(r * log0(x))
    if L_AVAILABLE:
        v = L.logmap0(X_lx, c=kappa)
        Y_lx = L.expmap0(r * v, c=kappa)
        Y_lx = lorentz_renorm(Y_lx, kappa)  # keep points on the manifold so later diagnostics are not distorted
    else:
        Y_lx = None

    print(f"\n=== MUL | dtype={dtype}, dim={d}, kappa={kappa}, batch={batch}, r={float(r.item())} ===")
    summarize("PV        scalar_mul output", Y_pv)
    summarize("Poincare  scalar_mul output", Y_pb)
    if L_AVAILABLE:
        summarize("Lorentz   scalar_mul output", Y_lx)

    # Near-boundary / constraint diagnostics
    pb_sat = poincare_saturation_rate(Y_pb, kappa)
    print(f"[Poincare] near-boundary saturation rate ≈ {pb_sat*100:.2f}%")
    if L_AVAILABLE:
        lx_viol = lorentz_constraint_violation(Y_lx, kappa)
        print(f"[Lorentz ] constraint violation rate ≈ {lx_viol*100:.2f}%")

    # Dynamic range diagnostics
    pv_norm = torch.linalg.norm(Y_pv, dim=-1)
    pb_norm = torch.linalg.norm(Y_pb, dim=-1)
    print(f"[Range] median ||Y|| — PV: {float(pv_norm.median().item()):.3e} | "
          f"Poincare: {float(pb_norm.median().item()):.3e}" +
          (f" | Lorentz: {float(torch.linalg.norm(Y_lx[...,1:], dim=-1).median().item()):.3e}" if L_AVAILABLE else ""))

@torch.no_grad()
def run_add_once(dtype=torch.float64, d=16, kappa=0.5, batch=4096, device=torch.device('cpu')):
    torch.set_default_dtype(dtype)

    dirs1 = make_unit_dirs(batch, d, device, dtype)
    dirs2 = make_unit_dirs(batch, d, device, dtype)
    radii1 = torch.logspace(0, 3, steps=batch, base=10.0, dtype=dtype, device=device)
    radii2 = torch.logspace(0, 3, steps=batch, base=10.0, dtype=dtype, device=device)

    X_pv = dirs1 * radii1.unsqueeze(-1)
    Y_pv = dirs2 * radii2.unsqueeze(-1)

    sqrt_c = math.sqrt(kappa)
    max_ball = (1.0 / sqrt_c) * (1.0 - 1e-6)
    X_pb = dirs1 * max_ball
    Y_pb = dirs2 * max_ball

    M = make_pv_manifold(kappa)
    B = PoincareBall()

    Z_pv = M.gyro_add(X_pv, Y_pv)
    Z_pb = B.mobius_add(X_pb, Y_pb, kappa)

    print(f"\n=== ADD | dtype={dtype}, dim={d}, kappa={kappa}, batch={batch} ===")
    summarize("PV       add output", Z_pv)
    summarize("Poincare add output", Z_pb)

    # Near-boundary saturation rate
    zpb_norm = torch.linalg.norm(Z_pb, dim=-1)
    sat_rate = poincare_saturation_rate(Z_pb, kappa)
    print(f"[Poincare-ADD] near-boundary saturation rate ≈ {sat_rate*100:.2f}%")

    # Dynamic range
    zpv_norm = torch.linalg.norm(Z_pv, dim=-1)
    print(f"[Range-ADD] PV median ||Z||={float(zpv_norm.median().item()):.3e} | "
          f"Poincare median ||Z||={float(zpb_norm.median().item()):.3e}")

def sweep_mul_threshold(dtype=torch.float64, d=16, kappa=0.5, batch=2048, r_list=(1,2,5,10,20,50,100,200), guard=False, device=torch.device('cpu')):
    torch.set_default_dtype(dtype)
    dirs = make_unit_dirs(batch, d, device, dtype)
    radii_pv = torch.logspace(0, 3, steps=batch, base=10.0, dtype=dtype, device=device)
    X_pv = dirs * radii_pv.unsqueeze(-1)
    M = make_pv_manifold(kappa)
    rates = []
    for r in r_list:
        r_t = torch.tensor(float(r), dtype=dtype, device=device)
        if guard:
            Y = pv_scalar_mul_guarded(X_pv, r_t, kappa, zmax=20.0 if dtype==torch.float64 else 10.0)
        else:
            Y = M.gyro_scalar_mul(r_t, X_pv)
        rate = failure_rate(Y)
        print(f"[SWEEP {'guard' if guard else 'raw'}] r={r:<6} fail_rate={rate:.4f}")
        rates.append((r, rate))
    return rates


@torch.no_grad()
def sweep_mul_threshold_poincare(dtype=torch.float64, d=16, kappa=0.5, batch=2048,
                                 r_list=(1,2,5,10,20,50,100,200), device=torch.device('cpu')):
    """
    Threshold sweep: Poincaré scalar multiplication x ↦ r ⊗ x (mobius_scalar_mul), detecting NaN/Inf.
    Inputs are placed at 0.6 of the ball radius so they do not start at the boundary.
    """
    torch.set_default_dtype(dtype)
    dirs = make_unit_dirs(batch, d, device, dtype)
    sqrt_c = math.sqrt(kappa)
    ball_radius = 1.0 / sqrt_c
    pb_frac = 0.6
    X_pb = dirs * (pb_frac * ball_radius)
    rates = []
    for r in r_list:
        r_t = torch.tensor(float(r), dtype=dtype, device=device)
        Y = mobius_scalar_mul(X_pb, r_t, c=kappa)
        rate = failure_rate(Y)
        print(f"[SWEEP Poincare] r={r:<6} fail_rate={rate:.4f}")
        rates.append((r, rate))
    return rates


@torch.no_grad()
def sweep_mul_threshold_lorentz(dtype=torch.float64, d=16, kappa=0.5, batch=2048,
                                r_list=(1,2,5,10,20,50,100,200), device=torch.device('cpu')):
    """
    Threshold sweep: Lorentz scalar multiplication x ↦ exp0( r * log0(x) ), without renormalisation (to expose instability).
    Inputs are hyperboloid points built from logspace radii.
    """
    if not L_AVAILABLE:
        print("[SWEEP Lorentz] LorentzManifold not available; skip.")
        return None
    torch.set_default_dtype(dtype)
    dirs = make_unit_dirs(batch, d, device, dtype)
    radii_pv = torch.logspace(0, 3, steps=batch, base=10.0, dtype=dtype, device=device)
    X_lx = make_lorentz_batch_from_dirs(dirs, radii_pv, kappa, dtype, device)
    L = LorentzManifold(c=kappa)
    rates = []
    for r in r_list:
        r_t = torch.tensor(float(r), dtype=dtype, device=device)
        v = L.logmap0(X_lx, c=kappa)
        Y = L.expmap0(r_t * v, c=kappa)
        rate = failure_rate(Y)
        print(f"[SWEEP Lorentz] r={r:<6} fail_rate={rate:.4f}")
        rates.append((r, rate))
    return rates


# =================== exp/log round-trip tests ===================

@torch.no_grad()
def roundtrip_exp0_log0(kappa=0.5, d=16, batch=4096, tau=10.0, device=torch.device('cpu'), dtype=torch.float32):
    """
    Round-trip error of exp0/log0 for each model:
      v -> y = exp0(v) -> v' = log0(y)  (expect v'≈v)
      y -> v = log0(y) -> y' = exp0(v)  (expect y'≈y)
    """
    torch.set_default_dtype(dtype)
    # Sample tangent vectors (norm set to tau)
    v = torch.randn(batch, d, device=device, dtype=dtype)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9) * tau

    # PV
    M = make_pv_manifold(kappa)
    y_pv = M.exp0(v)
    v_pv_rec = M.log0(y_pv)
    err_pv_v = (v_pv_rec - v).norm(dim=-1).mean().item()
    y_pv_rec = M.exp0(M.log0(y_pv))
    err_pv_y = (y_pv_rec - y_pv).norm(dim=-1).mean().item()

    # Poincaré
    B = PoincareBall()
    y_pb = B.expmap0(v, c=kappa)
    v_pb_rec = B.logmap0(y_pb, c=kappa)
    err_pb_v = (v_pb_rec - v).norm(dim=-1).mean().item()
    y_pb_rec = B.expmap0(B.logmap0(y_pb, c=kappa), c=kappa)
    err_pb_y = (y_pb_rec - y_pb).norm(dim=-1).mean().item()

    # Lorentz
    L = LorentzManifold(c=kappa) if L_AVAILABLE else None
    if L_AVAILABLE:
        y_lx = L.expmap0(v, c=kappa)
        v_lx_rec = L.logmap0(y_lx, c=kappa)
        err_lx_v = (v_lx_rec - v).norm(dim=-1).mean().item()
        y_lx_rec = L.expmap0(L.logmap0(y_lx, c=kappa), c=kappa)
        err_lx_y = (y_lx_rec - y_lx).norm(dim=-1).mean().item()
    else:
        err_lx_v = None
        err_lx_y = None

    print(f"\n[EXP0-LOG0 RT | dtype={dtype}] "
          f"PV: err_v={err_pv_v:.3e} err_y={err_pv_y:.3e} | "
          f"Poincare: err_v={err_pb_v:.3e} err_y={err_pb_y:.3e} | "
          f"Lorentz: err_v={('n/a' if err_lx_v is None else f'{err_lx_v:.3e}')} "
          f"err_y={('n/a' if err_lx_y is None else f'{err_lx_y:.3e}')}")


@torch.no_grad()
def roundtrip_exp_log_at_x(kappa=0.5, d=16, batch=4096, tau_v=5.0, tau_x=5.0,
                           device=torch.device('cpu'), dtype=torch.float32):
    """
    Round-trip error of exp/log at a general point x:
      y = Exp_x(v), v' = Log_x(y)  (expect v'≈v)
      x' = Exp_x(Log_x(y))         (expect x'≈y)
    """
    torch.set_default_dtype(dtype)
    # Scales of the tangent vectors and base points
    v = torch.randn(batch, d, device=device, dtype=dtype)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9) * tau_v

    base = torch.randn(batch, d, device=device, dtype=dtype)
    base = base / base.norm(dim=-1, keepdim=True).clamp_min(1e-9) * tau_x

    # PV
    M = make_pv_manifold(kappa)
    x_pv = M.exp0(base)
    y_pv = M.exp_map(x_pv, v)
    v_pv_rec = M.log_map(x_pv, y_pv)
    err_pv_v = (v_pv_rec - v).norm(dim=-1).mean().item()
    x_pv_rec = M.exp_map(x_pv, M.log_map(x_pv, y_pv))
    err_pv_y = (x_pv_rec - y_pv).norm(dim=-1).mean().item()

    # Poincaré
    B = PoincareBall()
    x_pb = B.expmap0(base, c=kappa)
    y_pb = B.expmap(v, x_pb, c=kappa)
    v_pb_rec = B.logmap(x_pb, y_pb, c=kappa)
    err_pb_v = (v_pb_rec - v).norm(dim=-1).mean().item()
    x_pb_rec = B.expmap(B.logmap(x_pb, y_pb, c=kappa), x_pb, c=kappa)
    err_pb_y = (x_pb_rec - y_pb).norm(dim=-1).mean().item()

    # Lorentz
    L = LorentzManifold(c=kappa) if L_AVAILABLE else None
    if L_AVAILABLE:
        x_lx = L.expmap0(base, c=kappa)
        y_lx = L.expmap(v, x_lx, c=kappa)
        v_lx_rec = L.logmap(x_lx, y_lx, c=kappa)
        err_lx_v = (v_lx_rec - v).norm(dim=-1).mean().item()
        x_lx_rec = L.expmap(L.logmap(x_lx, y_lx, c=kappa), x_lx, c=kappa)
        err_lx_y = (x_lx_rec - y_lx).norm(dim=-1).mean().item()
    else:
        err_lx_v = None
        err_lx_y = None

    print(f"[EXP-LOG@x RT | dtype={dtype}] "
          f"PV: err_v={err_pv_v:.3e} err_y={err_pv_y:.3e} | "
          f"Poincare: err_v={err_pb_v:.3e} err_y={err_pb_y:.3e} | "
          f"Lorentz: err_v={('n/a' if err_lx_v is None else f'{err_lx_v:.3e}')} "
          f"err_y={('n/a' if err_lx_y is None else f'{err_lx_y:.3e}')}")


# =================== Main ===================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--r_list", type=float, nargs="+", default=[1,2,5,10,20,50,75,100,150,200,300,500])
    # Representative r values for the one-shot scalar multiplication diagnostics (run_mul_once)
    parser.add_argument(
        "--r_repr",
        type=float,
        nargs="+",
        default=[1, 2, 5, 10, 20, 50, 75, 100, 150, 200, 300, 500, 700, 1000],
    )
    parser.add_argument("--device", type=str, default="auto")  # auto/cpu/cuda
    args = parser.parse_args()

    device = (torch.device('cuda') if (args.device == "auto" and torch.cuda.is_available()) else
              torch.device(args.device if args.device in ("cpu", "cuda") else "cpu"))

    print(f"Using device: {device}, Lorentz available: {L_AVAILABLE}")

    # dtypes to test: FP16/FP32/FP64 on GPU, FP32/FP64 on CPU
    dtypes = [torch.float32, torch.float64]
    if device.type == "cuda":
        dtypes = [torch.float16] + dtypes

    # ---- Threshold sweeps (PV raw vs guarded) for each precision ----
    for dtype in dtypes:
        sweep_mul_threshold(dtype=dtype, d=args.d, kappa=args.kappa, batch=args.batch//2,
                            r_list=args.r_list, guard=False, device=device)
        sweep_mul_threshold(dtype=dtype, d=args.d, kappa=args.kappa, batch=args.batch//2,
                            r_list=args.r_list + [700, 1000], guard=True, device=device)

        sweep_mul_threshold_poincare(dtype=dtype, d=args.d, kappa=args.kappa,
                                     batch=args.batch//2, r_list=args.r_list + [700, 1000],
                                     device=device)

        sweep_mul_threshold_lorentz(dtype=dtype, d=args.d, kappa=args.kappa,
                                    batch=args.batch//2, r_list=args.r_list + [700, 1000],
                                    device=device)

        for r in args.r_repr:
            run_mul_once(dtype=dtype, d=args.d, kappa=args.kappa, batch=args.batch, r_scale=r, device=device)
        run_add_once(dtype=dtype, d=args.d, kappa=args.kappa, batch=args.batch, device=device)

        # ---- exp/log round-trip tests ----
        roundtrip_exp0_log0(kappa=args.kappa, d=args.d, batch=args.batch, tau=5.0, device=device, dtype=dtype)
        roundtrip_exp_log_at_x(kappa=args.kappa, d=args.d, batch=args.batch, tau_v=5.0, tau_x=5.0, device=device, dtype=dtype)

    # ---- Low precision / memory / time comparison (scalar multiplication) ----
    kappa, d, batch = args.kappa, args.d, args.batch
    def build_inputs_for_mul(dt, dev):
        torch.set_default_dtype(dt)
        dirs = make_unit_dirs(batch, d, dev, dt)
        radii_pv = torch.logspace(0, 3, steps=batch, base=10.0, dtype=dt, device=dev)
        X_pv = dirs * radii_pv.unsqueeze(-1)
        sqrt_c = math.sqrt(kappa); ball_radius = (1.0/sqrt_c)
        pb_frac = 0.6
        X_pb = dirs * (pb_frac * ball_radius)
        X_lx = make_lorentz_batch_from_dirs(dirs, radii_pv, kappa, dt, dev) if L_AVAILABLE else None
        return {"PV": X_pv, "Poincare": X_pb, "Lorentz": X_lx}

    r = 100.0
    M = make_pv_manifold(kappa); B = PoincareBall()
    L = LorentzManifold(c=kappa) if L_AVAILABLE else None

    op_mul = {
        "PV": lambda X: pv_scalar_mul_guarded(X, torch.tensor(r, dtype=X.dtype, device=X.device), kappa, zmax=30.0),
        "Poincare": lambda X: mobius_scalar_mul(X, torch.tensor(r, dtype=X.dtype, device=X.device), c=kappa),
        "Lorentz": (lambda X: lorentz_renorm(L.expmap0(L.logmap0(X, c=kappa) * torch.tensor(r, dtype=X.dtype, device=X.device), c=kappa), kappa)) if L_AVAILABLE else None
    }
    mul_results = run_precision_sweep("scalar-mul", build_inputs_for_mul, op_mul, kappa=kappa, device=device)
    print("\nPRECISION SWEEP (scalar-mul):")
    for rec in mul_results:
        print(rec)

    # ---- Gradient curves (PV/Poincaré/Lorentz scalar multiplication) ----
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    v_seed = make_unit_dirs(batch, d, device, dtype)
    radii = torch.logspace(0, 3, steps=24, base=10., device=device, dtype=dtype)
    r_t = torch.tensor(r, device=device, dtype=dtype)

    pv_r, pv_g = grad_sweep(
        lambda v: pv_scalar_mul_guarded_diff(M.exp0(v), r_t, kappa, zmax=5.0),
        v_seed, radii=radii
    )
    pb_r, pb_g = grad_sweep(
        lambda v: mobius_scalar_mul(B.expmap0(v, c=kappa), r_t, c=kappa),
        v_seed, radii=radii
    )
    print("\nGRAD-SWEEP PV (first 8):", list(zip(pv_r[:8], pv_g[:8])))
    print("GRAD-SWEEP Poincare (first 8):", list(zip(pb_r[:8], pb_g[:8])))
    if L_AVAILABLE:
        def lorentz_op(v):
            v_l = lift_to_lorentz_tangent(v)
            x = L.expmap0(v_l, c=kappa)
            return L.expmap0(L.logmap0(x, c=kappa) * r_t, c=kappa)
        lx_r, lx_g = grad_sweep(lorentz_op, v_seed, radii=radii)
        print("GRAD-SWEEP Lorentz (first 8):", list(zip(lx_r[:8], lx_g[:8])))

if __name__ == '__main__':
    main()