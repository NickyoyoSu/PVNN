import torch
import torch.nn as nn
import math
from .PV_monifold import PVManifold


class GyroBNPV(nn.Module):
    """
    Corrected GyroBN PV implementation, fully aligned with GyroBN_new theory.

    Key corrections:
    1. Use correct gyro inverse gyroinv
    2. Use gyro translation gyrotrans instead of mobius_add
    3. Use gyro scalar multiplication pv_gyro_scalar_mul
    """
    
    def __init__(self, manifold: PVManifold, shape, batchdim=[0], momentum=0.1,
                 track_running_stats=True, init_1st_batch=False, eps=1e-6, max_iter=1000,
                 clamp_factor: float = -1.0,
                 diff_stats: bool = False, use_post_gain: bool = True,
                 max_step: float = 0.5, use_plain_logexp: bool = True,
                 use_euclid_stats: bool = False, use_gyro_midpoint: bool = False,
                 var_floor: float = 1e-3, max_tan_norm: float = 50.0,
                 scalar_sinh_clip: float = 30.0):
        super().__init__()
        self.manifold = manifold
        self.shape = shape
        self.batchdim = batchdim
        self.momentum = momentum
        self.eps = eps
        self.track_running_stats = track_running_stats
        self.init_1st_batch = init_1st_batch
        self.max_iter = max_iter
        
        # Set parameters and running statistics
        self.set_parameters()
        if self.track_running_stats:
            self.register_running_stats()
        
        # Extra tuning
        self.clamp_factor = clamp_factor  # -1 means no clamping
        self.diff_stats = diff_stats
        self.use_post_gain = use_post_gain
        self.max_step = max_step
        self.use_plain_logexp = use_plain_logexp
        # Whether to use tangent-space Euclidean stats (log-Euclidean approx: mu_t=mean(log0(x)); mu=exp0(mu_t))
        self.use_euclid_stats = use_euclid_stats
        # Whether to use geodesic midpoint reduction as batch mean (iterative reduction approx of two-point midpoint)
        self.use_gyro_midpoint = use_gyro_midpoint
        # Numerical stability parameters
        self.var_floor = float(var_floor)
        self.max_tan_norm = float(max_tan_norm)
        # Clamp threshold for sinh input in gyro scalar mul (relaxing increases dynamic range)
        self.scalar_sinh_clip = float(scalar_sinh_clip)
        # Learnable geometric gain after normalization, to restore sufficient feature magnitude
        self.post_gain = nn.Parameter(torch.tensor(1.5, dtype=torch.get_default_dtype()))
    
    def _get_param_shapes(self):
        """Get parameter shapes."""
        shape_prefix = list(self.shape[:-1])
        mean_shape = shape_prefix + [self.shape[-1]]
        # Align with GyroBNBase/GyroBNH: variance and shift are scalars
        var_shape = shape_prefix + [1] if shape_prefix else [1]
        return mean_shape, var_shape
    
    def set_parameters(self):
        """Set learnable parameters."""
        mean_shape, var_shape = self._get_param_shapes()
        # Weight: vector in tangent space
        self.weight = nn.Parameter(torch.zeros(*mean_shape))
        # Scale: align with GyroBN_new, use scalar (or broadcast by prefix dims)
        self.shift = nn.Parameter(torch.ones(*var_shape))
    
    def register_running_stats(self):
        """Register running statistics."""
        if self.init_1st_batch:
            self.running_mean = None
            self.running_var = None
        else:
            mean_shape, var_shape = self._get_param_shapes()
            # Initialize running mean at origin
            running_mean = torch.zeros(*mean_shape)
            # Initialize running variance to 1
            running_var = torch.ones(*var_shape)
            self.register_buffer('running_mean', running_mean)
            self.register_buffer('running_var', running_var)
    
    def gyroinv(self, x):
        """
        Gyro inverse (PV): gyroinv(x) = -x
        """
        return -x
    
    def gyrotrans(self, x, y, translate='Left'):
        """
        Gyro translation: gyrotrans(x, y) = x ⊕ y
        Correct implementation from GyroBN_new
        """
        if translate == 'Left':
            out = self.manifold.gyro_add(x, y)
        elif translate == 'Right':
            out = self.manifold.gyro_add(y, x)
        else:
            raise ValueError("translate must be 'Left' or 'Right'")
        return out
    
    def pv_gyro_scalar_mul(self,
                        y: torch.Tensor,
                        r: torch.Tensor | float,
                        c: float | None = None,
                        s: float | None = None,
                        eps: float = 1e-12) -> torch.Tensor:
        """
        Proper-Velocity (PV) scalar gyromultiplication:
            r ⊗_PV y = s * sinh( r * asinh( ||y|| / s ) ) * y / ||y||
        with s = 1/sqrt(c). Handles r as scalar or broadcastable tensor.

        Args:
            y: (..., d) PV vector
            r: scalar or tensor broadcastable to y[..., 1]
            c: positive curvature scale parameter (K = -c < 0)
            s: optional PV scale; if provided, c is ignored. Must satisfy s = 1/sqrt(c).
            eps: numerical epsilon

        Returns:
            (..., d) tensor = r ⊗_PV y
        """
        # --- choose s ---
        if s is None:
            if c is not None:
                s_val = 1.0 / math.sqrt(float(c))
            else:
                # derive from manifold (prefer precomputed s)
                s_val = float(getattr(self.manifold, 's', 1.0 / math.sqrt(float(getattr(self.manifold, 'c', 1.0)))))
        else:
            s_val = float(s)
        s = torch.as_tensor(s_val, dtype=y.dtype, device=y.device)

        # --- sanitize input then norms & unit direction (protect zero) ---
        y = torch.nan_to_num(y)
        norm_y = y.norm(dim=-1, keepdim=True)
        unit_y = y / norm_y.clamp_min(eps)

        # asinh(||y|| / s); allow r as tensor (auto broadcast)
        a = torch.asinh(norm_y / s)                 # (...,1)
        ra = torch.as_tensor(r, dtype=y.dtype, device=y.device) * a
        # Minimal invasive: clamp sinh input to avoid overflow (configurable)
        ra = torch.clamp(ra, -self.scalar_sinh_clip, self.scalar_sinh_clip)
        coef = s * torch.sinh(ra) / norm_y.clamp_min(eps)

        out = coef * y
        # exact r ⊗ 0 = 0
        return torch.where(norm_y <= eps, torch.zeros_like(out), out)
    
    def frechet_mean(self, x, max_iter=1000):
        """
        Compute Fréchet mean on PV manifold.
        Uses iterative method, aligned with GyroBN_new.
        """
        # For PV manifold, use iterative method for Fréchet mean (vectorized)
        x_flat = x.view(-1, x.shape[-1])  # (N, d)

        # Initialize mean as first point
        mean = x_flat[0:1].clone()  # (1, d)

        # Support "infinite iteration" for convergence (max_iter < 0 means until converge or safety cap)
        iters = 0
        safety_cap = 5000 if max_iter < 0 else max_iter
        tol = 1e-6
        last_applied_step_norm = 0.0
        while iters < safety_cap:
            # Vectorized log_map(mean, x_i) for all i
            # PV_monifold interface: log_map(u, w)
            log_maps = self.manifold.log_map(mean, x_flat)
            avg_log_map = log_maps.mean(dim=0, keepdim=True)
            # Step damping (configurable)
            step = avg_log_map
            step_norm = torch.linalg.norm(step).clamp_min(1e-8)
            if self.max_step is not None and self.max_step > 0:
                step = step * torch.clamp(self.max_step / step_norm, max=1.0)
            last_applied_step_norm = float(torch.linalg.norm(step).item())

            # Update mean via exponential map
            # PV_monifold interface: exp_map(u, v)
            new_mean = self.manifold.exp_map(mean, step)

            # Convergence check (Euclidean norm)
            if torch.linalg.norm(new_mean - mean) < tol:
                mean = new_mean
                break

            mean = new_mean
            iters += 1

        # Return (d,)
        return mean.squeeze(0)
    
    def frechet_variance(self, x, mean):
        """
        Align with GyroBNBase/GyroBNH: batch mean of squared distance (scalar).
        """
        x_flat = x.view(-1, x.shape[-1])  # (N, d)
        # Compute distance in one pass via broadcasting
        # PV_monifold interface: dist(x, y)
        d = self.manifold.dist(mean.view(1, -1), x_flat)
        variance = (d * d).mean()
        return torch.nan_to_num(variance, nan=0.0).clamp_min(1e-8).view(1)

    def euclid_mean_and_variance(self, x):
        """
        Tangent-space Euclidean mean/variance (log-Euclidean approx BN stats).
        Returns: (mean_on_manifold, scalar_var)
        """
        # Log to origin tangent space
        t = self.manifold.logmap0(x, c=self.manifold.c)
        mu_t = t.mean(dim=0)  # (d,)
        # Exp back to manifold
        mu = self.manifold.expmap0(mu_t, c=self.manifold.c)  # (d,)
        # Scalar variance (align with GyroBNBase, scalar not per-channel)
        diff = t - mu_t
        var = (diff * diff).sum(dim=-1).mean().view(1)
        var = torch.nan_to_num(var, nan=0.0).clamp_min(1e-8)
        return mu, var
    
    def updating_running_statistics(self, batch_mean, batch_var):
        """Update running statistics, aligned with GyroBN_new."""
        if self.running_mean is None:
            self.running_mean = batch_mean.detach()
        else:
            # Update mean via geodesic interpolation (using class geodesic)
            with torch.no_grad():
                self.running_mean = self.geodesic(self.running_mean, batch_mean.detach(), self.momentum)
        
        if self.running_var is None:
            self.running_var = batch_var.detach()
        else:
            # Variance uses linear interpolation
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()
    
    def geodesic(self, x, y, t):
        """
        Geodesic interpolation: geodesic(x, y, t) = exp(x, t * log(x, y))
        Aligned with GyroBN_new
        """
        return self.manifold.exp_map(x, t * self.manifold.log_map(x, y))

    def gyro_midpoint_mean(self, x):
        """
        Tangent-space mean (log-Euclidean): log to origin tangent space, Euclidean mean, exp back to manifold.
        Returns shape (d,).
        """
        x_flat = x.view(-1, x.shape[-1])  # (N,d)
        t = self.manifold.logmap0(x_flat, c=self.manifold.c)  # (N,d)
        mu_t = t.mean(dim=0)  # (d,)
        mu = self.manifold.expmap0(mu_t, c=self.manifold.c)  # (d,)
        return mu
    
    # Multi-center impl removed (_assign_to_centers/gyro_midpoint_means_multi/grouped_variance_scalar)

    def normalization(self, x, input_mean, input_var):
        """
        Perform normalization on PV manifold, fully aligned with GyroBN_new theory.

        Args:
            x: Input tensor
            input_mean: Input mean
            input_var: Input variance

        Returns:
            Normalized tensor
        """
        # 1. Set parameters: map weight to manifold
        # self.weight shape (C,) or higher, must align with input last dim
        weight = self.manifold.expmap0(self.weight, c=self.manifold.c)
        # Use mean, var per GyroBNBase (scalar variance and shift)
        if input_mean.dim() == 1:
            input_mean_b = input_mean.unsqueeze(0).expand_as(x)
        else:
            input_mean_b = input_mean
        input_var_b = input_var.view(1)
        
        # 2. Center: use gyro translation
        inv_input_mean = self.gyroinv(input_mean_b)  # Correct gyro inverse
        x_center = self.gyrotrans(inv_input_mean, x)  # Gyro translation
        x_center = torch.nan_to_num(x_center)
        
        # 3. Scale: use gyro scalar multiplication
        factor = self.shift / (input_var_b + self.eps).sqrt()
        # Variance floor to avoid excessive scaling from tiny variance
        factor = self.shift / (torch.clamp(input_var_b, min=self.var_floor) + self.eps).sqrt()
        if self.clamp_factor is not None and self.clamp_factor > 0:
            factor = factor.clamp(max=self.clamp_factor)
        x_scaled = self.pv_gyro_scalar_mul(x_center, factor)  # Gyro scalar mul
        
        # 4. Bias: use gyro translation
        x_normed = self.gyrotrans(weight, x_scaled)  # Gyro translation

        # 5. Post-normalization geometric gain (optional)
        if self.use_post_gain:
            # Clamp geometric gain magnitude to avoid instability from excessive amplification
            gain = torch.clamp(self.post_gain, 0.5, 3.0)
            x_out = self.pv_gyro_scalar_mul(x_normed, gain)
            return x_out
        else:
            return x_normed
    
    def forward(self, x):
        """
        Forward pass, aligned with GyroBN_new.

        Args:
            x: Input tensor, shape (batch_size, ...)

        Returns:
            Normalized tensor
        """
        # Debug print removed

        if self.training:
            # Training mode: batch stats (configurable diff; optional Euclidean approx)
            if self.diff_stats:
                if self.use_euclid_stats:
                    input_mean, input_var = self.euclid_mean_and_variance(x)
                elif self.use_gyro_midpoint:
                    input_mean = self.gyro_midpoint_mean(x)
                    input_var = self.frechet_variance(x, input_mean)
                else:
                    input_mean = self.frechet_mean(x, self.max_iter)
                    input_var = self.frechet_variance(x, input_mean)
                input_mean = torch.nan_to_num(input_mean, nan=0.0)
                input_var = torch.nan_to_num(input_var, nan=0.0).clamp_min(1e-8)
            else:
                with torch.no_grad():
                    if self.use_euclid_stats:
                        input_mean, input_var = self.euclid_mean_and_variance(x)
                    elif self.use_gyro_midpoint:
                        input_mean = self.gyro_midpoint_mean(x)
                        input_var = self.frechet_variance(x, input_mean)
                    else:
                        input_mean = self.frechet_mean(x, self.max_iter)
                        input_var = self.frechet_variance(x, input_mean)
                    input_mean = torch.nan_to_num(input_mean, nan=0.0)
                    input_var = torch.nan_to_num(input_var, nan=0.0).clamp_min(1e-8)
                    # Fallback: Euclidean stats
                    if torch.isnan(input_mean).any() or torch.isnan(input_var).any():
                        eu_mean = torch.nan_to_num(x.mean(dim=0), nan=0.0)
                        eu_var = torch.nan_to_num(x.var(dim=0, unbiased=False), nan=0.0).mean().view(1).clamp_min(1e-8)
                        input_mean = eu_mean
                        input_var = eu_var
                input_mean = input_mean.detach()
                input_var = input_var.detach()

            if self.track_running_stats:
                self.updating_running_statistics(input_mean, input_var)
        elif self.track_running_stats:
            # Eval mode: use running statistics
            input_mean = self.running_mean
            input_var = self.running_var
            input_mean = torch.nan_to_num(input_mean, nan=0.0)
            input_var = torch.nan_to_num(input_var, nan=0.0).clamp_min(1e-8)
        else:
            # Not tracking stats: recompute
            input_mean = self.frechet_mean(x, self.max_iter)
            input_var = self.frechet_variance(x, input_mean)
            input_mean = torch.nan_to_num(input_mean, nan=0.0)
            input_var = torch.nan_to_num(input_var, nan=0.0).clamp_min(1e-8)
        
        # Debug print removed

        # Perform normalization
        output = self.normalization(x, input_mean, input_var)

        # Debug print removed
        return output
    
    def __repr__(self):
        """String representation."""
        attributes = []
        
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    val_str = f"{value.item():.4f}"
                else:
                    val_str = f"shape={tuple(value.shape)}"
                attributes.append(f"{key}={val_str}")
            else:
                attributes.append(f"{key}={value}")
        
        for name, buffer in self.named_buffers(recurse=False):
            if buffer.numel() == 1:
                val_str = f"{buffer.item():.4f}"
            else:
                val_str = f"shape={tuple(buffer.shape)}"
            attributes.append(f"{name}={val_str}")
        
        return f"{self.__class__.__name__}({', '.join(attributes)})"


class PVGyroBN1d(GyroBNPV):
    """
    Corrected 1D GyroBN, fully aligned with GyroBN_new theory.
    """
    
    def __init__(self, manifold: PVManifold, num_features: int, momentum=0.1,
                 track_running_stats=True, eps=1e-6):
        super().__init__(
            manifold=manifold,
            shape=[num_features],
            batchdim=[0],
            momentum=momentum,
            track_running_stats=track_running_stats,
            eps=eps
        )
    
    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch_size, num_features, sequence_length)

        Returns:
            Normalized tensor, same shape as input
        """
        B, C, L = x.shape

        # Reshape to (B*L, C) for batch norm
        x_reshaped = x.permute(0, 2, 1).contiguous().view(B * L, C)

        # Apply GyroBN
        y_reshaped = super().forward(x_reshaped)

        # Restore original shape
        y = y_reshaped.view(B, L, C).permute(0, 2, 1).contiguous()
        
        return y


# Corrected PVBatchNorm1d using proper GyroBN implementation
class PVBatchNorm1d(nn.Module):
    def __init__(self, manifold: PVManifold, num_features: int, use_gyrobn: bool = True):
        super().__init__()
        self.manifold = manifold
        self.use_gyrobn = use_gyrobn

        if use_gyrobn:
            # Use corrected GyroBN
            self.gyrobn = PVGyroBN1d(manifold, num_features)
        else:
            # Use traditional Euclidean BN
            self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) on PV
        if self.use_gyrobn:
            # Use corrected GyroBN for batch norm on PV manifold
            return self.gyrobn(x)
        else:
            # Use traditional Euclidean BN (via tangent-space mapping)
            B, C, L = x.shape
            xt = self.manifold.logmap0(x.permute(0, 2, 1).contiguous().view(-1, C))
            xt = self.bn(xt.view(B, L, C).permute(0, 2, 1))
            y = self.manifold.expmap0(xt.permute(0, 2, 1).contiguous().view(-1, C)).view(B, L, C).permute(0, 2, 1)
            return y 





            
