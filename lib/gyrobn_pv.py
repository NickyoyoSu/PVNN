import torch
import torch.nn as nn
import math
from .PV_manifold import PVManifold


class GyroBNPV(nn.Module):
    
    def __init__(self, manifold: PVManifold, shape, batchdim=[0], momentum=0.1,
                 track_running_stats=True, init_1st_batch=False, eps=1e-6, max_iter=1000,
                 clamp_factor: float = -1.0, print_stats: bool = False,
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
        
        self.set_parameters()
        if self.track_running_stats:
            self.register_running_stats()
        
        self.clamp_factor = clamp_factor            
        self.print_stats = print_stats
        self.diff_stats = diff_stats
        self.use_post_gain = use_post_gain
        self.max_step = max_step
        self.use_plain_logexp = use_plain_logexp
        self.use_euclid_stats = use_euclid_stats
        self.use_gyro_midpoint = use_gyro_midpoint
        self.var_floor = float(var_floor)
        self.max_tan_norm = float(max_tan_norm)
        self.scalar_sinh_clip = float(scalar_sinh_clip)
        self._dbg_seen = 0
        self._dbg_print_limit = 3
        self._last_mean_iters = 0
        self._last_step_norm = 0.0
        self.post_gain = nn.Parameter(torch.tensor(1.5, dtype=torch.get_default_dtype()))
    
    def _get_param_shapes(self):
        shape_prefix = list(self.shape[:-1])
        mean_shape = shape_prefix + [self.shape[-1]]
        var_shape = shape_prefix + [1] if shape_prefix else [1]
        return mean_shape, var_shape
    
    def set_parameters(self):
        mean_shape, var_shape = self._get_param_shapes()
        self.weight = nn.Parameter(torch.zeros(*mean_shape))
        self.shift = nn.Parameter(torch.ones(*var_shape))
    
    def register_running_stats(self):
        if self.init_1st_batch:
            self.running_mean = None
            self.running_var = None
        else:
            mean_shape, var_shape = self._get_param_shapes()
            running_mean = torch.zeros(*mean_shape)
            running_var = torch.ones(*var_shape)
            self.register_buffer('running_mean', running_mean)
            self.register_buffer('running_var', running_var)
    
    def gyroinv(self, x):
        return -x
    
    def gyrotrans(self, x, y, translate='Left'):
        if translate == 'Left':
            out = self.manifold.gyro_add(x, y)
        elif translate == 'Right':
            out = self.manifold.gyro_add(y, x)
        else:
            raise ValueError("translate must be 'Left' or 'Right'")
        if torch.isnan(out).any() or torch.isinf(out).any():
            with torch.no_grad():
                print("[GyroBNPV][diag] NaN/Inf after gyrotrans; x/y absmax=",
                      float(x.abs().max().item()), float(y.abs().max().item()))
        return out
    
    def pv_gyro_scalar_mul(self,
                        y: torch.Tensor,
                        r: torch.Tensor | float,
                        c: float | None = None,
                        s: float | None = None,
                        eps: float = 1e-12) -> torch.Tensor:
        if s is None:
            if c is not None:
                s_val = 1.0 / math.sqrt(float(c))
            else:
                s_val = float(getattr(self.manifold, 's', 1.0 / math.sqrt(float(getattr(self.manifold, 'c', 1.0)))))
        else:
            s_val = float(s)
        s = torch.as_tensor(s_val, dtype=y.dtype, device=y.device)

        y = torch.nan_to_num(y)
        norm_y = y.norm(dim=-1, keepdim=True)
        unit_y = y / norm_y.clamp_min(eps)

        a = torch.asinh(norm_y / s)                          
        ra = torch.as_tensor(r, dtype=y.dtype, device=y.device) * a
        ra = torch.clamp(ra, -self.scalar_sinh_clip, self.scalar_sinh_clip)
        coef = s * torch.sinh(ra) / norm_y.clamp_min(eps)

        out = coef * y
        if torch.isnan(out).any() or torch.isinf(out).any():
            with torch.no_grad():
                print("[GyroBNPV][diag] NaN/Inf after scalar_mul; norm_y min/max=",
                      float(norm_y.min().item()), float(norm_y.max().item()))
        return torch.where(norm_y <= eps, torch.zeros_like(out), out)
    
    def frechet_mean(self, x, max_iter=1000):
        x_flat = x.view(-1, x.shape[-1])          

        mean = x_flat[0:1].clone()          

        iters = 0
        safety_cap = 5000 if max_iter < 0 else max_iter
        tol = 1e-6
        last_applied_step_norm = 0.0
        while iters < safety_cap:
            log_maps = self.manifold.log_map(mean, x_flat)
            avg_log_map = log_maps.mean(dim=0, keepdim=True)
            step = avg_log_map
            step_norm = torch.linalg.norm(step).clamp_min(1e-8)
            if self.max_step is not None and self.max_step > 0:
                step = step * torch.clamp(self.max_step / step_norm, max=1.0)
            last_applied_step_norm = float(torch.linalg.norm(step).item())

            new_mean = self.manifold.exp_map(mean, step)

            if torch.linalg.norm(new_mean - mean) < tol:
                mean = new_mean
                break

            mean = new_mean
            iters += 1

        self._last_mean_iters = iters
        self._last_step_norm = last_applied_step_norm
        return mean.squeeze(0)
    
    def frechet_variance(self, x, mean):
        x_flat = x.view(-1, x.shape[-1])          
        d = self.manifold.dist(mean.view(1, -1), x_flat)
        variance = (d * d).mean()
        return torch.nan_to_num(variance, nan=0.0).clamp_min(1e-8).view(1)

    def euclid_mean_and_variance(self, x):
        t = self.manifold.logmap0(x, c=self.manifold.c)
        mu_t = t.mean(dim=0)        
        mu = self.manifold.expmap0(mu_t, c=self.manifold.c)        
        diff = t - mu_t
        var = (diff * diff).sum(dim=-1).mean().view(1)
        var = torch.nan_to_num(var, nan=0.0).clamp_min(1e-8)
        return mu, var
    
    def updating_running_statistics(self, batch_mean, batch_var):
        if self.running_mean is None:
            self.running_mean = batch_mean.detach()
        else:
            with torch.no_grad():
                self.running_mean = self.geodesic(self.running_mean, batch_mean.detach(), self.momentum)
        
        if self.running_var is None:
            self.running_var = batch_var.detach()
        else:
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()
    
    def geodesic(self, x, y, t):
        return self.manifold.exp_map(x, t * self.manifold.log_map(x, y))

    def gyro_midpoint_mean(self, x):
        x_flat = x.view(-1, x.shape[-1])         
        t = self.manifold.logmap0(x_flat, c=self.manifold.c)         
        mu_t = t.mean(dim=0)        
        mu = self.manifold.expmap0(mu_t, c=self.manifold.c)        
        return mu
    
    
    def normalization(self, x, input_mean, input_var):
        weight = self.manifold.expmap0(self.weight, c=self.manifold.c)
        if input_mean.dim() == 1:
            input_mean_b = input_mean.unsqueeze(0).expand_as(x)
        else:
            input_mean_b = input_mean
        input_var_b = input_var.view(1)
        
        inv_input_mean = self.gyroinv(input_mean_b)             
        x_center = self.gyrotrans(inv_input_mean, x)          
        x_center = torch.nan_to_num(x_center)
        if torch.isnan(x_center).any() or torch.isinf(x_center).any():
            with torch.no_grad():
                print("[GyroBNPV][diag] NaN/Inf after center; var=", float(input_var_b.mean().item()))
        
        factor = self.shift / (input_var_b + self.eps).sqrt()
        factor = self.shift / (torch.clamp(input_var_b, min=self.var_floor) + self.eps).sqrt()
        if self.clamp_factor is not None and self.clamp_factor > 0:
            factor = factor.clamp(max=self.clamp_factor)
        if torch.isnan(factor).any() or torch.isinf(factor).any():
            with torch.no_grad():
                print("[GyroBNPV][diag] NaN/Inf in factor; shift mean=",
                      float(self.shift.mean().item()), " var_floor=", self.var_floor)
        x_scaled = self.pv_gyro_scalar_mul(x_center, factor)            
        if torch.isnan(x_scaled).any() or torch.isinf(x_scaled).any():
            with torch.no_grad():
                print("[GyroBNPV][diag] NaN/Inf after scale; factor min/max/mean=",
                      float(factor.min().item()), float(factor.max().item()), float(factor.mean().item()))
        
        x_normed = self.gyrotrans(weight, x_scaled)          
        if torch.isnan(x_normed).any() or torch.isinf(x_normed).any():
            with torch.no_grad():
                print("[GyroBNPV][diag] NaN/Inf after bias (gyrotrans)")

        if self.print_stats and self._dbg_seen < self._dbg_print_limit:
            with torch.no_grad():
                def _s(t):
                    t = t if isinstance(t, torch.Tensor) else torch.tensor(t)
                    return dict(min=float(t.min().item()), max=float(t.max().item()),
                                mean=float(t.mean().item()), std=float(t.std(unbiased=False).item()))
                fnorm = torch.linalg.norm(factor)
                xcn = torch.linalg.norm(x_center) / (x_center.numel() ** 0.5)
                xsn = torch.linalg.norm(x_scaled) / (x_scaled.numel() ** 0.5)
                iv = torch.nan_to_num(input_var_b, nan=0.0).view(-1)[0]
                print(f"[GyroBNPV] iters={self._last_mean_iters} step_norm={self._last_step_norm:.3e} factor_norm={float(fnorm.item()):.3e} x_center_rms={float(xcn.item()):.3e} x_scaled_rms={float(xsn.item()):.3e}")
                print(f"[GyroBNPV] factor stats: {_s(factor)}  shift={_s(self.shift)}")
                print(f"[GyroBNPV] input_var={float(iv.item()):.6e}  post_gain={float(self.post_gain.item()):.3f}")
                self._dbg_seen += 1

        if self.use_post_gain:
            gain = torch.clamp(self.post_gain, 0.5, 3.0)
            x_out = self.pv_gyro_scalar_mul(x_normed, gain)
            return x_out
        else:
            return x_normed
    
    def forward(self, x):

        if self.training:
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
            input_mean = self.running_mean
            input_var = self.running_var
            input_mean = torch.nan_to_num(input_mean, nan=0.0)
            input_var = torch.nan_to_num(input_var, nan=0.0).clamp_min(1e-8)
        else:
            input_mean = self.frechet_mean(x, self.max_iter)
            input_var = self.frechet_variance(x, input_mean)
            input_mean = torch.nan_to_num(input_mean, nan=0.0)
            input_var = torch.nan_to_num(input_var, nan=0.0).clamp_min(1e-8)
        

        output = self.normalization(x, input_mean, input_var)

        return output
    
    def __repr__(self):
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
        B, C, L = x.shape
        
        x_reshaped = x.permute(0, 2, 1).contiguous().view(B * L, C)
        
        y_reshaped = super().forward(x_reshaped)
        
        y = y_reshaped.view(B, L, C).permute(0, 2, 1).contiguous()
        
        return y


class PVBatchNorm1d(nn.Module):
    def __init__(self, manifold: PVManifold, num_features: int, use_gyrobn: bool = True):
        super().__init__()
        self.manifold = manifold
        self.use_gyrobn = use_gyrobn
        
        if use_gyrobn:
            self.gyrobn = PVGyroBN1d(manifold, num_features)
        else:
            self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gyrobn:
            return self.gyrobn(x)
        else:
            B, C, L = x.shape
            xt = self.manifold.logmap0(x.permute(0, 2, 1).contiguous().view(-1, C))
            xt = self.bn(xt.view(B, L, C).permute(0, 2, 1))
            y = self.manifold.expmap0(xt.permute(0, 2, 1).contiguous().view(-1, C)).view(B, L, C).permute(0, 2, 1)
            return y 





            
