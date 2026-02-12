import torch
from . import Lorentz_manifold as lm
import torch.nn as nn
import torch.nn.functional as F
import math            
from .math_utils import tanh, atanh




class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = torch.clamp(x, -0.99 + 1e-7, 0.99 - 1e-7)
        ctx.save_for_backward(x)
        z = x.double()        
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        denominator = (1 - x.pow(2)).clamp_min(1e-7)
        grad_input = grad_output / denominator
        return torch.clamp(grad_input, -1e4, 1e4)

def artanh(x):
    return Artanh.apply(x)

def proj(self, x, c=None):
    k_val = c if c is not None else self.k
    eps = 1e-10
    
    print(f"\n===== Klein projection debug =====")
    print(f"Input shape: {x.shape}, curvature: {k_val}")
    
    has_nan = torch.isnan(x).any()
    has_inf = torch.isinf(x).any()
    if has_nan or has_inf:
        print(f"Warning: input contains {'NaN' if has_nan else ''}{' and ' if has_nan and has_inf else ''}{'Inf' if has_inf else ''}")
    
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    print(f"Norm stats: min={x_norm.min().item():.6f}, max={x_norm.max().item():.6f}, mean={x_norm.mean().item():.6f}")
    
    very_small_norms = (x_norm < 1e-8).sum().item()
    if very_small_norms > 0:
        print(f"Warning: detected {very_small_norms} near-zero norm values")
    
    max_norm = (1.0 / math.sqrt(k_val)) - eps
    print(f"Maximum allowed norm: {max_norm:.6f}")
    
    mask = x_norm > max_norm
    scaling_count = mask.sum().item()
    print(f"Points requiring scaling: {scaling_count}")
    
    safe_x_norm = torch.clamp_min(x_norm, 1e-8)
    scaling_factor = max_norm / safe_x_norm
    
    if torch.isnan(scaling_factor).any() or torch.isinf(scaling_factor).any():
        print(f"Warning: scaling factor contains NaN/Inf, applying fix")
        scaling_factor = torch.clamp(scaling_factor, 0, 1e6)
    
    x_safe = torch.where(mask, x * scaling_factor, x)
    
    if torch.isnan(x_safe).any() or torch.isinf(x_safe).any():
        print(f"Warning: result contains NaN/Inf, using fallback projection")
        norm_ratio = torch.clamp_min(x_norm, 1e-8) / max_norm
        safe_ratio = torch.clamp(norm_ratio, max=0.99)          
        x_safe = x / norm_ratio * safe_ratio
    
    x_safe = torch.nan_to_num(x_safe, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"===== Klein projection complete =====\n")
    return x_safe


class KleinManifold(torch.nn.Module):
    def __init__(self, c=1.0):
        super().__init__()
        self.k = c
        self.min_norm = 1e-15
        self.name = "Klein"
        self.c = c   
    def proj_tan0(self, u, c=None, max_norm=20.0, eps=1e-15):
        u = torch.nan_to_num(u,nan=0.0)
        if max_norm is not None and max_norm > 0:
            norm = u.norm(dim=-1, keepdim=True, p=2).clamp_min(eps)
            scale = torch.where(norm > max_norm, max_norm / norm, 1.0)
            u = u * scale

        return torch.clamp(u, min=-1e6, max=1e6)

    def proj_tan(self, u, x, c=None):
        k_val = c if c is not None else self.k

        x_sq_norm = torch.sum(x * x, dim=-1, keepdim=True)
        
        x_u_inner = torch.sum(x * u, dim=-1, keepdim=True)
        
        factor = k_val * x_u_inner / (1 - k_val * x_sq_norm)
        proj_u = u - factor * x
        
        return proj_u
        
    def proj(self, x, c=None):
        k_val = c if c is not None else self.k
        eps = 1e-10

        x_norm = torch.norm(x, dim=-1, keepdim=True)
        
        x_norm_repaired = x_norm
        has_norm_nan_or_inf = torch.isnan(x_norm).any() or torch.isinf(x_norm).any()
        if has_norm_nan_or_inf:
            x_norm_repaired = torch.where(torch.isnan(x_norm) | torch.isinf(x_norm),
                           torch.zeros_like(x_norm), x_norm)
        
        max_norm = (1.0 / math.sqrt(k_val)) - eps
        mask = x_norm_repaired > max_norm
        safe_x_norm = x_norm_repaired.clamp_min(1e-9)
        
        x_safe = torch.where(mask, x * (max_norm / safe_x_norm), x)
        
        if torch.isnan(x_safe).any():
            print("\n--- NaN detected in Klein.proj, printing debug info ---")
            print(f"Input x shape: {x.shape}, c: {k_val}")
            if torch.isnan(x).any(): print("Warning: proj input x contains NaN")
            if torch.isinf(x).any(): print("Warning: proj input x contains Inf")
            
            print(f"Computed norm (x_norm): min={x_norm.min():.4f}, max={x_norm.max():.4f}, has_nan={torch.isnan(x_norm).any()}, has_inf={torch.isinf(x_norm).any()}")
            if has_norm_nan_or_inf:
                print("-> NaN/Inf detected in norm and repaired")

            print(f"Maximum allowed norm: {max_norm:.4f}, number of points to project: {mask.sum()}")
            print(f"Projected x_safe: has_nan={torch.isnan(x_safe).any()}, has_inf={torch.isinf(x_safe).any()}")
            print("--- Klein.proj debug finished ---\n")
        
        return x_safe


    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        if torch.isnan(p).any():
            print("Warning: logmap0 input contains NaN; attempting repair")
            p = torch.nan_to_num(p, nan=0.0)
        
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True)
        
        p_norm = p_norm.clamp_min(self.min_norm)
        
        arg = sqrt_c * p_norm
        arg = torch.clamp(arg, -0.99, 0.99)
        
        scale = 1. / sqrt_c * artanh(arg) / p_norm
        result = scale * p
        
        return torch.clamp(result, -50.0, 50.0)
    
    def mobius_matvec(self, m, x, c=None):
        k_val = c if c is not None else self.k
        
        row_norms = torch.norm(m, dim=1)
        zero_rows = row_norms < 1e-8
        if zero_rows.any():
            stabilizer = torch.zeros_like(m)
            zero_indices = zero_rows.nonzero().squeeze(-1)
            stabilizer[zero_indices, 0] = 1e-6
            m = m + stabilizer                 
        
        x_tan = self.logmap0(x, c=k_val)
        
        mx = x_tan @ m.transpose(-1, -2)
        mx = torch.clamp(mx, -1e3, 1e3)         
        
        result = self.expmap0(mx, c=k_val)
        
        return self.proj(result, c=k_val)
        


    def scalar_mul(self, r, x, c=None):
        k_val = c if c is not None else self.k
        
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        mask = (x_norm < self.min_norm) | (torch.abs(r) < self.min_norm)
        if mask.any():
            return torch.zeros_like(x) if mask.all() else x.clone() * 0
            
        sqrt_k = torch.sqrt(torch.tensor(k_val, device=x.device, dtype=x.dtype))
        x_norm_clamp = torch.clamp_min(x_norm, self.min_norm)
        
        atanh_norm = torch.atanh(sqrt_k * x_norm_clamp) / sqrt_k
        new_norm = torch.tanh(r * atanh_norm) / sqrt_k
        
        result = new_norm * x / x_norm_clamp
        return self.proj(result, c=k_val)                 


    def add(self, x, y, c=None, dim=-1):
        k_val = c if c is not None else self.k

        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 - k_val * y2) * x + (1 + k_val * x2) * y
        denom = 1 - k_val * xy
        
        result = num / denom.clamp_min(self.min_norm)
        final_result = self.proj(result, c=k_val)
        
        if torch.isnan(final_result).any():
            print("\n--- NaN detected in Klein.add (mobius_add), printing debug info ---")
            print(f"Input x shape: {x.shape}, y shape: {y.shape}, c: {k_val}")
            if torch.isnan(x).any(): print("Warning: input x contains NaN")
            if torch.isinf(x).any(): print("Warning: input x contains Inf")
            if torch.isnan(y).any(): print("Warning: input y contains NaN")
            if torch.isinf(y).any(): print("Warning: input y contains Inf")

            print(f"x2: min={x2.min():.4f}, max={x2.max():.4f}, has_nan={torch.isnan(x2).any()}")
            print(f"y2: min={y2.min():.4f}, max={y2.max():.4f}, has_nan={torch.isnan(y2).any()}")
            print(f"xy: min={xy.min():.4f}, max={xy.max():.4f}, has_nan={torch.isnan(xy).any()}")

            print(f"num: has_nan={torch.isnan(num).any()}, has_inf={torch.isinf(num).any()}")
            print(f"denom: min={denom.min():.4f}, max={denom.max():.4f}, has_nan={torch.isnan(denom).any()}, zeros={torch.sum(denom == 0)}")
            
            print(f"Result after division: has_nan={torch.isnan(result).any()}, has_inf={torch.isinf(result).any()}")
            print(f"Final projection final_result: has_nan={torch.isnan(final_result).any()}")
            print("--- Klein.add (mobius_add) debug finished ---\n")

        return final_result
    
    def mobius_add(self, x, y, c=None, dim=-1):
       return self.add(x, y, c, dim)
   

    def dist(self, x, y, c=None):
        k_val = c if c is not None else self.k
        
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)
        
        xy_inner = torch.sum(x * y, dim=-1, keepdim=True)
        
        numerator = 2 * k_val * xy_inner + (1 - k_val * x_norm_sq) * (1 - k_val * y_norm_sq)
        denominator = torch.sqrt((1 - k_val * x_norm_sq) * (1 - k_val * y_norm_sq))
        cosh_term = numerator / denominator.clamp_min(self.min_norm)
        
        from math_utils import acosh
        sqrt_k = torch.sqrt(torch.tensor(k_val, device=x.device, dtype=x.dtype))
        return acosh(cosh_term) / sqrt_k


class KleinManifoldMLR(nn.Module):
    def __init__(self, manifold, in_features, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = num_classes
        self.min_norm = 1e-15
        
        self.a = nn.Parameter(torch.zeros(num_classes,))
        self.z = nn.Parameter(F.pad(torch.zeros(num_classes, in_features-1), pad=(1,0), value=1))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_classes))
            self.has_bias = True
        else:
            self.register_parameter('bias', None)
            self.has_bias = False
            
        self.init_weights()
    
    def init_weights(self):
        stdv = 1. / math.sqrt(self.z.size(1))
        nn.init.uniform_(self.z, -stdv, stdv)
        nn.init.uniform_(self.a, -stdv, stdv)
        if self.has_bias:
            nn.init.uniform_(self.bias, -stdv, stdv)
    
class KleinManifoldMLR(nn.Module):
    def __init__(self, manifold, in_features, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold                             
        self.W = nn.Parameter(torch.empty(num_classes, in_features))
        if bias:
            self.b = nn.Parameter(torch.empty(num_classes))
        else:
            self.register_parameter('b', None)

        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.W, -bound, bound)
        if bias:
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        k_val = self.manifold.k
        
        x = self.manifold.proj(x, c=k_val)                         
        x_tan = self.manifold.logmap0(x, c=k_val)                  
        
        logits = x_tan @ self.W.t()                                
        if self.b is not None:
            logits = logits + self.b
        
        return logits 
    

if __name__ == "__main__":
    k=0.7
    n = 8

    x = torch.randn(10, n) * 0.3
    x = x / torch.norm(x, dim=-1, keepdim=True) * (0.9 / torch.sqrt(torch.tensor(k)))
    
    X = klein_to_lorentz(x, k)
    
    constraint = -X[...,0:1].pow(2) + X[...,1:].pow(2).sum(dim=-1, keepdim=True)
    target = torch.tensor(-1.0/k)
    print(f"Constraint error: {torch.abs(constraint - target).max().item():.8f}")
    
    x2 = lorentz_to_klein(X, k)
    
    print(f"Conversion error: {torch.abs(x - x2).max().item():.8f}")
    assert torch.allclose(x, x2, atol=1e-6)
