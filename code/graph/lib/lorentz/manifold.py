"""Hyperboloid manifold."""

from unittest import result
import torch
import torch.nn.functional as F
import math
from ..utils.math_ops import cosh, sinh, asinh
import torch
import torch.nn as nn

# 
def arcosh(x):
    dtype = x.dtype
    z = torch.sqrt(torch.clamp_min(x.pow(2) - 1.0, 1e-7))
    return torch.log(x + z).to(dtype)

@torch.jit.script
def _project(x, k: torch.Tensor, dim: int = -1):
    dn = x.size(dim) - 1
    right_ = x.narrow(dim, 1, dn)
    left_ = torch.sqrt(
        k + (right_ * right_).sum(dim=dim, keepdim=True)
    )
    x = torch.cat((left_, right_), dim=dim)
    return x

class LorentzManifold(nn.Module):
    def __init__(self, c=1.0, in_features=None):
        super().__init__()
        self.c = c
        self.k = c  # k
        self.name = "Lorentz"
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6
        self.in_features = in_features 

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def sqdist(self, x, y, c=None):
        c_val = c if c is not None else self.c
        K = 1. / c_val
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        return torch.clamp(sqdist, max=50.0)

    
    def proj(self, x, c=None):
        """
        
        
        Args:
            x: 
            c: 
        """
        c_val = c if c is not None else self.c
        k = torch.tensor(1.0 / c_val, device=x.device, dtype=x.dtype)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.sum(y**2, dim=-1, keepdim=True)
        y_sqnorm = torch.clamp(y_sqnorm, min=0.0, max=1e6)
        time_component = torch.sqrt(1.0/c_val + y_sqnorm)
        return torch.cat([time_component, y], dim=-1)

    def proj_tan(self, u, x, c=None):
        c_val = c if c is not None else self.c
        K = 1. / c_val
        d = x.size(1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
        mask = torch.ones_like(u)
        mask[:, 0] = 0
        vals = torch.zeros_like(u)
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])
        return vals + mask * u

    def proj_tan0(self, u, c=None):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals

    def expmap(self, u, x, c=None):
        c_val = c if c is not None else self.c
        K = 1. / c_val
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c_val)

    def logmap(self, x, y, c=None):
        c_val = c if c is not None else self.c
        K = 1. / c_val
        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K
        u = y + xy * x * c_val
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c_val) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x, c_val)

    def expmap0(self, u, c=None):
        c_val = c if c is not None else self.c
        K = 1. / c_val
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(res, c_val)

    def logmap0(self, x, c=None):
        c_val = c if c is not None else self.c
        K = 1. / c_val
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res

    def mobius_add(self, x, y, c=None):
        c_val = c if c is not None else self.c
        u = self.logmap0(y, c_val)
        v = self.ptransp0(x, u, c_val)
        return self.expmap(v, x, c_val)

    def mobius_matvec(self, m, x, c=None):
        u = self.logmap0(x)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu)
    '''
    def mobius_matvec(self, m, x, c=None):
        """
        lorentz_linear
        
        Args:
            m:  [out_features, in_features]
            x: Lorentz [batch_size, 1+dim]
            c: 
        """
        c_val = c if c is not None else self.c
        scale = 10.0  # 
        
        # [batch_size, 1+dim]
        if x.size(-1) > m.size(-1):
            # 
            x_space = x.narrow(-1, 1, x.size(-1) - 1)
            
            # 
            transformed = x_space @ m.transpose(-1, -2)
            
            # 
            time = transformed.narrow(-1, 0, 1).sigmoid() * scale + 1.1
            
            # 
            space = transformed.narrow(-1, 1, transformed.size(-1) - 1)
            space = space / ((space * space).sum(dim=-1, keepdim=True) / (time * time - 1)).sqrt()
            
            # 
            result = torch.cat([time, space], dim=-1)
        else:
            #  [batch_size, dim]
            # 
            transformed = x @ m.transpose(-1, -2)
            
            # 
            time = transformed.narrow(-1, 0, 1).sigmoid() * scale + 1.1
            
            # 
            space = transformed.narrow(-1, 1, transformed.size(-1) - 1)
            space = space / ((space * space).sum(dim=-1, keepdim=True) / (time * time - 1)).sqrt()
            
            # 
            result = torch.cat([time, space], dim=-1)
        
        # 
        return self.proj(result, c=c_val)
    '''
    def normalize_tan0(self, p, c=None):
        """
        
        
        Args:
            p: 
            c: 
        """
        c_val = c if c is not None else self.c
        d = p.size(1) - 1
        
        # 
        # Lorentz
        return self.proj_tan0(p, c_val)
    
    def ptransp0(self, x, u, c=None):
        c_val = c if c is not None else self.c
        K = 1. / c_val
        sqrtK = K ** 0.5
        x0 = x.narrow(-1, 0, 1)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=self.min_norm)
        y_normalized = y / y_norm
        v = torch.ones_like(x)
        v[:, 0:1] = -y_norm 
        v[:, 1:] = (sqrtK - x0) * y_normalized
        alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) / sqrtK
        res = u - alpha * v
        return self.proj_tan(res, x, c_val)
    

    def lorentz_to_poincare(self, x, dim=-1):
        dn = x.size(dim) - 1
        beta_sqrt = self.c.reciprocal().sqrt()
        x_space = x[..., 1:]
        x_space = beta_sqrt * x_space
        return x_space / (x.narrow(dim, 0, 1) + beta_sqrt)
    
    def poincare_to_lorentz(self, x, dim=-1, eps=1e-6):
        x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
        res = (
            torch.cat((1/self.c + x_norm_square, 2/self.c * x), dim=dim)
            / (1/self.c - x_norm_square + eps)
        )
        return (self.c.reciprocal().sqrt()) * res
    # 
    def to_lorentz(self, x):
        """"""
        # 
        if x.size(-1) == self.in_features + 1:
            return x
        
        #  - 
        batch_size = x.size(0)
        
        # x
        # : -t^2 + ||x||^2 = -1/c
        x_norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
        time_component = torch.sqrt(1.0/self.c + x_norm_sq)
        
        x_lorentz = torch.cat([time_component, x], dim=1)
        
        # 
        return self.proj(x_lorentz)
        
    def to_euclidean(self, x):
        """Lorentz"""
        return self.logmap0(x)

    def lorentz_plane_distance(
            self, X, N, sigma, c=None):
        """
        
        Args:
            X:  (..., 1+n)
            N:  (..., 1+n)
            sigma:  (..., 1)
            c: 
        """
        c_val = c if c is not None else self.c
        
        # Minkowski
        # X: [60, 17], N: [60, 17] -> b_wx: [60, 1]
        b_wx = (-X[...,0:1]*N[...,0:1] + (X[...,1:]*N[...,1:]).sum(dim=-1,keepdim=True))
        
        #  (Lorentz)
        # N
        # 
        wn = torch.sqrt(torch.clamp((-N[0:1,0]**2+(N[0:1,1:]**2).sum(dim=-1,keepdim=True)), min=1e-15))
        wn = wn.expand(X.size(0), 1)  # [60, 1]
        
        # 
        result = (1.0/c_val) * asinh(c_val * (b_wx - sigma) / wn)
        
        return result
    





class LorentzManifoldMLR(torch.nn.Module):
    def __init__(self, manifold, in_features, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = num_classes
        
        # HyperbolicCV
        self.a = torch.nn.Parameter(torch.zeros(num_classes,))
        self.z = torch.nn.Parameter(F.pad(torch.zeros(num_classes, in_features-1), pad=(1,0), value=1))
        
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(num_classes))
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
            
    def forward(self, x):
        # Lorentz
        x_lorentz = self.manifold.to_lorentz(x)
        
        # 
        c_tensor = torch.tensor(self.manifold.c, device=x.device, dtype=x.dtype)
        sqrt_mK = 1/torch.sqrt(c_tensor)
        norm_z = torch.norm(self.z, dim=-1)
        w_t = torch.sinh(sqrt_mK*self.a)*norm_z
        w_s = torch.cosh(sqrt_mK*self.a.view(-1,1))*self.z
        beta = torch.sqrt(torch.clamp(-w_t**2+torch.norm(w_s, dim=-1)**2, min=1e-15))
        
        # 
        alpha = -w_t*x_lorentz[:,0:1] + torch.matmul(x_lorentz[:,1:], w_s.transpose(0,1))
        d = torch.sqrt(c_tensor)*torch.abs(torch.asinh(sqrt_mK*alpha/beta.view(1,-1)))
        
        # 
        logits = torch.sign(alpha)*beta.view(1,-1)*d
        
        # 
        if self.has_bias:
            logits = logits + self.bias
        
        return logits