import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pv.manifold import PVManifold
from lib.pv.layers import PVManifoldMLR

# From Lorentz_monifold.py
def arcosh(x):
    z = torch.sqrt(torch.clamp_min(x.pow(2) - 1.0, 1e-7))
    return torch.log(x + z)

class LorentzManifold(nn.Module):
    def __init__(self, k=-1.0, learnable=False):  # default to negative curvature
        super().__init__()
        k = abs(k)
        if learnable:
            self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32))
        else:
            self.k = torch.tensor(k, dtype=torch.float32)
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def expmap0(self, u, c=None):
        c_val = c if c is not None else self.k
        K = 1. / c_val
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * torch.cosh(theta)
        res[:, 1:] = sqrtK * torch.sinh(theta) * x / x_norm
        return self.proj(res)

    def proj(self, x):
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.sum(y**2, dim=-1, keepdim=True)

        time_component = torch.sqrt(1.0 / self.k + y_sqnorm)
        return torch.cat([time_component, y], dim=-1)

    def to_lorentz(self, x):
        x_norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
        time_component = torch.sqrt(1.0 / self.k + x_norm_sq)
        x_lorentz = torch.cat([time_component, x], dim=1)
        return self.proj(x_lorentz)

class LorentzManifoldMLR(nn.Module):
    def __init__(self, manifold, in_features, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold
        self.a = nn.Parameter(torch.zeros(num_classes))
        self.z = nn.Parameter(F.pad(torch.zeros(num_classes, in_features-1), (1,0), value=1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_classes))
        stdv = 1. / math.sqrt(self.z.size(1))
        nn.init.uniform_(self.z, -stdv, stdv)
        nn.init.uniform_(self.a, -stdv, stdv)
        if bias:
            nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x):
        # Ensure the input is expressed in Lorentz coordinates
        x_lorentz = self.manifold.to_lorentz(x)

        # Hyperplane parameterization
        c_tensor = torch.tensor(self.manifold.c, device=x.device, dtype=x.dtype)
        sqrt_mK = 1/torch.sqrt(c_tensor)
        norm_z = torch.norm(self.z, dim=-1)
        w_t = torch.sinh(sqrt_mK*self.a)*norm_z
        w_s = torch.cosh(sqrt_mK*self.a.view(-1,1))*self.z
        beta = torch.sqrt(torch.clamp(-w_t**2+torch.norm(w_s, dim=-1)**2, min=1e-15))

        # Compute class-wise distances in a single pass
        alpha = -w_t*x_lorentz[:,0:1] + torch.matmul(x_lorentz[:,1:], w_s.transpose(0,1))
        d = torch.sqrt(c_tensor)*torch.abs(torch.asinh(sqrt_mK*alpha/beta.view(1,-1)))

        # Signed distance
        logits = torch.sign(alpha)*beta.view(1,-1)*d

        # Apply bias if available
        if self.has_bias:
            logits = logits + self.bias

        return logits

# From HNN_layers.py
class PoincareBall(nn.Module):
    def __init__(self, k=-1.0, learnable=False):  # default to negative curvature
        super().__init__()
        k = abs(k)  # always operate on |k|
        if learnable:
            self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32))
        else:
            self.k = torch.tensor(k, dtype=torch.float32)
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def expmap0(self, u):
        sqrt_k = torch.sqrt(self.k)
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = torch.tanh(sqrt_k * u_norm) * u / (sqrt_k * u_norm)
        return gamma_1

    def projx(self, x):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / torch.sqrt(self.k)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

class HyperbolicMLR(nn.Module):
    def __init__(self, manifold, in_features, num_classes):
        super().__init__()
        self.manifold = manifold
        self.weight = nn.Parameter(torch.empty(in_features, num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        std = (in_features ** -0.5)
        nn.init.normal_(self.weight, 0, std)
        nn.init.zeros_(self.bias)


    def forward(self, x):
        print("[DEBUG] HyperbolicMLR input x has nan:", torch.isnan(x).any().item(), "range:", x.min().item(), x.max().item())
        x = torch.clamp(x, -1e3, 1e3)
        x = self.manifold.projx(x)
        print("[DEBUG] HyperbolicMLR after projx x has nan:", torch.isnan(x).any().item(), "range:", x.min().item(), x.max().item())
        weight_t = self.weight.t()
        weight_norm = weight_t.norm(dim=1)
        print("[DEBUG] HyperbolicMLR weight_norm has nan:", torch.isnan(weight_norm).any().item(), "range:", weight_norm.min().item(), weight_norm.max().item())
        weight_unit = weight_t / weight_norm.unsqueeze(1).clamp_min(1e-15)
        print("[DEBUG] HyperbolicMLR weight_unit has nan:", torch.isnan(weight_unit).any().item(), "range:", weight_unit.min().item(), weight_unit.max().item())
        # Support both PoincaréBall (k<0) and Lorentz (k>0) by taking |k|
        k = self.manifold.k
        if torch.is_tensor(k):
            c = torch.abs(k).to(x.device, x.dtype).clamp_min(1e-15)
        else:
            try:
                c = abs(float(k))
            except Exception:
                c = 1.0
        rc = torch.sqrt(c)
        print("[DEBUG] HyperbolicMLR rc:", rc.item() if torch.numel(rc)==1 else rc)
        bias_clamped = torch.clamp(self.bias, -10.0, 10.0)
        drcr = 2.0 * rc * bias_clamped
        drcr = torch.clamp(drcr, -15.0, 15.0)
        print("[DEBUG] HyperbolicMLR drcr has nan:", torch.isnan(drcr).any().item(), "range:", drcr.min().item(), drcr.max().item())
        rcx = rc * x
        print("[DEBUG] HyperbolicMLR rcx has nan:", torch.isnan(rcx).any().item(), "range:", rcx.min().item(), rcx.max().item())
        cx2 = rcx.pow(2).sum(dim=1, keepdim=True)
        cx2 = torch.clamp(cx2, max=0.999)
        print("[DEBUG] HyperbolicMLR cx2 has nan:", torch.isnan(cx2).any().item(), "range:", cx2.min().item(), cx2.max().item())
        prod = torch.matmul(rcx, weight_unit.t())
        print("[DEBUG] HyperbolicMLR prod has nan:", torch.isnan(prod).any().item(), "range:", prod.min().item(), prod.max().item())
        numer = 2.0 * prod * torch.cosh(drcr) - (1.0 + cx2) * torch.sinh(drcr)
        print("[DEBUG] HyperbolicMLR numer has nan:", torch.isnan(numer).any().item(), "range:", numer.min().item(), numer.max().item())
        denom = torch.clamp_min(1.0 - cx2, 1e-15)
        print("[DEBUG] HyperbolicMLR denom has nan:", torch.isnan(denom).any().item(), "range:", denom.min().item(), denom.max().item())
        asinh_input = numer / denom
        asinh_input = torch.clamp(asinh_input, -15.0, 15.0)
        print("[DEBUG] HyperbolicMLR asinh_input has nan:", torch.isnan(asinh_input).any().item(), "range:", asinh_input.min().item(), asinh_input.max().item())
        dist = 2.0 * weight_norm / rc * torch.asinh(asinh_input)
        print("[DEBUG] HyperbolicMLR dist has nan:", torch.isnan(dist).any().item(), "range:", dist.min().item(), dist.max().item())
        result = dist  # Can flip the sign depending on logits definition
        result = torch.clamp(result, -1e3, 1e3)
        result = torch.nan_to_num(result, nan=0.0)
        
        return result


def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    """Unidirectional hyperbolic logistic regression helper."""
    if not torch.is_tensor(c):
        c = torch.tensor(c, device=x.device, dtype=x.dtype)
    c = torch.abs(c)  # Ensure positive curvature magnitude

    # Parameter preprocessing
    rc = torch.sqrt(c)
    r = torch.clamp(r, -10.0, 10.0)
    drcr = 2. * rc * r
    drcr = torch.clamp(drcr, -15.0, 15.0)

    # Input transform
    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)
    cx2 = torch.clamp(cx2, max=0.999)
    
    prod = torch.matmul(rcx, z_unit)
    
    # Hyperplane distance score
    numer = 2. * prod * torch.cosh(drcr) - (1. + cx2) * torch.sinh(drcr)
    denom = torch.clamp_min(1. - cx2, 1e-15)
    asinh_input = numer / denom
    asinh_input = torch.clamp(asinh_input, -15.0, 15.0)
    result = 2 * z_norm / rc * torch.asinh(asinh_input)
    
    result = torch.clamp(result, -1e3, 1e3)
    result = torch.nan_to_num(result, nan=0.0)
    
    return result

    
class HNNPlusPlusMLR(nn.Module):
    """Unidirectional hyperbolic classifier head (HNN++)."""

    def __init__(self, manifold, feat_dim, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        # Ensure curvature is represented as a tensor
        c = manifold.k if hasattr(manifold, 'k') else manifold.c
        c = abs(c)
        if isinstance(c, float):
            c_tensor = torch.tensor(c)
        else:
            c_tensor = c

        # Weight init after enforcing positive curvature
        weight = torch.empty(feat_dim, num_classes).normal_(
            mean=0, std=(feat_dim) ** -0.5 / torch.sqrt(c_tensor))

        self.weight_g = nn.Parameter(weight.norm(dim=0))  # magnitude scalar r
        self.weight_v = nn.Parameter(weight)              # direction vector p
        self.bias = nn.Parameter(torch.empty(num_classes), requires_grad=bias)  # distance shift d

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        weight_unit = self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15)

        # Use the updated unidirectional_poincare_mlr (handles negative c already)
        return unidirectional_poincare_mlr(
            x, self.weight_g, weight_unit, self.bias, self.manifold.k)  # pass k instead of c



class KleinManifold(torch.nn.Module):
    """
    Klein Manifold wrapper.
    Provides a unified nn.Module interface for Klein manifold operations.

    """
    def __init__(self, c=-1.0, k=None, learnable=False):  # keep k-compatibility
        super().__init__()
        # Support explicit k; fall back to c when not provided
        if k is not None:
            c = k
        self.k = abs(c)  # operate on |k|
        if learnable:
            self.k = nn.Parameter(torch.tensor(self.k, dtype=torch.float32))
        else:
            self.k = torch.tensor(self.k, dtype=torch.float32)
        self.min_norm = 1e-15
        self.name = "Klein"
        self.c = c
    def proj_tan0(self, u, c=None):
        """
        Tangent space at the origin is Euclidean, so only sanitize numerics.
        """
        # Quietly scrub NaN/Inf without spamming logs
        u_safe = torch.nan_to_num(u, nan=0.0)
        return torch.clamp(u_safe, min=-1e6, max=1e6)

    def proj_tan(self, u, x, c=None):
        """Project onto the tangent space at point x in the Klein model."""
        k_val = c if c is not None else self.k

        # Tangent-space projection formula
        # 1) squared norm of x
        x_sq_norm = torch.sum(x * x, dim=-1, keepdim=True)

        # 2) inner product
        x_u_inner = torch.sum(x * u, dim=-1, keepdim=True)

        # 3) assemble projection
        factor = k_val * x_u_inner / (1 - k_val * x_sq_norm)
        proj_u = u - factor * x
        
        return proj_u
        
    def proj(self, x, c=None):
        """Project onto the Klein disk."""
        k_val = c if c is not None else self.k
        eps = 1e-10

        # Norm computation
        x_norm = torch.norm(x, dim=-1, keepdim=True)

        # Detect anomalies
        if torch.isnan(x_norm).any() or torch.isinf(x_norm).any():
            print("Warning: NaN/Inf detected in Klein proj")
            x_norm = torch.where(torch.isnan(x_norm) | torch.isinf(x_norm),
                            torch.ones_like(x_norm) * 0.01, x_norm)

        # Safe projection
        max_norm = (1.0 / math.sqrt(k_val)) - eps
        mask = x_norm > max_norm
        x_safe = torch.where(mask, x * (max_norm / x_norm), x)
        
        return x_safe


    def expmap0(self, u):
        sqrt_k = torch.sqrt(self.k)
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = torch.tanh(sqrt_k * u_norm) * u / (sqrt_k * u_norm)
        return gamma_1

    def logmap0(self, p):
        """Extra-stable logmap0 implementation."""
        # Detect and repair NaN entries before proceeding
        if torch.isnan(p).any():
            print("Warning: logmap0 input contains NaN; attempting repair")
            p = torch.nan_to_num(p, nan=0.0)

        sqrt_k = torch.sqrt(self.k)
        p_norm = p.norm(dim=-1, p=2, keepdim=True)

        # Safely handle tiny norms
        p_norm = p_norm.clamp_min(self.min_norm)

        # Clamp artanh argument
        arg = sqrt_k * p_norm
        arg = torch.clamp(arg, -0.99, 0.99)

        # Evaluate logmap
        scale = 1. / sqrt_k * torch.atanh(arg) / p_norm
        result = scale * p

        # Final clipping
        return torch.clamp(result, -50.0, 50.0)
    
    def mobius_matvec(self, m, x, c=None):
        k_val = c if c is not None else self.k
        
        # Improvement 1: guard against all-zero rows
        row_norms = torch.norm(m, dim=1)
        zero_rows = row_norms < 1e-8
        if zero_rows.any():
            # Inject a tiny stabilizer
            stabilizer = torch.zeros_like(m)
            zero_indices = zero_rows.nonzero().squeeze(-1)
            stabilizer[zero_indices, 0] = 1e-6
            m = m + stabilizer  # avoid later singularities

        # Improvement 2: use the stable logmap0
        x_tan = self.logmap0(x, c=k_val)

        # Improvement 3: clamp intermediate results
        mx = x_tan @ m.transpose(-1, -2)
        mx = torch.clamp(mx, -1e3, 1e3)  # avoid extreme values

        # Stable expmap0
        result = self.expmap0(mx, c=k_val)

        # Project back into the Klein disk
        return self.proj(result, c=k_val)
        


    def scalar_mul(self, r, x, c=None):
        """
        Einstein scalar multiplication implemented directly in the Klein model.
        r: scalar multiplier
        x: point on the Klein model
        """
        k_val = c if c is not None else self.k

        # Handle x=0 or r=0 explicitly
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        mask = (x_norm < self.min_norm) | (torch.abs(r) < self.min_norm)
        if mask.any():
            return torch.zeros_like(x) if mask.all() else x.clone() * 0

        # Standard implementation
        sqrt_k = torch.sqrt(k_val)
        x_norm_clamp = torch.clamp_min(x_norm, self.min_norm)

        # Compute tanh(r * atanh(||x||))
        atanh_norm = torch.atanh(sqrt_k * x_norm_clamp) / sqrt_k
        new_norm = torch.tanh(r * atanh_norm) / sqrt_k

        # Result = new norm * unit vector
        result = new_norm * x / x_norm_clamp
        return self.proj(result, c=k_val)  # keep inside the Klein disk


    def add(self, x, y, c=None, dim=-1):
        """Einstein addition implemented directly on the Klein model."""
        k_val = c if c is not None else self.k
        x2 = x.pow(2).sum(dim=dim, keepdim=True)  # ||x||^2
        xy = (x * y).sum(dim=dim, keepdim=True)   # <x, y>
        gamma_x = 1 / torch.sqrt(1 - k_val * x2)  # Lorentz factor

        # Einstein addition formula
        num = x + (1 / gamma_x) * y + k_val * (gamma_x / (1 + gamma_x)) * xy * x
        denom = 1 + k_val * xy

        result = num / denom.clamp_min(self.min_norm)
        return self.proj(result, c=k_val)  # keep inside the Klein disk
    
    def mobius_add(self, x, y, c=None, dim=-1):
        """
        Möbius addition is equivalent to Einstein addition on the Klein model.
        This alias keeps the API consistent with other manifolds.
        """
        return self.add(x, y, c, dim)
   

    def dist(self, x, y, c=None):
        """Compute the distance between two Klein points."""
        k_val = c if c is not None else self.k

        # Squared norms
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)

        # Inner product
        xy_inner = torch.sum(x * y, dim=-1, keepdim=True)

        # Klein distance formula
        numerator = 2 * k_val * xy_inner + (1 - k_val * x_norm_sq) * (1 - k_val * y_norm_sq)
        denominator = torch.sqrt((1 - k_val * x_norm_sq) * (1 - k_val * y_norm_sq))
        cosh_term = numerator / denominator.clamp_min(self.min_norm)

        # Use acosh to invert the hyperbolic cosine
        cosh_term = torch.clamp(cosh_term, min=1.0 + 1e-7)
        sqrt_k = torch.sqrt(k_val)
        return torch.acosh(cosh_term) / sqrt_k



class KleinManifoldMLR(nn.Module):
    """
    Multinomial Logistic Regression on the Klein model
    (Ganea et al. 2018; Klein formulation  [oai_citation:1‡arXiv](https://arxiv.org/abs/2410.16813?utm_source=chatgpt.com) [oai_citation:2‡CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2020/papers/Khrulkov_Hyperbolic_Image_Embeddings_CVPR_2020_paper.pdf?utm_source=chatgpt.com))
    """
    def __init__(self, manifold, in_features, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold          # provides proj/logmap0
        self.W = nn.Parameter(torch.empty(num_classes, in_features))
        if bias:
            self.b = nn.Parameter(torch.empty(num_classes))
        else:
            self.register_parameter('b', None)

        # He-uniform initialization
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.W, -bound, bound)
        if bias:
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        # Klein projection & tangent mapping
        x = self.manifold.proj(x)                # (...,d)
        x_tan = self.manifold.logmap0(x)         # curvature is positive now
        
        # Vectorized MLR: logits = x·Wᵀ + b
        logits = x_tan @ self.W.t()                       # (...,C)
        if self.b is not None:
            logits = logits + self.b
        
        # Clamp for stability
        logits = torch.clamp(logits, -1e3, 1e3)
        logits = torch.nan_to_num(logits, nan=0.0)
        
        return logits 
# ... existing code ...

class KleinManifoldMLR(nn.Module):
    """
    Multinomial Logistic Regression on the Klein model
    (Ganea et al. 2018; Klein formulation  [oai_citation:1‡arXiv](https://arxiv.org/abs/2410.16813?utm_source=chatgpt.com) [oai_citation:2‡CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2020/papers/Khrulkov_Hyperbolic_Image_Embeddings_CVPR_2020_paper.pdf?utm_source=chatgpt.com))
    """
    def __init__(self, manifold, in_features, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold          # provides proj/logmap0
        self.W = nn.Parameter(torch.empty(num_classes, in_features))
        if bias:
            self.b = nn.Parameter(torch.empty(num_classes))
        else:
            self.register_parameter('b', None)

        # He-uniform initialization
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.W, -bound, bound)
        if bias:
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        # Klein projection & tangent mapping
        x = self.manifold.proj(x)                # (...,d)
        x_tan = self.manifold.logmap0(x)         # curvature is positive now
        
        # Vectorized MLR: logits = x·Wᵀ + b
        logits = x_tan @ self.W.t()                       # (...,C)
        if self.b is not None:
            logits = logits + self.b
        
        # Clamp for stability
        logits = torch.clamp(logits, -1e3, 1e3)
        logits = torch.nan_to_num(logits, nan=0.0)
        
        return logits 


class EuclideanMLR(nn.Module):
    """
    Multiclass logistic regression head in Euclidean space.
    Standard linear classifier used when no manifold constraints are needed.
    """
    def __init__(self, manifold, in_features, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold  # keep API symmetry even though no manifold ops are required
        self.in_features = in_features
        self.num_classes = num_classes

        # Standard linear layer parameters
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_classes))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Use Xavier initialization."""
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input features [batch_size, in_features]
        Returns:
            logits: classification logits [batch_size, num_classes]
        """
        # Standard Euclidean linear transform: logits = x @ Wᵀ + b
        logits = F.linear(x, self.weight, self.bias)

        # Clamp for numerical stability
        logits = torch.clamp(logits, -1e3, 1e3)
        logits = torch.nan_to_num(logits, nan=0.0)

        return logits

class EuclideanMLR(nn.Module):
    """
    Multiclass logistic regression head in Euclidean space.
    Standard linear classifier used when no manifold constraints are needed.
    """
    def __init__(self, manifold, in_features, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold  # keep API symmetry even though no manifold ops are required
        self.in_features = in_features
        self.num_classes = num_classes

        # Standard linear layer parameters
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_classes))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Use Xavier initialization."""
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input features [batch_size, in_features]
        Returns:
            logits: classification logits [batch_size, num_classes]
        """
        # Standard Euclidean linear transform: logits = x @ Wᵀ + b
        logits = F.linear(x, self.weight, self.bias)

        # Clamp for numerical stability
        logits = torch.clamp(logits, -1e3, 1e3)
        logits = torch.nan_to_num(logits, nan=0.0)

        return logits 