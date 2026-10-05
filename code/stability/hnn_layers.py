import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Keep the input in the valid range
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass
        x, = ctx.saved_tensors
        # The derivative of artanh is 1/(1-x²)
        grad_input = grad_output / (1 - x.pow(2)).clamp_min(1e-15)
        return grad_input
    
def artanh(x):
    return Artanh.apply(x)

def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


class PoincareBall():
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def sqdist(self, p1, p2, c):
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    def _lambda_x(self, x, c):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c, max_norm=20.0, eps=1e-15):
        if max_norm is not None and max_norm > 0:
            norm = u.norm(dim=-1, keepdim=True, p=2).clamp_min(eps)
            scale = torch.where(norm > max_norm, max_norm / norm, 1.0)
            u = u * scale
        return u

    def expmap(self, u, p, c):
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def gyro_scalar_mul(self, r, y, c):
        """
        Poincaré-ball scalar gyromultiplication: r ⊗ y = exp_0( r · log_0(y) ).
        This matches the closed form:
            r ⊗ y = tanh( r · artanh(√c · ||y||) ) · y / (√c · ||y||),
        with the convention r ⊗ 0 = 0. Numerically stable at ||y||≈0.
        Args:
            r: scalar or tensor broadcastable to y[...,1]
            y: (..., d) point on the Poincaré ball (spatial coord)
            c: curvature parameter (ball radius = 1/√c)
        """
        return mobius_scalar_mul(y, r, c)

    def mobius_add(self, x, y, c, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m, x, c):
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

def mobius_scalar_mul(x: torch.Tensor, r: torch.Tensor | float, c: float) -> torch.Tensor:
    """
    Möbius scalar multiplication on the Poincaré ball D_c^n.
    x: (..., n), r: scalar (float or 0-d/1-d tensor), c>0 (curvature param of ball has curvature -c)
    returns: same shape as x
    """
    x = torch.as_tensor(x)
    r = torch.as_tensor(r, dtype=x.dtype, device=x.device)
    assert c > 0, "c must be positive"
    sqrt_c = x.new_tensor(c ** 0.5)

    # ‖x‖ and safe unit direction
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    eps = torch.finfo(x.dtype).eps
    direction = torch.where(norm > 0, x / norm, torch.zeros_like(x))

    # clip to keep √c‖x‖ < 1
    z = torch.clamp(sqrt_c * norm, max=1 - 1e-7)

    out_norm = (1.0 / sqrt_c) * torch.tanh(r * torch.atanh(z))
    return direction * out_norm

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, c, dim: int = -1):
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp_(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp0(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def to_hyperboloid(self, x, c):
        K = 1./ c
        sqrtK = K ** 0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)



#=====================================================================================================================================
#=====================================================================================================================================

def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures

class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )

class HNNPlusPlusLinear(nn.Module):
    """
    Hyperbolic linear layer with the HNN++ parameterisation
    """
    def __init__(self, manifold, in_features, out_features, c, dropout=0.0, use_bias=True, gain=1.0):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        
        # Parameterisation: split the weight into magnitude and direction
        weight = torch.empty(in_features, out_features).normal_(
            mean=0, std=(2 * in_features * out_features) ** -0.5 * gain)
        self.weight_g = nn.Parameter(weight.norm(dim=0))  # magnitude
        self.weight_v = nn.Parameter(weight)  # direction
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # Apply dropout to the direction vectors
        if self.training and self.dropout > 0:
            weight_v = F.dropout(self.weight_v, p=self.dropout)
        else:
            weight_v = self.weight_v
        
        # Normalised direction vectors
        weight_unit = weight_v / weight_v.norm(dim=0).clamp_min(1e-15)
        bias = self.bias if self.use_bias else torch.zeros(self.out_features, device=x.device)
        
        # First compute the MLR projection
        mlr_out = unidirectional_poincare_mlr(x, self.weight_g, weight_unit, bias, self.c)
        
        # Then map the result back to the manifold (the key difference from a plain MLR)
        rc = torch.sqrt(self.c) if isinstance(self.c, torch.Tensor) else math.sqrt(self.c)
        sinh_out = torch.sinh(rc * mlr_out) / rc
        
        # Project back into hyperbolic space
        denom = 1 + torch.sqrt(1 + self.c * sinh_out.pow(2).sum(dim=-1, keepdim=True))
        result = sinh_out / denom
        
        # Make sure the result lies on the manifold
        return self.manifold.proj(result, self.c)


class HNNPlusPlusLayer(nn.Module):
    """
    Full HNN++ layer: linear transformation + activation
    """
    def __init__(self, manifold, in_features, out_features, c, dropout=0.0, act=F.relu, use_bias=True):
        super().__init__()
        self.linear = HNNPlusPlusLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.act_fn = act
        self.manifold = manifold
        self.c = c
    
    def forward(self, x):
        # Linear transformation
        h = self.linear(x)
        
        # Apply the activation in the tangent space
        h_tan = self.manifold.logmap0(h, c=self.c)
        h_tan = self.act_fn(h_tan)
        h_tan = self.manifold.proj_tan0(h_tan, c=self.c)
        
        # Map back to the manifold
        h = self.manifold.expmap0(h_tan, c=self.c)
        h = self.manifold.proj(h, c=self.c)
        
        return h

class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
    


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)



def _mobius_add(x, y, k, dim=-1):
    """Möbius addition in Poincaré ball"""
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 + 2 * k * xy + k * y2) * x + (1 - k * x2) * y
    denom = 1 + 2 * k * xy + k ** 2 * x2 * y2
    return num / denom.clamp_min(1e-15)

def clamp_abs(x, eps=1e-15):
    """Clamp absolute values"""
    return torch.clamp(x.abs(), min=eps) * x.sign()

def arsin_k(x, k):
    """Arcsin with curvature k"""
    sqrt_k = torch.sqrt(k)
    return torch.asinh(sqrt_k * x) / sqrt_k

def _dist2plane(
    x: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    k: torch.Tensor,
    keepdim: bool = False,
    signed: bool = False,
    scaled: bool = False,
    dim: int = -1,
):
    """Distance from points to hyperplanes in hyperbolic space"""
    diff = _mobius_add(-p, x, k, dim=dim)
    diff_norm2 = diff.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(1e-15)
    sc_diff_a = (diff * a).sum(dim=dim, keepdim=keepdim)
    if not signed:
        sc_diff_a = sc_diff_a.abs()
    a_norm = a.norm(dim=dim, keepdim=keepdim, p=2)
    num = 2.0 * sc_diff_a
    denom = clamp_abs((1 + k * diff_norm2) * a_norm)
    distance = arsin_k(num / denom, k)
    if scaled:
        distance = distance * a_norm
    return distance

class HyperbolicMLR(nn.Module):
    """
    Multi-class hyperplane classifier in hyperbolic space
    using the standard point-to-hyperplane distance
    """
    
    def __init__(self, manifold, in_features, num_classes):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.num_classes = num_classes
        
        # Hyperplane parameters: normal vector a and point p
        self.a = nn.Parameter(torch.Tensor(num_classes, in_features))  # normal vector
        self.p = nn.Parameter(torch.Tensor(num_classes, in_features))  # point on the hyperplane
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialise the normal vectors
        std = (self.in_features ** -0.5) / (self.manifold.c ** 0.5)
        nn.init.normal_(self.a, mean=0, std=std)
        
        # Initialise the points on the hyperplanes (inside the unit ball)
        nn.init.normal_(self.p, mean=0, std=0.1)
        # Make sure the points are inside the unit ball
        with torch.no_grad():
            p_norm = self.p.norm(dim=1, keepdim=True)
            max_norm = 0.9 / torch.sqrt(torch.tensor(self.manifold.c))
            self.p.data = self.p.data * torch.clamp(max_norm / p_norm, max=1.0)
    
    def forward(self, x):
        # Make sure the input lies on the manifold
        x = self.manifold.proj(x, c=self.manifold.c)
        
        batch_size = x.size(0)
        logits = torch.zeros(batch_size, self.num_classes, device=x.device, dtype=x.dtype)
        
        # Make sure the curvature is a tensor
        k = torch.tensor(self.manifold.c, device=x.device, dtype=x.dtype)
        
        # Distance to the hyperplane of each class
        for i in range(self.num_classes):
            a = self.a[i]  # normal vector
            p = self.p[i]  # point on the hyperplane
            
            # Point-to-hyperplane distance (negated so that points closer to a class score higher)
            distance = _dist2plane(
                x, a.unsqueeze(0).expand(batch_size, -1), 
                p.unsqueeze(0).expand(batch_size, -1), 
                k, signed=True, scaled=False
            )
            
            logits[:, i] = -distance.squeeze(-1)
        
        return logits
    

def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    """Hyperbolic logistic regression with the unidirectional parameterisation"""
    # Make sure c is a tensor
    if not torch.is_tensor(c):
        c = torch.tensor(c, device=x.device, dtype=x.dtype)
    
    # Parameter preprocessing
    rc = torch.sqrt(c)
    drcr = 2. * rc * r
    
    # Input transformation
    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)
    
    # Hyperplane distance scores
    return 2 * z_norm / rc * torch.asinh(
        (2. * torch.matmul(rcx, z_unit) * torch.cosh(drcr) - 
         (1. + cx2) * torch.sinh(drcr)) / torch.clamp_min(1. - cx2, 1e-15)
    )

    
class HNNPlusPlusMLR(nn.Module):
    """Hyperbolic classifier with the unidirectional parameterisation (HNN++)"""
    
    def __init__(self, manifold, feat_dim, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        
        # Make sure the curvature is a tensor
        if isinstance(manifold.c, float):
            c_tensor = torch.tensor(manifold.c)
        else:
            c_tensor = manifold.c
        
        # Initialise the weights
        weight = torch.empty(feat_dim, num_classes).normal_(
            mean=0, std=(feat_dim) ** -0.5 / torch.sqrt(c_tensor))
        
        # Parameterisation: (direction p, magnitude r, offset d)
        self.weight_g = nn.Parameter(weight.norm(dim=0))  # r: magnitude
        self.weight_v = nn.Parameter(weight)              # used to produce the unit vector p
        self.bias = nn.Parameter(torch.empty(num_classes), requires_grad=bias)  # d: offset
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # Normalised direction vectors
        weight_unit = self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15)
        
        # Classification scores
        return unidirectional_poincare_mlr(
            x, self.weight_g, weight_unit, self.bias, self.manifold.c)