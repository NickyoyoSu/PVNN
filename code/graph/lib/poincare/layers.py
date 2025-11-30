import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from .manifold import PoincareBall, artanh, tanh

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
    HNN++ 
    """
    def __init__(self, manifold, in_features, out_features, c, dropout=0.0, use_bias=True, gain=1.0):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        
        # 
        weight = torch.empty(in_features, out_features).normal_(
            mean=0, std=(2 * in_features * out_features) ** -0.5 * gain)
        self.weight_g = nn.Parameter(weight.norm(dim=0))  # 
        self.weight_v = nn.Parameter(weight)  # 
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # dropout
        if self.training and self.dropout > 0:
            weight_v = F.dropout(self.weight_v, p=self.dropout)
        else:
            weight_v = self.weight_v
        
        # 
        weight_unit = weight_v / weight_v.norm(dim=0).clamp_min(1e-15)
        bias = self.bias if self.use_bias else torch.zeros(self.out_features, device=x.device)
        
        # MLR
        mlr_out = unidirectional_poincare_mlr(x, self.weight_g, weight_unit, bias, self.c)
        
        #  - MLR
        rc = torch.sqrt(self.c) if isinstance(self.c, torch.Tensor) else math.sqrt(self.c)
        sinh_out = torch.sinh(rc * mlr_out) / rc
        
        # 
        denom = 1 + torch.sqrt(1 + self.c * sinh_out.pow(2).sum(dim=-1, keepdim=True))
        result = sinh_out / denom
        
        # 
        return self.manifold.proj(result, self.c)


class HNNPlusPlusLayer(nn.Module):
    """
    HNN++ +
    """
    def __init__(self, manifold, in_features, out_features, c, dropout=0.0, act=F.relu, use_bias=True):
        super().__init__()
        self.linear = HNNPlusPlusLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.act_fn = act
        self.manifold = manifold
        self.c = c
    
    def forward(self, x):
        # 
        h = self.linear(x)
        
        # 
        h_tan = self.manifold.logmap0(h, c=self.c)
        h_tan = self.act_fn(h_tan)
        h_tan = self.manifold.proj_tan0(h_tan, c=self.c)
        
        # 
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
    """Mbius addition in Poincar ball"""
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
    """"""
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
    
    
    """
    
    def __init__(self, manifold, in_features, num_classes):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.num_classes = num_classes
        
        # ap
        self.a = nn.Parameter(torch.Tensor(num_classes, in_features))  # 
        self.p = nn.Parameter(torch.Tensor(num_classes, in_features))  # 
        self.reset_parameters()
    
    def reset_parameters(self):
        # 
        std = (self.in_features ** -0.5) / (self.manifold.c ** 0.5)
        nn.init.normal_(self.a, mean=0, std=std)
        
        # 
        nn.init.normal_(self.p, mean=0, std=0.1)
        # 
        with torch.no_grad():
            p_norm = self.p.norm(dim=1, keepdim=True)
            max_norm = 0.9 / torch.sqrt(torch.tensor(self.manifold.c))
            self.p.data = self.p.data * torch.clamp(max_norm / p_norm, max=1.0)
    
    def forward(self, x):
        # 
        x = self.manifold.proj(x, c=self.manifold.c)
        
        batch_size = x.size(0)
        logits = torch.zeros(batch_size, self.num_classes, device=x.device, dtype=x.dtype)
        
        # tensor
        k = torch.tensor(self.manifold.c, device=x.device, dtype=x.dtype)
        
        # 
        for i in range(self.num_classes):
            a = self.a[i]  # 
            p = self.p[i]  # 
            
            # 
            distance = _dist2plane(
                x, a.unsqueeze(0).expand(batch_size, -1), 
                p.unsqueeze(0).expand(batch_size, -1), 
                k, signed=True, scaled=False
            )
            
            logits[:, i] = -distance.squeeze(-1)
        
        return logits
    

def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    """"""
    #  c  tensor
    if not torch.is_tensor(c):
        c = torch.tensor(c, device=x.device, dtype=x.dtype)
    
    # 
    rc = torch.sqrt(c)
    drcr = 2. * rc * r
    
    # 
    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)
    
    # 
    return 2 * z_norm / rc * torch.asinh(
        (2. * torch.matmul(rcx, z_unit) * torch.cosh(drcr) - 
         (1. + cx2) * torch.sinh(drcr)) / torch.clamp_min(1. - cx2, 1e-15)
    )

    
class HNNPlusPlusMLR(nn.Module):
    """HNN++"""
    
    def __init__(self, manifold, feat_dim, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        
        # tensor
        if isinstance(manifold.c, float):
            c_tensor = torch.tensor(manifold.c)
        else:
            c_tensor = manifold.c
        
        # 
        weight = torch.empty(feat_dim, num_classes).normal_(
            mean=0, std=(feat_dim) ** -0.5 / torch.sqrt(c_tensor))
        
        # : (p, r, d)
        self.weight_g = nn.Parameter(weight.norm(dim=0))  # r: 
        self.weight_v = nn.Parameter(weight)              # p
        self.bias = nn.Parameter(torch.empty(num_classes), requires_grad=bias)  # d: 
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # 
        weight_unit = self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15)
        
        # 
        return unidirectional_poincare_mlr(
            x, self.weight_g, weight_unit, self.bias, self.manifold.c)