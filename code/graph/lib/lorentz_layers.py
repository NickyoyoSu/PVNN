import pdb

import torch.nn as nn
import torch.nn.functional
import torch.nn.init as init
from .Lorentz_manifold import LorentzManifold
import math
from geoopt import ManifoldParameter
from geoopt.optim.rsgd import RiemannianSGD
from geoopt.optim.radam import RiemannianAdam



class LorentzLayerNorm(nn.Module):
    def __init__(self, manifold, in_features, manifold_out=None):
        super().__init__()
        self.in_features = in_features
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.layer = nn.LayerNorm(self.in_features)
        self.reset_parameters()

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x):
        x_space = x[..., 1:]
        x_space = self.layer(x_space)
        k = self.manifold.c if hasattr(self.manifold, 'c') else self.manifold.k
        x_time = ((x_space**2).sum(dim=-1, keepdim=True) + k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)

        if self.manifold_out is not None:
            k_out = self.manifold_out.c if hasattr(self.manifold_out, 'c') else self.manifold_out.k
            x = x * (k_out / k).sqrt()
        return x

class LorentzNormalization(nn.Module):
    def __init__(self, manifold, manifold_out=None):
        super().__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out

    def forward(self, x):
        x_space = x[..., 1:]
        x_space = x_space / x_space.norm(dim=-1, keepdim=True)
        k = self.manifold.c if hasattr(self.manifold, 'c') else self.manifold.k
        x_time = ((x_space**2).sum(dim=-1, keepdim=True) + k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        
        if self.manifold_out is not None:
            k_out = self.manifold_out.c if hasattr(self.manifold_out, 'c') else self.manifold_out.k
            x = x * (k_out / k).sqrt()
        return x

class LorentzActivation(nn.Module):
    def __init__(self, manifold, activation, manifold_out=None):
        super().__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.activation = activation

    def forward(self, x):
        x_space = x[..., 1:]
        x_space = self.activation(x_space)
        k = self.manifold.c if hasattr(self.manifold, 'c') else self.manifold.k
        x_time = ((x_space**2).sum(dim=-1, keepdim=True) + k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        
        if self.manifold_out is not None:
            k_out = self.manifold_out.c if hasattr(self.manifold_out, 'c') else self.manifold_out.k
            x = x * (k_out / k).sqrt()
        return x

class LorentzDropout(nn.Module):
    def __init__(self, manifold, dropout, manifold_out=None):
        super().__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.training:
            x_space = x[..., 1:]
            x_space = self.dropout(x_space)
            k = self.manifold.c if hasattr(self.manifold, 'c') else self.manifold.k
            x_time = ((x_space**2).sum(dim=-1, keepdim=True) + k).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)
            
            if self.manifold_out is not None:
                k_out = self.manifold_out.c if hasattr(self.manifold_out, 'c') else self.manifold_out.k
                x = x * (k_out / k).sqrt()
        return x

class LorentzLinear(nn.Module):

    def __init__(self, manifold, in_features, out_features, bias=True, manifold_out=None, num_heads=1):
        super().__init__()
        self.in_features = in_features           
        self.out_features = out_features
        self.bias = bias
        self.manifold = manifold
        self.c = manifold.c if hasattr(manifold, 'c') else manifold.k
        self.manifold_out = manifold_out
        self.linear = nn.Linear(self.in_features + 1, self.out_features, bias=bias)
        self.reset_parameters()
        self.num_heads = num_heads

    def reset_parameters(self):
        init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        init.constant_(self.linear.bias, 0)

    def forward(self, x, x_manifold='hyp', return_space=False):
        if x_manifold != 'hyp':
            x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
            x = self.manifold.expmap0(x)
        x_space = self.linear(x)
        if self.num_heads > 1:
            dim_per_head = self.out_features // self.num_heads
            x_space = x_space.reshape(x_space.size(0), x_space.size(1), self.num_heads, dim_per_head)
        if return_space:
            x = x_space
        else:
            x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-8).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.c / self.c).sqrt()
        return x

class LorentzCLS(nn.Module):
    def __init__(self, manifold, in_channels, out_channels, bias=True):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        cls_emb = self.manifold.random_normal((self.out_channels, self.in_channels + 1), 
                                             mean=0, std=1. / math.sqrt(self.in_channels + 1))
        self.cls = ManifoldParameter(cls_emb, self.manifold, requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
            self.has_bias = True
        else:
            self.register_parameter('bias', None)
            self.has_bias = False

    def cinner(self, x, y):
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def forward(self, x, x_manifold='hyp', return_type='neg_dist'):
        if x_manifold != 'hyp':
            x = self.manifold.expmap0(torch.cat([torch.zeros_like(x)[..., 0:1], x], dim=-1))
            
        k = self.manifold.c if hasattr(self.manifold, 'c') else self.manifold.k
        dist = -2 * k - 2 * self.cinner(x, self.cls)
        if self.has_bias:
            dist = dist + self.bias
        dist = dist.clamp(min=0)

        if return_type == 'neg_dist':
            return -dist
        elif return_type == 'prob':
            return 10 / (1.0 + dist)
        elif return_type == 'neg_log_prob':
            return -10*torch.log(1.0 + dist)
        else:
            raise NotImplementedError("不支持的返回类型")

class LorentzOptimizer:
    def __init__(self, model, args):
        euc_optimizer_type = getattr(args, 'euc_optimizer_type', 'adam')
        hyp_optimizer_type = getattr(args, 'hyp_optimizer_type', 'radam')
        euc_lr = getattr(args, 'euc_lr', 0.001)
        hyp_lr = getattr(args, 'hyp_lr', 0.001)
        euc_weight_decay = getattr(args, 'euc_weight_decay', 0.0)
        hyp_weight_decay = getattr(args, 'hyp_weight_decay', 0.0)

        euc_params = [p for n, p in model.named_parameters() 
                     if p.requires_grad and not isinstance(p, ManifoldParameter)]
        hyp_params = [p for n, p in model.named_parameters() 
                     if p.requires_grad and isinstance(p, ManifoldParameter)]

        if euc_optimizer_type == 'adam':
            optimizer_euc = torch.optim.Adam(euc_params, lr=euc_lr, weight_decay=euc_weight_decay)
        elif euc_optimizer_type == 'sgd':
            optimizer_euc = torch.optim.SGD(euc_params, lr=euc_lr, weight_decay=euc_weight_decay)
        else:
            raise NotImplementedError("不支持的欧几里德优化器类型")

        if hyp_params:
            if hyp_optimizer_type == 'radam':
                optimizer_hyp = RiemannianAdam(hyp_params, lr=hyp_lr, stabilize=50, 
                                              weight_decay=hyp_weight_decay)
            elif hyp_optimizer_type == 'rsgd':
                optimizer_hyp = RiemannianSGD(hyp_params, lr=hyp_lr, stabilize=50, 
                                             weight_decay=hyp_weight_decay)
            else:
                raise NotImplementedError("不支持的黎曼优化器类型")
            self.optimizer = [optimizer_euc, optimizer_hyp]
        else:
            self.optimizer = [optimizer_euc]

    def step(self):
        for optimizer in self.optimizer:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizer:
            optimizer.zero_grad()
