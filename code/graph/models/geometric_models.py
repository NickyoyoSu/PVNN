import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from lib.pv.graph_ops import PVManifold, PVManifoldMLR, PVFC
from lib.lorentz.graph_manifold import LorentzManifold, LorentzManifoldMLR
from lib.lorentz.lorentz_layers import LorentzLinear, LorentzActivation, LorentzDropout, LorentzLayerNorm
from lib.klein.manifold import KleinManifold, KleinManifoldMLR
from lib.math_utils import artanh, tanh
from lib.poincare.hnn_manifold import (
    PoincareBall,
    HNNLayer,
    HyperbolicMLR,
    HNNPlusPlusMLR,
    HNNPlusPlusLayer,
)
import torch.nn.init as init
import sys

ACT_MAP = {
    "identity": nn.Identity,
    "tanh"    : nn.Tanh,
    "softplus": lambda : nn.Softplus(beta=1, threshold=20.)
}


def _as_negative_curvature(value):
    if value is None:
        return -1.0
    return value if value < 0 else -abs(value)


def _resolve_pv_curvature(manifold, default=-1.0):
    k_attr = getattr(manifold, "k", None)
    if k_attr is not None:
        return k_attr if k_attr < 0 else -abs(k_attr)
    c_attr = getattr(manifold, "c", None)
    if c_attr is not None:
        return -abs(c_attr)
    return default
def check_nan(tensor, location_name):
    if torch.isnan(tensor).any():
        print(f"*** NaN detected at: {location_name} ***")
        print(f"Tensor shape: {tensor.shape}")
        nan_indices = torch.nonzero(torch.isnan(tensor))
        if nan_indices.numel() > 0:
            print(f"NaN indices: {nan_indices[:10]}...")           
        sys.exit(1)          

class GeometricModel(nn.Module):
    def __init__(self, model_type, dim=16,hidden_dim = 16, n_classes=2, c=1.0, p_drop: float = 0.0, final_act: str = "softplus",inner_act = 'none',outer_act = 'tangent', linear_type: str = 'pvfc'):
        super().__init__()
        self.dropout = p_drop 
        self.model_type = model_type.lower()
        act_cls = ACT_MAP[final_act.lower()]
        self.final_act = act_cls()  
        self.in_features = dim             
        self.hidden_dim = hidden_dim         
        self.out_features = n_classes
        self.inner_act = inner_act
        self.outer_act = outer_act
        self.linear_type = linear_type

        if self.model_type == 'fc':
            self.manifold = None
            self.layer1 = nn.Linear(dim, dim)
            self.activation = nn.ReLU()
            self.classifier = nn.Linear(dim, n_classes)
        elif self.model_type in ['hnn', 'hnn++']:
            self.manifold = PoincareBall()
            
            if self.model_type == 'hnn++':
                print("Building parametrized HNN++ with learnable curvature")
                self.c = c
            else:
                print("Building standard HNN")
                self.c = c
                
            self.manifold.c = self.c
            
            if self.model_type == 'hnn++':
                self.layers = nn.Sequential(
                    HNNPlusPlusLayer(self.manifold, self.in_features, self.hidden_dim, self.c, p_drop, lambda x: x),
                    HNNPlusPlusLayer(self.manifold, self.hidden_dim, self.hidden_dim, self.c, p_drop, lambda x: x)
                )
            else:
                self.layers = nn.Sequential(
                    HNNLayer(self.manifold, self.in_features, self.hidden_dim, self.c, p_drop, lambda x: x, True),
                    HNNLayer(self.manifold, self.hidden_dim, self.hidden_dim, self.c, p_drop, lambda x: x, True)
                )
            
            if self.model_type == 'hnn++':
                self.classifier = HNNPlusPlusMLR(self.manifold, self.hidden_dim, n_classes)
            else:
                self.classifier = HyperbolicMLR(self.manifold, self.hidden_dim, n_classes)

        elif self.model_type == 'pvnn':
            k = _as_negative_curvature(c)
            self.manifold = PVManifold(k=k)
            setattr(self.manifold, 'projx', self.manifold.proj)
            
            self.layers = nn.Sequential(
                 CustomHyperbolicLayer(self.in_features, self.hidden_dim, self.manifold, p_drop, inner_act=self.inner_act, outer_act=self.outer_act, linear_type=self.linear_type),
                 CustomHyperbolicLayer(self.hidden_dim, self.hidden_dim, self.manifold, p_drop, inner_act=self.inner_act, outer_act=self.outer_act, linear_type=self.linear_type)
              )
            self.classifier = PVManifoldMLR(_resolve_pv_curvature(self.manifold), self.hidden_dim, n_classes)

        elif self.model_type == 'lnn':
            self.manifold = self._get_manifold(model_type, c)
            print(f"Building a two-layer {model_type} architecture on {self.manifold.name} manifold")
            
            setattr(self.manifold, 'projx', self.manifold.proj)
            
            self.layers = nn.Sequential(
                
                LorentzLayerNorm(self.manifold, self.in_features-1), 
                LorentzLinear(self.manifold, self.in_features-1, self.hidden_dim, True),  
                LorentzActivation(self.manifold, nn.ReLU()),
                LorentzDropout(self.manifold, p_drop),
                LorentzLinear(self.manifold, self.hidden_dim-1, self.hidden_dim, True),
                LorentzActivation(self.manifold, nn.ReLU()),
                LorentzDropout(self.manifold, p_drop)
            )
            self.classifier = LorentzManifoldMLR(self.manifold, self.hidden_dim+1, n_classes)
    
        else:
            
            self.manifold = self._get_manifold(model_type, c)
            print(f"Building a two-layer {model_type} architecture on {self.manifold.name} manifold")
            
            setattr(self.manifold, 'projx', self.manifold.proj)
            
            self.layers = nn.Sequential(
                CustomHyperbolicLayer(
                    self.in_features,
                    self.hidden_dim,
                    self.manifold,
                    p_drop,
                    inner_act=self.inner_act,
                    outer_act=self.outer_act,
                    linear_type=self.linear_type,
                ),
                CustomHyperbolicLayer(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.manifold,
                    p_drop,
                    inner_act=self.inner_act,
                    outer_act=self.outer_act,
                    linear_type=self.linear_type,
                ),
            )
            
            self.classifier = KleinManifoldMLR(self.manifold, self.hidden_dim, n_classes)

    def _get_manifold(self, model_type, c):
        if model_type == 'lnn':
            print("Using Lorentz manifold")
            return LorentzManifold(c=c, in_features=self.in_features)
        elif model_type == 'knn':
            print("Using Klein manifold")
            return KleinManifold(c=c)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x):
        check_nan(x, "model input")
        
        if self.model_type == 'fc':
            h = self.activation(self.layer1(x))
            check_nan(h, "after FC activation")
            return self.classifier(h)
        elif self.model_type in ['hnn', 'hnn++', 'lnn', 'knn']:
            x_tan = self.manifold.proj_tan0(x, c=self.manifold.c)
            check_nan(x_tan, "after proj_tan0")
            
            x_hyp = self.manifold.expmap0(x_tan, c=self.manifold.c)
            check_nan(x_hyp, "after expmap0")
            
            x_hyp = self.manifold.proj(x_hyp, c=self.manifold.c)
            check_nan(x_hyp, "after proj")
            
            h = self.layers[0](x_hyp)       
            check_nan(h, "after first layer")
            
            h = self.layers[1](h)       
            check_nan(h, "after second layer")
            
            h_tangent = self.manifold.logmap0(h, c=self.manifold.c)
            check_nan(h_tangent, "after logmap0")
            
            logits = self.classifier(h_tangent)
            check_nan(logits, "classifier output")
            
            return logits
            
        elif self.model_type == 'pvnn':
            x_tan = self.manifold.proj_tan0(x, c=self.manifold.c)
            check_nan(x_tan, "after proj_tan0")
            x_hyp = self.manifold.expmap0(x_tan, c=self.manifold.c)
            check_nan(x_hyp, "after expmap0")
            x_hyp = self.manifold.proj(x_hyp, c=self.manifold.c)
            check_nan(x_hyp, "after proj")
 
            h = self.layers[0](x_hyp)       
            check_nan(h, "after first layer")

            h = self.layers[1](h)       
            check_nan(h, "after second layer")
             
            logits = self.classifier(h)
            check_nan(logits, "classifier output")
            return logits



class CustomHyperbolicLayer(nn.Module):
    def __init__(self, in_features, out_features, manifold, p_drop=0.0, inner_act = 'none', outer_act = 'tangent', linear_type: str = 'pvfc'):
        super().__init__()
        self.manifold = manifold
        self._linear_type = linear_type
        self.inner_pre_activation = ManifoldDirectReLU(manifold) if inner_act == 'relu' else None
        if linear_type == 'pv':
            self.linear = PVLinear(manifold, in_features, out_features, bias=True, manifold_out=manifold, p_drop=p_drop)
        elif linear_type == 'pv_lfc':
            psi = self.inner_pre_activation if self.inner_pre_activation is not None else nn.Identity()
            self.linear = PVLinearLFC(in_features, out_features, _resolve_pv_curvature(self.manifold), psi, p_drop, learnable_lambda=True)
            self.inner_pre_activation = None
        elif linear_type == 'pvfc':
            self.linear = PVFC(_resolve_pv_curvature(self.manifold), in_features, out_features, use_bias=True, inner_act=(inner_act if isinstance(inner_act, str) else 'none'))
            self.inner_pre_activation = None
        elif linear_type == 'euc_tangent_fc':
            self.linear = CustomHyperbolicLinear(in_features, out_features, manifold, p_drop, activation=nn.Identity(), use_bias=True)
        else:
            self.linear = CustomHyperbolicLinear(in_features, out_features, manifold, p_drop, activation=None, use_bias=True)
        if outer_act ==  'none':
            self.activation = nn.Identity()
        elif outer_act == 'tangent':
            self.activation = CustomHyperbolicActivation(manifold, nn.Tanh())
        elif outer_act == 'direct':
            print(f"Using direct ReLU outside tangent space")
            self.activation = ManifoldDirectReLU(manifold)
        elif outer_act == 'direct_tanh':
            self.activation = ManifoldTanh(manifold)
        else:
            raise ValueError(f"Unknown activation type: {outer_act}")
        
    
    def forward(self, x):
        check_nan(x, f"CustomHyperbolicLayer input")
        if self.inner_pre_activation is not None and self._linear_type != 'pv_lfc':
            x = self.inner_pre_activation(x)
            check_nan(x, f"after built-in pre-activation")
        h = self.linear(x)
        if torch.isnan(h).any():
            with torch.no_grad():
                def _stat(t):
                    t2 = torch.nan_to_num(t)
                    return (float(t2.min().item()), float(t2.max().item()), float(t2.mean().item()))
                print("[CustomHyperbolicLayer] NaN after linear; x(min,max,mean)=", _stat(x),
                      " h(min,max,mean)=", _stat(h), " type=", getattr(self, "_linear_type", "unknown"))
                if getattr(self, "_linear_type", "") == 'pvfc':
                    print("[Hint] Check lib.pv.graph_ops.PVFC.forward for z clamping and diagnostics.")
        check_nan(h, f"after CustomHyperbolicLinear")
        
        h = self.activation(h)
        check_nan(h, f"after activation")
        
        return h
    
class ManifoldDirectReLU(nn.Module):
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold
        
    def forward(self, x):
        result = F.relu(x)
        
        if hasattr(self.manifold, 'c'):
            c = self.manifold.c
        elif hasattr(self.manifold, 'k'):
            c = self.manifold.k
        else:
            c = 1.0
            
        return self.manifold.proj(result, c=c)

class CustomHyperbolicLinear(nn.Module):
    def __init__(self, in_features, out_features, manifold, p_drop=0.0, activation=None, use_bias=True):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = p_drop
        self.activation = activation                                                     
        self.use_bias = use_bias

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias   = nn.Parameter(torch.Tensor(out_features)) if use_bias else None

        if hasattr(self.manifold, 'c'):
            self.c = self.manifold.c
        elif hasattr(self.manifold, 'k'):
            self.c = self.manifold.k
        else:
            self.c = 1.0

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x):
        check_nan(x, "CustomHyperbolicLinear input")

        if self.activation is not None:
            t = self.manifold.logmap0(x, c=self.c)
            check_nan(t, "after logmap0")

            if torch.isnan(self.weight).any() or torch.isinf(self.weight).any():
                with torch.no_grad():
                    self.reset_parameters()
            W = F.dropout(self.weight, self.dropout, training=self.training)
            if torch.isnan(W).any() or torch.isinf(W).any():
                with torch.no_grad():
                    self.reset_parameters()
                    W = F.dropout(self.weight, self.dropout, training=self.training)
                    if torch.isnan(W).any() or torch.isinf(W).any():
                        W = torch.nan_to_num(W, nan=0.0, posinf=1e6, neginf=-1e6).clamp_(-1e3, 1e3)
            check_nan(W, "after weight dropout")

            t = t @ W.t()
            if self.use_bias:
                t = t + self.bias
            check_nan(t, "after tangent-space linear (+bias)")

            t = self.activation(t)
            check_nan(t, "after tangent-space activation")

            y = self.manifold.expmap0(t, c=self.c)

            y = self.manifold.proj(y, c=self.c)
            return y

        if torch.isnan(self.weight).any() or torch.isinf(self.weight).any():
            with torch.no_grad():
                self.reset_parameters()
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        if torch.isnan(drop_weight).any() or torch.isinf(drop_weight).any():
            with torch.no_grad():
                self.reset_parameters()
                drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
                if torch.isnan(drop_weight).any() or torch.isinf(drop_weight).any():
                    drop_weight = torch.nan_to_num(drop_weight, nan=0.0, posinf=1e6, neginf=-1e6).clamp_(-1e3, 1e3)
        check_nan(drop_weight, "after weight dropout")

        mv = self.manifold.pv_gyro_matvec(drop_weight, x, self.c)
        check_nan(mv, "after mobius_matvec")

        res = self.manifold.proj(mv, self.c)
        check_nan(res, "after result proj")

        if self.use_bias:
            bias_tan = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            check_nan(bias_tan, "bias after proj_tan0")

            hyp_bias = self.manifold.expmap0(bias_tan, self.c)
            check_nan(hyp_bias, "hyp_bias after expmap0")

            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            check_nan(hyp_bias, "hyp_bias after proj")

            res = self.manifold.pv_gyro_add(res, hyp_bias, self.c)
            check_nan(res, "after mobius_add")

            res = self.manifold.proj(res, self.c)
            check_nan(res, "after final proj")

        return res

class CustomHyperbolicActivation(nn.Module):
    def __init__(self, manifold, activation_fn=nn.ReLU()):
        super().__init__()
        self.manifold = manifold
        self.activation = activation_fn
        
        if hasattr(self.manifold, 'c'):
            self.c_in = self.manifold.c
            self.c_out = self.manifold.c
        elif hasattr(self.manifold, 'k'):
            self.c_in = self.manifold.k
            self.c_out = self.manifold.k
        else:
            print(f"Warning: {manifold.__class__.__name__} has no curvature parameter 'c' or 'k'; using default 1.0")
            self.c_in = 1.0
            self.c_out = 1.0

    def forward(self, x):
        x = self.manifold.proj(x, c=self.c_in)
        logmap_result = self.manifold.logmap0(x, c=self.c_in)
        check_nan(logmap_result, "logmap0 result")
        
        activated = self.activation(logmap_result)
        check_nan(activated, "after activation function")
        
        xt = self.manifold.proj_tan0(activated, c=self.c_out)
        check_nan(xt, "after proj_tan0")
        
        expmap_result = self.manifold.expmap0(xt, c=self.c_out)
        check_nan(expmap_result, "expmap0 result")
        
        final_result = self.manifold.proj(expmap_result, c=self.c_out)
        check_nan(final_result, "after final proj")
        
        return final_result
    
class ManifoldTanh(nn.Module):
    def __init__(self, manifold):
        super().__init__()
        self.M = manifold
        self.eps = 1e-9

    def forward(self, h):
        u = self.M.logmap0(h)                            
        r = torch.norm(u, dim=-1, keepdim=True)                 
        s = torch.tanh(r) / (r + self.eps)               
        u_hat = u * s                                     
        return self.M.expmap0(u_hat)   


def get_curvature_for_model(model_type):
    if model_type == 'pvnn':
        return 1.2
    elif model_type == 'hnn':
        return 1.0
    elif model_type == 'hnn++':
        return 1.0
    return 1.0       
def build_model(model_type, dim=16, hidden_dim=None, n_classes=2, p_drop=0.5, c=1.0, weight_decay=0, inner_act = 'none', outer_act = 'tangent', linear_type: str = 'pvfc'):
    print(f"Building model: {model_type}")
    if hidden_dim is None:
        hidden_dim = dim
        
    return GeometricModel(model_type=model_type, dim=dim, hidden_dim=hidden_dim, 
                         n_classes=n_classes, c=c, p_drop=p_drop, 
                         inner_act=inner_act, outer_act=outer_act, linear_type=linear_type)
