import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from lib.pv.manifold import PVManifold, PVManifoldMLR, PVFC
from lib.pv.layers import PVLinear, PVLinearLFC
from lib.pv.gyrobn import GyroBNPV
from lib.lorentz.manifold import LorentzManifold, LorentzManifoldMLR
from lib.lorentz.layers import LorentzLinear, LorentzActivation, LorentzDropout, LorentzLayerNorm
from lib.klein.manifold import KleinManifold, KleinManifoldMLR
from lib.utils.math_ops import artanh, tanh
from lib.poincare.manifold import PoincareBall
from lib.poincare.layers import HNNLayer, HyperbolicMLR, HNNPlusPlusMLR, HNNPlusPlusLayer

PVManifoldGyro = PVManifold

# ========================= Helper utilities =========================
ACT_MAP = {
    "identity": nn.Identity,
    "tanh"    : nn.Tanh,
    "softplus": lambda : nn.Softplus(beta=1, threshold=20.)
}
def check_nan(tensor, location_name):
    """Utility to guard against NaNs in intermediate tensors."""
    if torch.isnan(tensor).any():
        raise RuntimeError(f"NaNs detected at {location_name} (shape={tensor.shape})")

# ========================= Unified architecture wrapper =========================
class GeometricModel(nn.Module):
    """Unified geometric model wrapper used across Section 6.2 experiments."""

    def __init__(
        self,
        model_type,
        dim=16,
        hidden_dim=16,
        n_classes=2,
        c=1.0,
        p_drop: float = 0.0,
        final_act: str = "softplus",
        inner_act="none",
        outer_act="tangent",
    ):
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

        if self.model_type == "fc":
            # Plain Euclidean baseline
            self.manifold = None
            self.layer1 = nn.Linear(dim, dim)
            self.activation = nn.ReLU()
            self.classifier = nn.Linear(dim, n_classes)
        elif self.model_type in ["hnn", "hnn++"]:
# Poincare HNN / HNN++
            self.manifold = PoincareBall()

            self.c = c
            if self.model_type == "hnn++":
                print("Initializing HNN++ with learnable curvature")
            else:
                print("Initializing vanilla HNN")

            self.manifold.c = self.c

            if self.model_type == "hnn++":
                self.layers = nn.Sequential(
                    HNNPlusPlusLayer(self.manifold, self.in_features, self.hidden_dim, self.c, p_drop, lambda x: x),
                    HNNPlusPlusLayer(self.manifold, self.hidden_dim, self.hidden_dim, self.c, p_drop, lambda x: x),
                )
            else:
                self.layers = nn.Sequential(
                    HNNLayer(self.manifold, self.in_features, self.hidden_dim, self.c, p_drop, lambda x: x, True),
                    HNNLayer(self.manifold, self.hidden_dim, self.hidden_dim, self.c, p_drop, lambda x: x, True),
                )
            
            if self.model_type == "hnn++":
                self.classifier = HNNPlusPlusMLR(self.manifold, self.hidden_dim, n_classes)
            else:
                self.classifier = HyperbolicMLR(self.manifold, self.hidden_dim, n_classes)

        elif self.model_type == "pvnn":
            # PVNN: two PVFC blocks plus optional GyroBN
            self.manifold = PVManifold(c=c)
            self.tan_bn = nn.BatchNorm1d(self.in_features, affine=False, eps=1e-3)
            setattr(self.manifold, "projx", self.manifold.proj)
            self.gyro_manifold = PVManifoldGyro(c=c)
            
            self.layers = nn.Sequential(
                CustomHyperbolicLayer(
                    self.in_features,
                    self.hidden_dim,
                    self.manifold,
                    p_drop,
                    inner_act=self.inner_act,
                    outer_act=self.outer_act,
                    linear_type="pvfc",
                ),
                CustomHyperbolicLayer(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.manifold,
                    p_drop,
                    inner_act=self.inner_act,
                    outer_act=self.outer_act,
                    linear_type="pvfc",
                ),
            )
            self.use_mid_bn = False
            self.use_mid_log_euc_bn = False
            self.no_proj_exp = False

            self.mid_tan_bn = nn.BatchNorm1d(self.hidden_dim, affine=False, eps=1e-3)
            self.bn_mid = GyroBNPV(
                manifold=self.gyro_manifold,
                shape=[self.hidden_dim],
                track_running_stats=True,
                momentum=0.1,
                use_euclid_stats=False,
                use_gyro_midpoint=True,
                clamp_factor=-1.0,
                print_stats=False,
                use_post_gain=False,
                var_floor=1e-2,
                max_tan_norm=20.0,
                scalar_sinh_clip=30.0,
            )
            
            self.classifier = PVManifoldMLR(getattr(self.manifold, "c", 1.0), self.hidden_dim, n_classes)

        elif self.model_type == "lnn":
            self.manifold = self._get_manifold(model_type, c)
            print(f"Building {model_type} with {self.manifold.name} manifold")
            setattr(self.manifold, "projx", self.manifold.proj)

            self.layers = nn.Sequential(
                
                LorentzLayerNorm(self.manifold, self.in_features - 1),
                LorentzLinear(self.manifold, self.in_features - 1, self.hidden_dim, True),
                LorentzActivation(self.manifold, nn.ReLU()),
                LorentzDropout(self.manifold, p_drop),
                LorentzLinear(self.manifold, self.hidden_dim - 1, self.hidden_dim, True),
                LorentzActivation(self.manifold, nn.ReLU()),
                LorentzDropout(self.manifold, p_drop)
            )
            self.classifier = LorentzManifoldMLR(self.manifold, self.hidden_dim+1, n_classes)
    
        else:

            self.manifold = self._get_manifold(model_type, c)
            print(f"Building {model_type} with {self.manifold.name} manifold")

            setattr(self.manifold, "projx", self.manifold.proj)

            self.layers = nn.Sequential(
                CustomHyperbolicLayer(
                    self.in_features,
                    self.hidden_dim,
                    self.manifold,
                    p_drop,
                    inner_act=self.inner_act,
                    outer_act=self.outer_act,
                ),
                CustomHyperbolicLayer(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.manifold,
                    p_drop,
                    inner_act=self.inner_act,
                    outer_act=self.outer_act,
                ),
            )
            
            self.classifier = KleinManifoldMLR(self.manifold, self.hidden_dim, n_classes)
            #self.classifier = nn.Linear(self.hidden_dim, n_classes)

    def _get_manifold(self, model_type, c):
        if model_type == "lnn":
            print("Using Lorentz manifold")
            return LorentzManifold(c=c, in_features=self.in_features)
        elif model_type == "knn":
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
            check_nan(x_hyp, "after projection")
            
            h = self.layers[0](x_hyp)  # first block
            check_nan(h, "after first layer")

            h = self.layers[1](h)  # second block
            check_nan(h, "after second layer")

            h_tangent = self.manifold.logmap0(h, c=self.manifold.c)
            check_nan(h_tangent, "after logmap0")
            
            logits = self.classifier(h_tangent)
            check_nan(logits, "classifier output")
            
            return logits
            
        elif self.model_type == 'pvnn':
            x_tan = self.manifold.proj_tan0(x, c=self.manifold.c)
            check_nan(x_tan, "after proj_tan0")
            if getattr(self, 'no_proj_exp', False):
                x_hyp = x_tan
            else:
                x_hyp = self.manifold.expmap0(x_tan, c=self.manifold.c)
                check_nan(x_hyp, "after expmap0")
                x_hyp = self.manifold.proj(x_hyp, c=self.manifold.c)
                check_nan(x_hyp, "after projection")
 
            h = self.layers[0](x_hyp)  # first block
            check_nan(h, "after first layer")
            if getattr(self, 'use_mid_bn', False) and not getattr(self, 'no_proj_exp', False):
                h = self.bn_mid(h)
                check_nan(h, "after GyroBN")
            elif getattr(self, 'use_mid_log_euc_bn', False) and not getattr(self, 'no_proj_exp', False):
                t_mid = self.manifold.logmap0(h, c=self.manifold.c)
                check_nan(t_mid, "intermediate logmap0")
                t_mid = self.mid_tan_bn(t_mid)
                check_nan(t_mid, "after tangent BN")
                h = self.manifold.expmap0(t_mid, c=self.manifold.c)
                check_nan(h, "intermediate expmap0")
                h = self.manifold.proj(h, c=self.manifold.c)
                check_nan(h, "intermediate projection")

            h = self.layers[1](h)  # second block
            check_nan(h, "after second layer")
            logits = self.classifier(h)
            check_nan(logits, "classifier output")
            return logits

# ========================= Custom building blocks (PV, Lorentz, Klein) =========================


class CustomHyperbolicLayer(nn.Module):
    """Generic hyperbolic layer that wraps multiple linear/activation choices."""

    def __init__(
        self,
        in_features,
        out_features,
        manifold,
        p_drop=0.0,
        inner_act="none",
        outer_act="tangent",
        linear_type: str = "pvfc",
    ):
        super().__init__()
        self.manifold = manifold
        self._linear_type = linear_type
        self.inner_pre_activation = ManifoldDirectReLU(manifold) if inner_act == "relu" else None
        if linear_type == "pv":
            self.linear = PVLinear(manifold, in_features, out_features, bias=True, manifold_out=manifold, p_drop=p_drop)
        elif linear_type == "pv_lfc":
            psi = self.inner_pre_activation if self.inner_pre_activation is not None else nn.Identity()
            self.linear = PVLinearLFC(
                in_features,
                out_features,
                getattr(self.manifold, "c", 1.0),
                psi,
                p_drop,
                learnable_lambda=True,
            )
            self.inner_pre_activation = None
        elif linear_type == "pvfc":
            self.linear = PVFC(
                getattr(self.manifold, "c", 1.0),
                in_features,
                out_features,
                use_bias=True,
                inner_act=(inner_act if isinstance(inner_act, str) else "none"),
            )
            self.inner_pre_activation = None
        elif linear_type == "euc_tangent_fc":
            self.linear = CustomHyperbolicLinear(in_features, out_features, manifold, p_drop, activation=nn.Identity(), use_bias=True)
        else:
            self.linear = CustomHyperbolicLinear(in_features, out_features, manifold, p_drop, activation=None, use_bias=True)
        if outer_act == "none":
            self.activation = nn.Identity()
        elif outer_act == "tangent":
            self.activation = CustomHyperbolicActivation(manifold, nn.Tanh())
        elif outer_act == "direct":
            print("Applying manifold ReLU directly")
            self.activation = ManifoldDirectReLU(manifold)
        else:
            raise ValueError(f"Unknown activation type: {outer_act}")
        
    
    def forward(self, x):
        check_nan(x, f"CustomHyperbolicLayer input")
        if self.inner_pre_activation is not None and self._linear_type != "pv_lfc":
            x = self.inner_pre_activation(x)
            check_nan(x, "after inner activation")
        h = self.linear(x)
        if torch.isnan(h).any():
            with torch.no_grad():
                def _stat(t):
                    t2 = torch.nan_to_num(t)
                    return (float(t2.min().item()), float(t2.max().item()), float(t2.mean().item()))
                print("[CustomHyperbolicLayer] NaN after linear; x(min,max,mean)=", _stat(x),
                      " h(min,max,mean)=", _stat(h), " type=", getattr(self, "_linear_type", "unknown"))
                if getattr(self, "_linear_type", "") == "pvfc":
                    print("[Hint] Inspect PVFC forward path for saturation.")
        check_nan(h, "after CustomHyperbolicLinear")
        
        h = self.activation(h)
        check_nan(h, "after activation")
        
        return h
    
class ManifoldDirectReLU(nn.Module):
    """Applies ReLU directly in the ambient space and re-projects to the manifold."""

    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold

    def forward(self, x):
        result = F.relu(x)
        if hasattr(self.manifold, "c"):
            c = self.manifold.c
        elif hasattr(self.manifold, "k"):
            c = self.manifold.k
        else:
            c = 1.0
        return self.manifold.proj(result, c=c)

class CustomHyperbolicLinear(nn.Module):
    """Hyperbolic linear layer with optional tangent-space activation."""
    def __init__(self, in_features, out_features, manifold, p_drop=0.0, activation=None, use_bias=True):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = p_drop
        self.activation = activation     # e.g., nn.ReLU(), nn.GELU(), nn.Tanh(), or None
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
            check_nan(t, "after tangent linear (+bias)")

            t = self.activation(t)
            check_nan(t, "after tangent activation")

            y = self.manifold.expmap0(t, c=self.c)
            #check_nan(y, "after expmap0")

            y = self.manifold.proj(y, c=self.c)
            #check_nan(y, "after projection")
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

        #mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        mv = self.manifold.pv_gyro_matvec(drop_weight, x, self.c)
        check_nan(mv, "after mobius_matvec")

        res = self.manifold.proj(mv, self.c)
        check_nan(res, "after projection")

        if self.use_bias:
            bias_tan = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            check_nan(bias_tan, "bias after proj_tan0")

            hyp_bias = self.manifold.expmap0(bias_tan, self.c)
            check_nan(hyp_bias, "hyp_bias after expmap0")

            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            check_nan(hyp_bias, "hyp_bias after projection")

            #res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.pv_gyro_add(res, hyp_bias, self.c)
            check_nan(res, "after mobius_add")

            res = self.manifold.proj(res, self.c)
            check_nan(res, "after final projection")

        return res
'''
    def forward(self, x):
        u      = self.manifold.logmap0(x)
        u      = self.dropout(u)
        h_tan  = u @ self.weight.T + self.bias
        h      = self.manifold.expmap0(h_tan)
        return self.manifold.projx(h)
'''
'''def forward(self, x):
    x_tan = self.manifold.logmap0(x)

        W_masked = self.dropout(self.weight)   # DropConnect mask
    h_tan    = x_tan @ W_masked.T          # matvec
    # ----------------------

        h_tan = h_tan + self.bias              # bias is added in tangent space
    h      = self.manifold.expmap0(h_tan)
    return self.manifold.proj(h)'''

class CustomHyperbolicActivation(nn.Module):
    """Hyperbolic activation identical to HypAct."""
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
            print(f"Warning: {manifold.__class__.__name__} has no curvature attribute; falling back to 1.0")
            self.c_in = 1.0
            self.c_out = 1.0

    def forward(self, x):
        x = self.manifold.proj(x, c=self.c_in)
        logmap_result = self.manifold.logmap0(x, c=self.c_in)
        check_nan(logmap_result, "logmap0 result")
        
        activated = self.activation(logmap_result)
        check_nan(activated, "after activation")
        
        xt = self.manifold.proj_tan0(activated, c=self.c_out)
        check_nan(xt, "after proj_tan0")
        
        expmap_result = self.manifold.expmap0(xt, c=self.c_out)
        check_nan(expmap_result, "expmap0 result")
        
        final_result = self.manifold.proj(expmap_result, c=self.c_out)
        check_nan(final_result, "after final projection")
        
        return final_result
    
class ManifoldTanh(nn.Module):
    """Radial tanh on the manifold to keep norms bounded."""
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


def build_model(model_type, dim=16, hidden_dim=None, n_classes=2, p_drop=0.5, c=1.0, weight_decay=0, inner_act='none', outer_act='tangent'):
    """Factory that instantiates the requested geometric model."""
    print(f"Building model: {model_type}")
    if hidden_dim is None:
        hidden_dim = dim

    return GeometricModel(
        model_type=model_type,
        dim=dim,
        hidden_dim=hidden_dim,
        n_classes=n_classes,
        c=c,
        p_drop=p_drop,
        inner_act=inner_act,
        outer_act=outer_act,
    )

def build_pvnn_frechet_sweep(
    dim=16,
    hidden_dim=None,
    n_classes=2,
    p_drop=0.5,
    c=1.0,
    iters_list=None,
    inner_act='none',
    outer_act='tangent',
    bn_mode: str = 'gyro',
):
    """Build a family of PVNN models with different Frechet iterations for GyroBN."""
    if hidden_dim is None:
        hidden_dim = dim
    if iters_list is None:
        iters_list = [1, 2, 5, 10, 'inf']

    models = {}
    for it in iters_list:
        max_iter = -1 if (isinstance(it, str) and it.lower() in ['inf', 'infinite']) else int(it)
        m = GeometricModel(
            model_type='pvnn', dim=dim, hidden_dim=hidden_dim, n_classes=n_classes,
            c=c, p_drop=p_drop, final_act='softplus', inner_act=inner_act, outer_act=outer_act
        )
        if hasattr(m, 'bn_mid'):
            if bn_mode == 'gyro':
                if hasattr(m, 'use_mid_bn'):
                    m.use_mid_bn = True
                if hasattr(m, 'use_mid_log_euc_bn'):
                    m.use_mid_log_euc_bn = False
                m.bn_mid.use_gyro_midpoint = False
                m.bn_mid.use_euclid_stats = False
                m.bn_mid.max_iter = max_iter
            elif bn_mode == 'log_euc':
                if hasattr(m, 'use_mid_bn'):
                    m.use_mid_bn = False
                if hasattr(m, 'use_mid_log_euc_bn'):
                    m.use_mid_log_euc_bn = True
            elif bn_mode == 'none':
                if hasattr(m, 'use_mid_bn'):
                    m.use_mid_bn = False
                if hasattr(m, 'use_mid_log_euc_bn'):
                    m.use_mid_log_euc_bn = False
        models[max_iter] = m
    return models

