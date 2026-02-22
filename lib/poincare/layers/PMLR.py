import torch
import torch.nn as nn

from lib.geoopt.manifolds.stereographic.math import arsinh, artanh

class UnidirectionalPoincareMLR(nn.Module):
    """MLR in the Poincare model by Shimizu et al. (2020)."""

    __constants__ = ["feat_dim", "num_outcome"]

    def __init__(self, feat_dim, num_outcome, bias=True, ball=None):
        super().__init__()
        self.ball = ball
        self.feat_dim = feat_dim
        self.num_outcome = num_outcome
        weight = torch.empty(feat_dim, num_outcome).normal_(
            mean=0, std=(self.feat_dim) ** -0.5 / self.ball.c.data.sqrt()
        )
        self.weight_g = nn.Parameter(weight.norm(dim=0))
        self.weight_v = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.empty(num_outcome), requires_grad=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return unidirectional_poincare_mlr(
            x,
            self.weight_g,
            self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15),
            self.bias,
            self.ball.c,
        )

    def extra_repr(self):
        return "feat_dim={}, num_outcome={}, bias={}".format(
            self.feat_dim, self.num_outcome, self.bias.requires_grad
        )


@torch.jit.script
def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    rc = c.sqrt()
    drcr = 2.0 * rc * r
    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)

    return 2 * z_norm / rc * arsinh(
        (2.0 * torch.matmul(rcx, z_unit) * drcr.cosh() - (1.0 + cx2) * drcr.sinh())
        / torch.clamp_min(1.0 - cx2, 1e-15)
    )


class BusemannPoincareMLR(nn.Module):
    """
    Hyperbolic multiclass logistic regression layer based on Busemann functions.
    """

    def __init__(self, feat_dim, num_outcome, c, ball):
        super(BusemannPoincareMLR, self).__init__()
        self.ball = ball
        #self.ball = PoincareBall(c=c, learnable=learn_curvature)
        self.feat_dim = feat_dim
        self.num_outcome = num_outcome    
        self.c = c
        
        INIT_EPS = 1e-3
        self.point = nn.Parameter(torch.randn(self.num_outcome, self.feat_dim) * INIT_EPS)
        self.tangent = nn.Parameter(torch.randn(self.num_outcome, self.feat_dim) * INIT_EPS)

    def forward(self, input):                    
        bs = input.shape[0]
        d = input.shape[1]
        input = torch.reshape(input, (-1, self.feat_dim))
        point = self.ball.expmap0(self.point)    
        distances = torch.zeros_like(torch.empty(input.shape[0], self.num_outcome), device="cuda:0", requires_grad=False)
        for i in range(self.num_outcome):
            point_i = point[i]
            tangent_i = self.tangent[i] 
            
            distances[:, i] = self.ball.dist2planebm(      
                x=input, a=tangent_i, p=point_i, c=self.c
            )
        return distances 
