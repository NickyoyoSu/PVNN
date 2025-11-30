import torch
import torch.nn as nn

from lib.geoopt.manifolds.stereographic.math import arsinh, artanh

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
