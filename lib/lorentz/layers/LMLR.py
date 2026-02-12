import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from lib.lorentz.manifold import CustomLorentz

class LorentzMLR(nn.Module):
    """ Multinomial logistic regression (MLR) in the Lorentz model
    """
    def __init__(
            self, 
            manifold: CustomLorentz, 
            num_features: int, 
            num_classes: int
        ):
        super(LorentzMLR, self).__init__()

        self.manifold = manifold

        self.a = torch.nn.Parameter(torch.zeros(num_classes,))
        self.z = torch.nn.Parameter(F.pad(torch.zeros(num_classes, num_features-2), pad=(1,0), value=1)) # z should not be (0,0)

        self.init_weights()

    def forward(self, x):
        # Hyperplane
        sqrt_mK = 1/self.manifold.k.sqrt()
        
        norm_z = torch.norm(self.z, dim=-1)
        
        # Clamp self.a first to avoid sinh/cosh overflow (empirically [-20, 20])
        a_clamped = torch.clamp(self.a, -20.0, 20.0)
        
        w_t = (torch.sinh(sqrt_mK * a_clamped) * norm_z)
        w_s = torch.cosh(sqrt_mK * a_clamped.view(-1,1)) * self.z
        
        # Keep previous clamp on w_t/w_s for stability
        w_t = torch.clamp(w_t, -1e5, 1e5)
        w_s = torch.clamp(w_s, -1e4, 1e4)
        
        # Ensure beta_input is non-negative before sqrt
        beta_input = -w_t**2 + torch.norm(w_s, dim=-1)**2
        beta_input = torch.clamp(beta_input, min=0.0)
        beta = torch.sqrt(beta_input + 1e-15)
        
        alpha = -w_t * x.narrow(-1, 0, 1) + (torch.cosh(sqrt_mK * a_clamped) * torch.inner(x.narrow(-1, 1, x.shape[-1]-1), self.z))
        
        # Clamp asinh_input to avoid infinities
        asinh_input = sqrt_mK * alpha / beta
        asinh_input = torch.clamp(asinh_input, -1e6, 1e6)
        
        d = self.manifold.k.sqrt() * torch.abs(torch.asinh(asinh_input))  # Distance to hyperplane
        
        logits = torch.sign(alpha) * beta * d

        return logits
        
    def init_weights(self):
        stdv = 1. / math.sqrt(self.z.size(1))
        nn.init.uniform_(self.z, -stdv, stdv)
        nn.init.uniform_(self.a, -stdv, stdv)