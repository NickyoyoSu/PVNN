

import torch
import torch.nn as nn
import math
from lib.geoopt.manifolds.stereographic.math import dist2plane

class GaneaPoincareMLR(nn.Module):
    def __init__(self, manifold, embed_dim, num_classes):
        super().__init__()
        self.manifold = manifold
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        # Initialization
        bound = 1 / math.sqrt(embed_dim)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        batch_size = x.size(0)
        # Prepare p = 0 for every class
        p = torch.zeros(batch_size, self.num_classes, self.embed_dim, device=x.device, dtype=x.dtype)
        # Expand a and x
        a = self.weight.unsqueeze(0).expand(batch_size, -1, -1)
        x_exp = x.unsqueeze(1).expand(batch_size, self.num_classes, -1)
        # Compute signed distance
        dist = dist2plane(x_exp, p, a, k=self.manifold.k, signed=True, dim=-1)
        # logits = -dist + bias
        logits = -dist + self.bias.unsqueeze(0).expand(batch_size, -1)
        return logits



