import torch
import torch.nn as nn
import torch.nn.functional as F


class EuclideanMLR(nn.Module):
    """Simple Euclidean multiclass logistic regression head."""

    def __init__(self, manifold, in_features, num_classes, bias=True):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_classes))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        logits = F.linear(x, self.weight, self.bias)
        logits = torch.clamp(logits, -1e3, 1e3)
        logits = torch.nan_to_num(logits, nan=0.0)
        return logits
