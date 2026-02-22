import torch
import torch.nn as nn

from .manifold import PVManifold


class PVManifoldMLR(torch.nn.Module):
    """PV流形上的多类逻辑回归分类器，使用点到超平面距离"""
    def __init__(self, manifold, in_features, num_classes):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.num_classes = num_classes
        # w_s 是超平面法向量
        self.weight = torch.nn.Parameter(torch.Tensor(num_classes, in_features))
        # sigma 是超平面偏移量
        self.bias = torch.nn.Parameter(torch.Tensor(num_classes))
        self.c = manifold.c  # 曲率参数
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化权重和偏置，确保满足κ²‖w_s‖² > σ²的条件"""
        torch.nn.init.xavier_uniform_(self.weight, gain=0.1)
        torch.nn.init.zeros_(self.bias)
        # 可以在这里添加额外的限制以确保κ²‖w_s‖² > σ²
    
    def forward(self, x):
        batch_size = x.size(0)
        logits = torch.zeros(batch_size, self.num_classes,
                            device=x.device, dtype=x.dtype)

        for i in range(self.num_classes):
            w_s = self.weight[i]          # (D,)
            sigma = self.bias[i]          # scalar

            # ——改动①：约束幅值但保留符号——
            w_norm_sq = (w_s * w_s).sum()
            max_mag = 0.99 * self.c * torch.sqrt(w_norm_sq + 1e-12)
            sigma_constrained = sigma.sign() * torch.minimum(sigma.abs(), max_mag)

            # 计算距离
            distance = self.manifold.hyperbolic_distance(
                x, w_s.unsqueeze(0).expand(batch_size, -1),
                sigma_constrained.expand(batch_size, 1),
                c=self.c
            )

            # ——改动②：logits 取负距离（靠近=更大）——
            logits[:, i] = (-distance).squeeze(-1)

        return logits

