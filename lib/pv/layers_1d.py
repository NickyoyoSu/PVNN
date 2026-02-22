import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as T

from .PV_monifold import PVManifold, PVFC
from .gyrobn_pv import PVGyroBN1d
from . import debug as pv_debug


class PVConv2d(nn.Module):
    """
    一个真正的双曲 2D 卷积层，适用于 PV 流形。

    它遵循 "Transform-then-Aggregate" 范式：
    1. 为 k*k 核中的每个位置 (i,j) 创建一个独立的 PVFC(in, out) 层。
    2. 对输入 x 的每个 patch，将其 (i,j) 位置的 C_in 维向量
       送入 PVFC_{i,j} 层，得到 C_out 维的双曲向量。
    3. 将 k*k 个 C_out 维的双曲向量 Log0 到切空间。
    4. 在切空间中对它们求和（聚合）。
    5. 在切空间中应用激活函数 (phi) 和偏置 (b)。
    6. 将结果 Exp0 映射回 PV 流形。
    """
    def __init__(self, 
                 c: float, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int = 1, 
                 padding: int = 0, 
                 bias: bool = True, 
                 activation: T.Callable = F.relu):
        """
        初始化 PV 卷积层。

        参数:
            c (float): 流形曲率 (c > 0)。
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数 (即 "滤波器" 数量)。
            kernel_size (int): 卷积核大小 (k x k)。
            stride (int): 步幅。
            padding (int): 填充。
            bias (bool): 是否在切空间聚合后添加欧几里得偏置。
            activation (Callable): 在切空间中应用的激活函数 (例如 F.relu)。
        """
        super().__init__()
        self.manifold = PVManifold(c)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 统一处理 kernel_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
            
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        
        # 这是设计的核心：
        # 创建 k*k 个独立的 PVFC 层，每个层都执行 C_in -> C_out 的变换。
        self.k_h, self.k_w = self.kernel_size
        self.num_kernels = self.k_h * self.k_w
        
        self.kernels = nn.ModuleList()
        for _ in range(self.num_kernels):
            # 在 PVFC 内部不使用偏置，因为我们在聚合 *之后* 添加一个共享偏置
            self.kernels.append(PVFC(c, in_channels, out_channels, use_bias=False))

        # 共享的欧几里得偏置，在切空间中聚合后添加
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        x 形状: [B, C_in, H, W]
        """
        B, C_in, H, W = x.shape
        if C_in != self.in_channels:
            raise ValueError(f"输入通道 ({C_in}) 与声明的 in_channels ({self.in_channels}) 不匹配")

        # --- 1. 提取 Patches ---
        # 使用 F.unfold 高效提取所有感受野
        patches = F.unfold(x, 
                           kernel_size=self.kernel_size, 
                           padding=self.padding, 
                           stride=self.stride)
        # patches 形状: [B, C_in * k*k, L]
        # L 是输出像素数 (H' * W')
        
        # --- 2. 重塑 Patches 以便应用核 ---
        # [B, C_in * k*k, L] -> [B, C_in, k*k, L]
        patches = patches.view(B, C_in, self.num_kernels, -1)
        
        # -> [B, L, k*k, C_in] (将 L 移到批次维度附近)
        patches = patches.permute(0, 3, 2, 1).contiguous()
        
        # -> [B*L, k*k, C_in] (展平 B 和 L，以便批量处理所有像素)
        # 我们现在有 B*L 个 "patch"，每个 patch 包含 k*k 个 C_in 维向量
        L = patches.shape[1] # 获取真实的 L (H_out * W_out)
        patches = patches.view(B * L, self.num_kernels, C_in)

        # --- 3. 应用 PVFC 核、Log0 并聚合 ---
        v_patches_list = []
        for i in range(self.num_kernels):
            # 获取所有 patch 中第 i 个位置的 C_in 维向量
            # x_i 形状: [B*L, C_in]
            x_i = patches[:, i, :] 
            
            # 步骤 1: 特征变换 (在 PV 流形上)
            # h_i 形状: [B*L, C_out]
            h_i = self.kernels[i](x_i)
            
            # 步骤 2: 映射到切空间 (准备聚合)
            v_i = self.manifold.log0(h_i)
            v_patches_list.append(v_i)
        
        # 步骤 3: 聚合 (在切空间中)
        # v_stack 形状: [k*k, B*L, C_out]
        v_stack = torch.stack(v_patches_list, dim=0)
        # v_agg 形状: [B*L, C_out]
        v_agg = v_stack.sum(dim=0)
        
        # --- 4. 应用偏置和激活 (在切空间中) ---
        if self.bias is not None:
            v_agg = v_agg + self.bias  # 广播 [B*L, C_out] + [C_out]
        
        if self.activation is not None:
            v_agg = self.activation(v_agg)
            
        # --- 5. 映射回 PV 流形 ---
        # x_out_flat 形状: [B*L, C_out]
        x_out_flat = self.manifold.exp0(v_agg)
        
        # --- 6. 恢复图像形状 ---
        # 计算 H_out, W_out
        H_out = (H + 2 * self.padding - self.k_h) // self.stride + 1
        W_out = (W + 2 * self.padding - self.k_w) // self.stride + 1
        
        # [B*L, C_out] -> [B, L, C_out]
        x_out = x_out_flat.view(B, L, self.out_channels)
        
        # [B, L, C_out] -> [B, H_out, W_out, C_out]
        x_out = x_out.view(B, H_out, W_out, self.out_channels)
        
        # [B, H_out, W_out, C_out] -> [B, C_out, H_out, W_out] (恢复 Pytorch 的 NCHW 格式)
        x_out = x_out.permute(0, 3, 1, 2).contiguous()
        
        return x_out



class PVConv1d(nn.Module):
    """
    "先聚合-后变换" (Aggregate-then-Transform) 风格的 PV 卷积层。
    
    这实现了导师的 PVFC(Concat(x_i)) 指令。
    它不使用 Log0/Exp0 封装，而是利用 PV 流形就是 R^n 的特性。

    1.  提取 k*k 感受野中的 C_in 维向量 x_i。
    2.  在欧几里得空间中将所有 k 个 x_i 拼接 (Concat) 起来，
        形成一个 k*C_in 维的超长向量 x_concat。
    3.  将 x_concat 视为一个 PV 空间 (R^{k*C_in}) 上的点。
    4.  使用一个 PVFC 层（Thm 5.3） 将 x_concat 
        从 PV^{k*C_in} 变换到 PV^{C_out}。
    """
    def __init__(self, 
                 c: float, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int = 1, 
                 padding: int = 0, 
                 bias: bool = True,
                 inner_act: str = 'none'):
        
        super().__init__()
        self.manifold = PVManifold(c)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.num_kernel_positions = self.kernel_size
        
        # 这是设计的核心：
        # 一个单独的 PVFC 层 。
        # 它的输入维度是 k * C_in (拼接后的向量)
        # 它的输出维度是 C_out
        self.pvfc_transform = PVFC(
            c=c,
            in_features=self.num_kernel_positions * self.in_channels,
            out_features=self.out_channels,
            use_bias=bias,
            inner_act=inner_act
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        x 形状: [B, C_in, L_in] (Pytorch 1D 卷积的标准格式)
        """
        B, C_in, L_in = x.shape
        if C_in != self.in_channels:
            raise ValueError(f"输入通道 ({C_in}) 与声明的 in_channels ({self.in_channels}) 不匹配")

        # --- 1. 提取 Patches (欧式) ---
        # F.unfold 需要 4D 输入，我们将 1D 视为 H=1 的 2D
        x_4d = x.unsqueeze(2)  # [B, C_in, 1, L_in]
        
        patches = F.unfold(x_4d, 
                           kernel_size=(1, self.kernel_size), 
                           padding=(0, self.padding), 
                           stride=(1, self.stride))
        # patches 形状: [B, C_in * k, L_out]
        
        L_out = patches.shape[2] # 获取真实的 L_out

        # --- 2. 欧式聚合 (Concat) ---
        # [B, C_in * k, L_out] -> [B, L_out, C_in * k]
        patches_permuted = patches.permute(0, 2, 1).contiguous()
        
        # -> [B*L_out, C_in * k] (展平 B 和 L_out)
        # 这就是 x_concat，一个在 R^{k*C_in} 上的大向量
        x_concat = patches_permuted.view(B * L_out, -1)
        
        # --- 3. 应用 PVFC 变换 (在流形上) ---
        # x_concat 被视为 PV^{k*C_in} 上的一个点
        # x_out_flat 形状: [B*L_out, C_out]
        x_out_flat = self.pvfc_transform(x_concat)
            
        # --- 4. 恢复序列形状 ---
        # [B*L_out, C_out] -> [B, L_out, C_out]
        x_out = x_out_flat.view(B, L_out, self.out_channels)
        
        # [B, L_out, C_out] -> [B, C_out, L_out] (恢复 Pytorch 的 NCL 格式)
        x_out = x_out.permute(0, 2, 1).contiguous()
        
        return x_out


class PVReLU(nn.Module):
    """
    在 PV 流形上的 ReLU：logmap0 → ReLU → expmap0
    """
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.manifold.logmap0(x, c=getattr(self.manifold, 'c', None))
        v = F.relu(v)
        y = self.manifold.expmap0(v, c=getattr(self.manifold, 'c', None))
        return y


class PVAct1d(nn.Module):
    """
    在 PV 流形上的可选激活：logmap0 → act → expmap0

    支持:
    - relu
    - tanh
    - softplus
    - none / identity
    """
    def __init__(self, manifold, act: str = "relu"):
        super().__init__()
        self.manifold = manifold
        self.act = (act or "none").lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pv_debug.report_if_nonfinite("PVAct1d.input", x)
        if self.act in ("none", "identity", "linear"):
            return x
        v = self.manifold.logmap0(x, c=getattr(self.manifold, 'c', None))
        pv_debug.report_if_nonfinite("PVAct1d.logmap0", v)
        if self.act == "relu":
            v = F.relu(v)
        elif self.act == "tanh":
            v = torch.tanh(v)
        elif self.act == "softplus":
            v = F.softplus(v, beta=1.0, threshold=20.0)
        else:
            raise ValueError(f"Unsupported PV activation: {self.act}")
        y = self.manifold.expmap0(v, c=getattr(self.manifold, 'c', None))
        pv_debug.report_if_nonfinite("PVAct1d.output", y)
        return y


class PVFullyConnected(nn.Module):
    """
    PV 全连接（保持在流形上）：使用 PVFC 的封装
    """
    def __init__(self, manifold, in_features: int, out_features: int, use_bias: bool = True, inner_act: str = 'none'):
        super().__init__()
        c_val = float(getattr(manifold, 'c', getattr(manifold, 'curvature', 1.0)))
        self.fc = PVFC(c=c_val, in_features=in_features, out_features=out_features, use_bias=use_bias, inner_act=inner_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class PVBatchNorm1d(nn.Module):
    """
    兼容接口的 PV BN：
    - use_gyrobn=True：使用 gyrobn_pv 中的 PVGyroBN1d，但用 PV_monifold.PVManifold(c) 构建内部流形
    - use_gyrobn=False：切空间欧式 BN
    接口兼容 blocks_1d 的扩展参数（忽略未使用项）
    """
    def __init__(self,
                 manifold,
                 num_features: int,
                 use_gyrobn: bool = True,
                 print_stats: bool = False,
                 clamp_factor: float = 3.0,
                 use_euclid_stats: bool = True,
                 **kwargs):
        super().__init__()
        self.manifold = manifold
        self.use_gyrobn = use_gyrobn
        if use_gyrobn:
            c_val = float(getattr(manifold, 'c', getattr(manifold, 'curvature', 1.0)))
            inner_manifold = PVManifold(c=c_val)
            self.impl = PVGyroBN1d(inner_manifold, num_features)
        else:
            self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gyrobn:
            return self.impl(x)
        # 切空间 BN
        B, C, L = x.shape
        xt = self.manifold.logmap0(x.permute(0, 2, 1).contiguous().view(-1, C), c=getattr(self.manifold, 'c', None))
        xt = self.bn(xt.view(B, L, C).permute(0, 2, 1))
        y = self.manifold.expmap0(xt.permute(0, 2, 1).contiguous().view(-1, C), c=getattr(self.manifold, 'c', None)).view(B, L, C).permute(0, 2, 1)
        return y