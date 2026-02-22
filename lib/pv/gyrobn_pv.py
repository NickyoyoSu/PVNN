import torch
import torch.nn as nn
import math
from .PV_monifold import PVManifold


class GyroBNPV(nn.Module):
    """
    修正后的GyroBN PV实现，完全按照GyroBN_new的理论
    
    关键修正：
    1. 使用正确的陀螺逆元 gyroinv
    2. 使用陀螺变换 gyrotrans 而不是 mobius_add
    3. 使用陀螺标量乘法 pv_gyro_scalar_mul
    """
    
    def __init__(self, manifold: PVManifold, shape, batchdim=[0], momentum=0.1,
                 track_running_stats=True, init_1st_batch=False, eps=1e-6, max_iter=1000,
                 clamp_factor: float = -1.0, print_stats: bool = False,
                 diff_stats: bool = False, use_post_gain: bool = True,
                 max_step: float = 0.5, use_plain_logexp: bool = True,
                 use_euclid_stats: bool = False, use_gyro_midpoint: bool = False,
                 var_floor: float = 1e-3, max_tan_norm: float = 50.0,
                 scalar_sinh_clip: float = 30.0):
        super().__init__()
        self.manifold = manifold
        self.shape = shape
        self.batchdim = batchdim
        self.momentum = momentum
        self.eps = eps
        self.track_running_stats = track_running_stats
        self.init_1st_batch = init_1st_batch
        self.max_iter = max_iter
        
        # 设置参数和运行统计量
        self.set_parameters()
        if self.track_running_stats:
            self.register_running_stats()
        
        # 额外调参与调试
        self.clamp_factor = clamp_factor  # -1 表示不裁剪
        self.print_stats = print_stats
        self.diff_stats = diff_stats
        self.use_post_gain = use_post_gain
        self.max_step = max_step
        self.use_plain_logexp = use_plain_logexp
        # 是否使用切空间欧式统计（log-Euclidean 近似：μ_t=mean(log0(x)); μ=exp0(μ_t)）
        self.use_euclid_stats = use_euclid_stats
        # 是否使用测地中点归约作为批均值（两点中点的迭代归约近似）
        self.use_gyro_midpoint = use_gyro_midpoint
        # 数值稳健参数
        self.var_floor = float(var_floor)
        self.max_tan_norm = float(max_tan_norm)
        # 陀螺标量乘法中 sinh 的输入裁剪阈值（放宽可增大动态范围）
        self.scalar_sinh_clip = float(scalar_sinh_clip)
        # 调试状态
        self._dbg_seen = 0
        self._dbg_print_limit = 3
        self._last_mean_iters = 0
        self._last_step_norm = 0.0
        # 归一化后的几何放大因子（可学习），用于恢复足够的特征幅度
        self.post_gain = nn.Parameter(torch.tensor(1.5, dtype=torch.get_default_dtype()))
    
    def _get_param_shapes(self):
        """获取参数形状"""
        shape_prefix = list(self.shape[:-1])
        mean_shape = shape_prefix + [self.shape[-1]]
        # 与 GyroBNBase/GyroBNH 保持一致：方差与 shift 为标量
        var_shape = shape_prefix + [1] if shape_prefix else [1]
        return mean_shape, var_shape
    
    def set_parameters(self):
        """设置可学习参数"""
        mean_shape, var_shape = self._get_param_shapes()
        # 权重参数：在切空间中的向量
        self.weight = nn.Parameter(torch.zeros(*mean_shape))
        # 缩放参数：与 GyroBN_new 一致，使用标量（或按前缀维度广播）
        self.shift = nn.Parameter(torch.ones(*var_shape))
    
    def register_running_stats(self):
        """注册运行统计量"""
        if self.init_1st_batch:
            self.running_mean = None
            self.running_var = None
        else:
            mean_shape, var_shape = self._get_param_shapes()
            # 初始化运行均值为原点
            running_mean = torch.zeros(*mean_shape)
            # 初始化运行方差为1
            running_var = torch.ones(*var_shape)
            self.register_buffer('running_mean', running_mean)
            self.register_buffer('running_var', running_var)
    
    def gyroinv(self, x):
        """
        陀螺逆元（PV）：gyroinv(x) = -x
        """
        return -x
    
    def gyrotrans(self, x, y, translate='Left'):
        """
        陀螺变换：gyrotrans(x, y) = x ⊕ y
        这是GyroBN_new中的正确实现
        """
        if translate == 'Left':
            out = self.manifold.gyro_add(x, y)
        elif translate == 'Right':
            out = self.manifold.gyro_add(y, x)
        else:
            raise ValueError("translate must be 'Left' or 'Right'")
        # 仅诊断，不做数值替换，避免特征被归零
        if torch.isnan(out).any() or torch.isinf(out).any():
            with torch.no_grad():
                print("[GyroBNPV][diag] NaN/Inf after gyrotrans; x/y absmax=",
                      float(x.abs().max().item()), float(y.abs().max().item()))
        return out
    
    def pv_gyro_scalar_mul(self,
                        y: torch.Tensor,
                        r: torch.Tensor | float,
                        c: float | None = None,
                        s: float | None = None,
                        eps: float = 1e-12) -> torch.Tensor:
        """
        Proper-Velocity (PV) scalar gyromultiplication:
            r ⊗_PV y = s * sinh( r * asinh( ||y|| / s ) ) * y / ||y||
        with s = 1/sqrt(c). Handles r as scalar or broadcastable tensor.

        Args:
            y: (..., d) PV vector
            r: scalar or tensor broadcastable to y[..., 1]
            c: positive curvature scale parameter (K = -c < 0)
            s: optional PV scale; if provided, c is ignored. Must satisfy s = 1/sqrt(c).
            eps: numerical epsilon

        Returns:
            (..., d) tensor = r ⊗_PV y
        """
        # --- choose s ---
        if s is None:
            if c is not None:
                s_val = 1.0 / math.sqrt(float(c))
            else:
                # derive from manifold (prefer precomputed s)
                s_val = float(getattr(self.manifold, 's', 1.0 / math.sqrt(float(getattr(self.manifold, 'c', 1.0)))))
        else:
            s_val = float(s)
        s = torch.as_tensor(s_val, dtype=y.dtype, device=y.device)

        # --- sanitize input then norms & unit direction (protect zero) ---
        y = torch.nan_to_num(y)
        norm_y = y.norm(dim=-1, keepdim=True)
        unit_y = y / norm_y.clamp_min(eps)

        # asinh(||y|| / s)  ; 允许 r 为张量（自动广播）
        a = torch.asinh(norm_y / s)                 # (...,1)
        ra = torch.as_tensor(r, dtype=y.dtype, device=y.device) * a
        # 最小侵入式：限制 sinh 的输入，避免溢出（可配置）
        ra = torch.clamp(ra, -self.scalar_sinh_clip, self.scalar_sinh_clip)
        coef = s * torch.sinh(ra) / norm_y.clamp_min(eps)

        out = coef * y
        if torch.isnan(out).any() or torch.isinf(out).any():
            with torch.no_grad():
                print("[GyroBNPV][diag] NaN/Inf after scalar_mul; norm_y min/max=",
                      float(norm_y.min().item()), float(norm_y.max().item()))
        # exact r ⊗ 0 = 0
        return torch.where(norm_y <= eps, torch.zeros_like(out), out)
    
    def frechet_mean(self, x, max_iter=1000):
        """
        计算PV流形上的Fréchet均值
        使用迭代方法，与GyroBN_new保持一致
        """
        # 对于PV流形，我们使用迭代方法计算Fréchet均值（向量化实现）
        x_flat = x.view(-1, x.shape[-1])  # (N, d)

        # 初始化均值为第一个点
        mean = x_flat[0:1].clone()  # (1, d)

        # 支持“无限迭代”以收敛（max_iter < 0 表示直到收敛或达到安全上限）
        iters = 0
        safety_cap = 5000 if max_iter < 0 else max_iter
        tol = 1e-6
        last_applied_step_norm = 0.0
        while iters < safety_cap:
            # 向量化计算 log_map(mean, x_i) for all i
            # PV_monifold 接口：log_map(u, w)
            log_maps = self.manifold.log_map(mean, x_flat)
            avg_log_map = log_maps.mean(dim=0, keepdim=True)
            # 步长 damping（可配置）
            step = avg_log_map
            step_norm = torch.linalg.norm(step).clamp_min(1e-8)
            if self.max_step is not None and self.max_step > 0:
                step = step * torch.clamp(self.max_step / step_norm, max=1.0)
            last_applied_step_norm = float(torch.linalg.norm(step).item())

            # 使用指数映射更新均值
            # PV_monifold 接口：exp_map(u, v)
            new_mean = self.manifold.exp_map(mean, step)

            # 收敛判定（欧氏范数）
            if torch.linalg.norm(new_mean - mean) < tol:
                mean = new_mean
                break

            mean = new_mean
            iters += 1

        # 记录调试信息
        self._last_mean_iters = iters
        self._last_step_norm = last_applied_step_norm
        # 返回 (d,)
        return mean.squeeze(0)
    
    def frechet_variance(self, x, mean):
        """
        与 GyroBNBase/GyroBNH 一致：距离平方的批均值（标量）。
        """
        x_flat = x.view(-1, x.shape[-1])  # (N, d)
        # 广播方式一次性计算距离
        # PV_monifold 接口：dist(x, y)
        d = self.manifold.dist(mean.view(1, -1), x_flat)
        variance = (d * d).mean()
        return torch.nan_to_num(variance, nan=0.0).clamp_min(1e-8).view(1)

    def euclid_mean_and_variance(self, x):
        """
        切空间欧式均值/方差（log-Euclidean 近似 BN 统计）。
        返回: (mean_on_manifold, scalar_var)
        """
        # log 到原点切空间
        t = self.manifold.logmap0(x, c=self.manifold.c)
        mu_t = t.mean(dim=0)  # (d,)
        # exp 回流形
        mu = self.manifold.expmap0(mu_t, c=self.manifold.c)  # (d,)
        # 标量方差（与 GyroBNBase 一致，标量而非逐通道）
        diff = t - mu_t
        var = (diff * diff).sum(dim=-1).mean().view(1)
        var = torch.nan_to_num(var, nan=0.0).clamp_min(1e-8)
        return mu, var
    
    def updating_running_statistics(self, batch_mean, batch_var):
        """更新运行统计量，与GyroBN_new保持一致"""
        if self.running_mean is None:
            self.running_mean = batch_mean.detach()
        else:
            # 使用测地线插值更新均值（调用本类封装的 geodesic）
            with torch.no_grad():
                self.running_mean = self.geodesic(self.running_mean, batch_mean.detach(), self.momentum)
        
        if self.running_var is None:
            self.running_var = batch_var.detach()
        else:
            # 方差使用线性插值
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()
    
    def geodesic(self, x, y, t):
        """
        测地线插值：geodesic(x, y, t) = exp(x, t * log(x, y))
        与GyroBN_new保持一致
        """
        return self.manifold.exp_map(x, t * self.manifold.log_map(x, y))

    def gyro_midpoint_mean(self, x):
        """
        切空间均值（log-Euclidean）：先 log 到原点切空间，做欧氏均值，再 exp 回流形。
        返回形状 (d,)。
        """
        x_flat = x.view(-1, x.shape[-1])  # (N,d)
        t = self.manifold.logmap0(x_flat, c=self.manifold.c)  # (N,d)
        mu_t = t.mean(dim=0)  # (d,)
        mu = self.manifold.expmap0(mu_t, c=self.manifold.c)  # (d,)
        return mu
    
    # 已移除多中心相关实现（_assign_to_centers/gyro_midpoint_means_multi/grouped_variance_scalar）
    
    def normalization(self, x, input_mean, input_var):
        """
        执行PV流形上的归一化，完全按照GyroBN_new的理论
        
        Args:
            x: 输入张量
            input_mean: 输入均值
            input_var: 输入方差
            
        Returns:
            归一化后的张量
        """
        # 1. 设置参数：权重映射到流形
        # self.weight 形状为 (C,) 或更高维，需与输入末维对齐
        weight = self.manifold.expmap0(self.weight, c=self.manifold.c)
        # 将均值、方差按 GyroBNBase 的方式使用（标量方差与 shift）
        if input_mean.dim() == 1:
            input_mean_b = input_mean.unsqueeze(0).expand_as(x)
        else:
            input_mean_b = input_mean
        input_var_b = input_var.view(1)
        
        # 2. 中心化：使用陀螺变换
        inv_input_mean = self.gyroinv(input_mean_b)  # 使用正确的陀螺逆元
        x_center = self.gyrotrans(inv_input_mean, x)  # 使用陀螺变换
        x_center = torch.nan_to_num(x_center)
        if torch.isnan(x_center).any() or torch.isinf(x_center).any():
            with torch.no_grad():
                print("[GyroBNPV][diag] NaN/Inf after center; var=", float(input_var_b.mean().item()))
        
        # 3. 缩放：使用陀螺标量乘法
        factor = self.shift / (input_var_b + self.eps).sqrt()
        # 方差下界，避免极小方差导致过大缩放
        factor = self.shift / (torch.clamp(input_var_b, min=self.var_floor) + self.eps).sqrt()
        if self.clamp_factor is not None and self.clamp_factor > 0:
            factor = factor.clamp(max=self.clamp_factor)
        if torch.isnan(factor).any() or torch.isinf(factor).any():
            with torch.no_grad():
                print("[GyroBNPV][diag] NaN/Inf in factor; shift mean=",
                      float(self.shift.mean().item()), " var_floor=", self.var_floor)
        # 调试打印已关闭
        x_scaled = self.pv_gyro_scalar_mul(x_center, factor)  # 使用陀螺标量乘法
        if torch.isnan(x_scaled).any() or torch.isinf(x_scaled).any():
            with torch.no_grad():
                print("[GyroBNPV][diag] NaN/Inf after scale; factor min/max/mean=",
                      float(factor.min().item()), float(factor.max().item()), float(factor.mean().item()))
        
        # 4. 偏置：使用陀螺变换
        x_normed = self.gyrotrans(weight, x_scaled)  # 使用陀螺变换
        if torch.isnan(x_normed).any() or torch.isinf(x_normed).any():
            with torch.no_grad():
                print("[GyroBNPV][diag] NaN/Inf after bias (gyrotrans)")

        # 调试打印（有限次数）
        if self.print_stats and self._dbg_seen < self._dbg_print_limit:
            with torch.no_grad():
                def _s(t):
                    t = t if isinstance(t, torch.Tensor) else torch.tensor(t)
                    return dict(min=float(t.min().item()), max=float(t.max().item()),
                                mean=float(t.mean().item()), std=float(t.std(unbiased=False).item()))
                fnorm = torch.linalg.norm(factor)
                xcn = torch.linalg.norm(x_center) / (x_center.numel() ** 0.5)
                xsn = torch.linalg.norm(x_scaled) / (x_scaled.numel() ** 0.5)
                iv = torch.nan_to_num(input_var_b, nan=0.0).view(-1)[0]
                print(f"[GyroBNPV] iters={self._last_mean_iters} step_norm={self._last_step_norm:.3e} factor_norm={float(fnorm.item()):.3e} x_center_rms={float(xcn.item()):.3e} x_scaled_rms={float(xsn.item()):.3e}")
                print(f"[GyroBNPV] factor stats: {_s(factor)}  shift={_s(self.shift)}")
                print(f"[GyroBNPV] input_var={float(iv.item()):.6e}  post_gain={float(self.post_gain.item()):.3f}")
                self._dbg_seen += 1

        # 5. 归一化后几何放大（可选）
        if self.use_post_gain:
            # 限制几何放大系数的幅度，避免过度放大导致不稳定
            gain = torch.clamp(self.post_gain, 0.5, 3.0)
            x_out = self.pv_gyro_scalar_mul(x_normed, gain)
            return x_out
        else:
            return x_normed
    
    def forward(self, x):
        """
        前向传播，与GyroBN_new保持一致
        
        Args:
            x: 输入张量，形状为 (batch_size, ...)
            
        Returns:
            归一化后的张量
        """
        # 调试打印已移除

        if self.training:
            # 训练模式：批统计（可配置是否可微；可选择欧式近似）
            if self.diff_stats:
                if self.use_euclid_stats:
                    input_mean, input_var = self.euclid_mean_and_variance(x)
                elif self.use_gyro_midpoint:
                    input_mean = self.gyro_midpoint_mean(x)
                    input_var = self.frechet_variance(x, input_mean)
                else:
                    input_mean = self.frechet_mean(x, self.max_iter)
                    input_var = self.frechet_variance(x, input_mean)
                input_mean = torch.nan_to_num(input_mean, nan=0.0)
                input_var = torch.nan_to_num(input_var, nan=0.0).clamp_min(1e-8)
            else:
                with torch.no_grad():
                    if self.use_euclid_stats:
                        input_mean, input_var = self.euclid_mean_and_variance(x)
                    elif self.use_gyro_midpoint:
                        input_mean = self.gyro_midpoint_mean(x)
                        input_var = self.frechet_variance(x, input_mean)
                    else:
                        input_mean = self.frechet_mean(x, self.max_iter)
                        input_var = self.frechet_variance(x, input_mean)
                    input_mean = torch.nan_to_num(input_mean, nan=0.0)
                    input_var = torch.nan_to_num(input_var, nan=0.0).clamp_min(1e-8)
                    # 极端回退：欧式统计
                    if torch.isnan(input_mean).any() or torch.isnan(input_var).any():
                        eu_mean = torch.nan_to_num(x.mean(dim=0), nan=0.0)
                        eu_var = torch.nan_to_num(x.var(dim=0, unbiased=False), nan=0.0).mean().view(1).clamp_min(1e-8)
                        input_mean = eu_mean
                        input_var = eu_var
                input_mean = input_mean.detach()
                input_var = input_var.detach()

            if self.track_running_stats:
                self.updating_running_statistics(input_mean, input_var)
        elif self.track_running_stats:
            # 评估模式：使用运行统计量
            input_mean = self.running_mean
            input_var = self.running_var
            input_mean = torch.nan_to_num(input_mean, nan=0.0)
            input_var = torch.nan_to_num(input_var, nan=0.0).clamp_min(1e-8)
        else:
            # 不跟踪统计量：重新计算
            input_mean = self.frechet_mean(x, self.max_iter)
            input_var = self.frechet_variance(x, input_mean)
            input_mean = torch.nan_to_num(input_mean, nan=0.0)
            input_var = torch.nan_to_num(input_var, nan=0.0).clamp_min(1e-8)
        
        # 调试打印已移除

        # 执行归一化
        output = self.normalization(x, input_mean, input_var)

        # 调试打印已移除
        return output
    
    def __repr__(self):
        """字符串表示"""
        attributes = []
        
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    val_str = f"{value.item():.4f}"
                else:
                    val_str = f"shape={tuple(value.shape)}"
                attributes.append(f"{key}={val_str}")
            else:
                attributes.append(f"{key}={value}")
        
        for name, buffer in self.named_buffers(recurse=False):
            if buffer.numel() == 1:
                val_str = f"{buffer.item():.4f}"
            else:
                val_str = f"shape={tuple(buffer.shape)}"
            attributes.append(f"{name}={val_str}")
        
        return f"{self.__class__.__name__}({', '.join(attributes)})"


class PVGyroBN1d(GyroBNPV):
    """
    修正后的1D GyroBN，完全按照GyroBN_new的理论
    """
    
    def __init__(self, manifold: PVManifold, num_features: int, momentum=0.1,
                 track_running_stats=True, eps=1e-6):
        super().__init__(
            manifold=manifold,
            shape=[num_features],
            batchdim=[0],
            momentum=momentum,
            track_running_stats=track_running_stats,
            eps=eps
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, num_features, sequence_length)
            
        Returns:
            归一化后的张量，形状与输入相同
        """
        B, C, L = x.shape
        
        # 重塑为 (B*L, C) 进行批归一化
        x_reshaped = x.permute(0, 2, 1).contiguous().view(B * L, C)
        
        # 应用GyroBN
        y_reshaped = super().forward(x_reshaped)
        
        # 恢复原始形状
        y = y_reshaped.view(B, L, C).permute(0, 2, 1).contiguous()
        
        return y


# 修正后的PVBatchNorm1d，使用正确的GyroBN实现
class PVBatchNorm1d(nn.Module):
    def __init__(self, manifold: PVManifold, num_features: int, use_gyrobn: bool = True):
        super().__init__()
        self.manifold = manifold
        self.use_gyrobn = use_gyrobn
        
        if use_gyrobn:
            # 使用修正后的GyroBN
            self.gyrobn = PVGyroBN1d(manifold, num_features)
        else:
            # 使用传统的欧几里得BN
            self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) on PV
        if self.use_gyrobn:
            # 使用修正后的GyroBN进行PV流形上的批归一化
            return self.gyrobn(x)
        else:
            # 使用传统的欧几里得BN（通过切空间映射）
            B, C, L = x.shape
            xt = self.manifold.logmap0(x.permute(0, 2, 1).contiguous().view(-1, C))
            xt = self.bn(xt.view(B, L, C).permute(0, 2, 1))
            y = self.manifold.expmap0(xt.permute(0, 2, 1).contiguous().view(-1, C)).view(B, L, C).permute(0, 2, 1)
            return y 





            