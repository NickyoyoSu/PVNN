import torch
import torch.nn as nn

from .Geometry.constantcurvature import classes
from .GyroBNBase import GyroBNBase

class GyroBNH(GyroBNBase):
    """
    GyroBN layer for Constant Curvature Spaces (CCSs): Stereographic, Hyperboloid, Klein or Sphere models.

    Note: This implementation only supports input of shape [batch_size, dim].

    Args:
        shape (list[int]): Manifold dimensions, e.g., [d] for vectors in L^d, which are represented as vectors in R^{d+1}.
        model (str, default='Stereographic'): Type of CCS model. One of {"Stereographic", "Hyperboloid", "Klein", "Sphere"}.
        K (float, default=-1.0): Curvature of the CCS.
        max_iter (int, default=1000): Maximum iterations for computing the Fréchet mean.
    """

    def __init__(self, shape, model='Stereographic', K=-1.0, max_iter=1000, momentum=0.1,
                 track_running_stats=True, init_1st_batch=False,eps=1e-6):
        super().__init__(shape=shape, batchdim=[0], momentum=momentum,
                         track_running_stats=track_running_stats, init_1st_batch=init_1st_batch, eps=eps)
        self.model = model;self.K=K;self.max_iter=max_iter
        self.get_manifold()
        self.set_parameters()
        if self.track_running_stats:
            self.register_running_stats()

    def get_manifold(self):
        self.manifold = classes[self.model](K=self.K)

    def set_parameters(self):
        mean_shape, var_shape = self._get_param_shapes()
        self.weight = nn.Parameter(self.manifold.zero_tan(mean_shape))
        self.shift = nn.Parameter(torch.ones(*var_shape))

    def register_running_stats(self):
        """
        In [a], they use the following for the variance:
            Initialization:
                self.updates = 0
            updates:
                self.running_var = (1 - 1 / self.updates) * self.running_var + batch_var / self.updates
                self.updates += 1
        But we follow the standard Euclidean BN.
        [a] Differentiating through the Fréchet Mean
        """
        if self.init_1st_batch:
            self.running_mean = None
            self.running_var = None
        else:
            mean_shape, var_shape = self._get_param_shapes()
            running_mean = self.manifold.zero(mean_shape)
            running_var = torch.ones(*var_shape)
            self.register_buffer('running_mean', running_mean)
            self.register_buffer('running_var', running_var)

    def forward(self, x):
        if self.training:
            input_mean = self.manifold.frechet_mean(x,max_iter=self.max_iter)
            input_var = self.manifold.frechet_variance(x, input_mean)
            if self.track_running_stats:
                self.updating_running_statistics(input_mean, input_var)
        elif self.track_running_stats:
            input_mean = self.running_mean
            input_var = self.running_var
        else:
            input_mean = self.manifold.frechet_mean(x,max_iter=self.max_iter)
            input_var = self.manifold.frechet_variance(x, input_mean)

        output = self.normalization(x,input_mean,input_var)
        return output

    def normalization(self,x,input_mean,input_var):
        # set parameters
        weight = self.manifold.exp0(self.weight)
        # centering
        inv_input_mean = self.manifold.gyroinv(input_mean)
        x_center = self.manifold.gyrotrans(inv_input_mean,x)
        # shifting
        factor = self.shift / (input_var + self.eps).sqrt()
        x_scaled = self.manifold.gyroscalarprod(x_center,factor)
        # biasing
        x_normed = self.manifold.gyrotrans(weight, x_scaled)
        return x_normed

