import torch
from torch import nn
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from .VariationalLayer import VariationalLayer

def compute_log_alpha(log_sigma, theta):
  r''' 
      Compute the log \alpha values from \theta and log \sigma^2.

      The relationship between \sigma^2, \theta, and \alpha as defined in the
      paper https://arxiv.org/abs/1701.05369 is \sigma^2 = \alpha * \theta^2.

      This method calculates the log \alpha values based on this relation:
        \log(\alpha) = 2*\log(\sigma) - 2*\log(\theta)
  ''' 
  log_alpha = log_sigma * 2.0 - 2.0 * torch.log(1e-16 + torch.abs(theta))
  log_alpha = torch.clamp(log_alpha, -10, 10) # clipping for a numerical stability
  return log_alpha

# Linear Sparse Variational Dropout
# See https://arxiv.org/pdf/1701.05369.pdf for details
class LinearSVD(nn.Linear, VariationalLayer):
    def __init__(self, in_features, out_features, p_threshold = 0.952572, bias=True) -> None:
        r'''
            Parameters
            ----------
                in_features: int,
                    Number of input features.

                out_features: int,
                    Number of output features.
                
                p_threshold: float,
                    It consists in the \rho (binary dropout rate) threshold used in order to discard the weight.
                    In this approach, an Gaussian Dropout is being used which std is \alpha = \rho/(1-\rho) so, 
                    Infinitely large \sigma_{ij} corresponds to infinitely large multiplicative noise in w_{ij}. By 
                    default, the threshold is set to 0.952572 (\log(\sigma) ~ 3).

                bias: bool,
                    If True, adds a bias term to the output.
        '''
        super(LinearSVD, self).__init__(in_features, out_features, bias)
    
        self.log_alpha_threshold = np.log(p_threshold / (1-p_threshold))
        self.log_sigma = Parameter(torch.Tensor(out_features, in_features))

        self.log_sigma.data.fill_(-5) # Initialization based on the paper, Figure 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # LRT = local reparametrization trick (For details, see https://arxiv.org/pdf/1506.02557.pdf)
            lrt_mean =  F.linear(x, self.weight, self.bias)
            lrt_std = torch.sqrt(F.linear(x * x, torch.exp(self.log_sigma * 2.0)) + 1e-8)
            eps = torch.normal(0, torch.ones_like(lrt_std))
            return lrt_mean + lrt_std * eps
        
        self.log_alpha = compute_log_alpha(self.log_sigma, torch.abs(self.weight))
        return F.linear(x, self.weight * (self.log_alpha < self.log_alpha_threshold).float(), self.bias)

    def kl_reg(self):
        k1, k2, k3 = torch.Tensor([0.63576]).cuda(), torch.Tensor([1.8732]).cuda(), torch.Tensor([1.48695]).cuda()
        k1, k2, k3 = torch.Tensor([0.63576]).cuda(), torch.Tensor([1.8732]).cuda(), torch.Tensor([1.48695]).cuda()
        kl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        return -(torch.sum(kl))
