import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .VariationalLayer import VariationalLayer

class LinearCD(nn.Linear, VariationalLayer):
    r'''
        Linear layer with Concrete Dropout regularization.

        Code strongly inspired by: 
            https://github.com/danielkelshaw/ConcreteDropout/blob/master/condrop/concrete_dropout.py

        Note the relationship between the weight regularizer (w_reg) and dropout regularization (drop_reg):
        
            w_reg/drop_reg = (l^2)/2 
        
        with prior lengthscale l (number of in_features). 
        
        Note also that the factor of two should be ignored for cross-entropy loss, and used only for the
        Euclidean loss.
    '''
    def __init__(self, in_features, out_features, bias=True, w_reg=1e-6, drop_reg=1e-3, init_min=0.05, init_max=0.1):
        super(LinearCD, self).__init__(in_features, out_features, bias)        
        logit_init_min = np.log(init_min) - np.log(1. - init_min)
        logit_init_max = np.log(init_max) - np.log(1. - init_max)
        
        # The probability of deactive a neuron.
        self.logit_p = nn.Parameter(torch.rand(in_features) * (logit_init_max - logit_init_min) + logit_init_min)
        
        # The weight and Dropout regularization term.
        self.w_reg = w_reg
        self.drop_reg = drop_reg

    def forward(self, x):
        if self.training:
            return F.linear(self.concrete_bernoulli(x), self.weight, self.bias)

        return F.linear(x, self.weight, self.bias)

    def concrete_bernoulli(self, x):
        eps = 1e-8
        unif_noise = torch.cuda.FloatTensor(*x.size()).uniform_() if self.logit_p.is_cuda else torch.FloatTensor(*x.size()).uniform_()

        p = torch.sigmoid(self.logit_p)
        tmp = .1

        drop_prob = (torch.log(p + eps) - torch.log((1-p) + eps) + torch.log(unif_noise + eps)
        - torch.log((1. - unif_noise) + eps))
        drop_prob = torch.sigmoid(drop_prob / tmp)

        random_tensor = 1 - drop_prob
        retain_prob = 1 - p # rescale factor typical for dropout
        return torch.mul(x, random_tensor) / retain_prob

    def kl_reg(self):
        # KL regularization term
        # For more deatils, see https://arxiv.org/pdf/1705.07832.pdf
        p = torch.sigmoid(self.logit_p)

        square_param = torch.sum(torch.pow(self.weight, 2), dim=0)
        if self.bias is not None: # Tiene sentido el bias?!
            square_param += torch.sum(torch.pow(self.bias, 2))

        # Weights regularization divided by (1-p) because of the rescaling 
        # factor in the dropout distribution.
        weights_reg = self.w_reg * square_param / (1.0 - p) 

        # dropout regularization term (bernolli entropy) 
        d = self.weight.size(1)
        dropout_reg = (p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        dropout_reg = (self.drop_reg * d) * dropout_reg

        kl_reg = torch.sum(weights_reg + dropout_reg)
        return kl_reg