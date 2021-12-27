import torch
from torch import nn
from functional import sparse_relu

class ReLUWithSparsity(nn.Module):
    def __init__(self, beta, rho=0.05):
        '''
            Params
            ------
                beta: float
                    Controls the weight of the sparsity penalty term.

                rho: float
                    Sparsity parameter, typically a small value close to zero (i.e 0.05), which
                    constraint the activation of each neuron. To satisfy this constraint, the
                    hidden unit’s activations must mostly be near 0.
        '''
        super(ReLUWithSparsity, self).__init__()
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("rho", torch.tensor(rho))
        # self.beta = torch.nn.Param(beta)
        # self.rho = torch.tensor(rho)

    def forward(self, x):
        return sparse_relu.apply(x, self.rho, self.beta)
