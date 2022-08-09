from torch import nn
from torch.nn.functional import cross_entropy
from .VariationalLayer import VariationalLayer

class SGVBL(nn.Module):
    ''' 
        Stocastich Gradient Variational Bayes (SGVB) Loss function.
        More details in https://arxiv.org/pdf/1506.02557.pdf and https://arxiv.org/pdf/1312.6114.pdf
    '''

    def __init__(self, model, train_size, loss=cross_entropy):
        super(SGVBL, self).__init__()
        self.train_size = train_size
        self.net = model
        self.loss = loss

        self.variational_layers = []
        for module in model.modules():
            if isinstance(module, (VariationalLayer)):
                self.variational_layers.append(module)

    def forward(self, input, target, kl_weight=1.0):
        assert not target.requires_grad
        kl = 0.0
        for layer in self.variational_layers:
            kl += layer.kl_reg()
            
        return self.loss(input, target) * self.train_size + kl_weight * kl

