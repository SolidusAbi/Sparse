import torch
from torch.nn import functional as F

def kl_divergence(p, q: torch.Tensor):
    '''
        Kullback-Leibler (KL) divergence between a Bernoulli random variable with mean
        p and a Bernoulli random variable with mean q.
        
        Notas:
            q.flatten(1) para que funcione en convoluciones... no sé si está bien aplicado
        
    '''
    rho_hat = torch.mean(F.sigmoid(q).flatten(1), 1) # sigmoid because we need the probability distributions
    rho = torch.ones(rho_hat.shape).to(q.device) * p
    return torch.sum(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))