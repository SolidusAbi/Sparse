import torch
from torch import sigmoid

def kl_divergence(p: float, q: torch.Tensor, apply_sigmoid=True) -> torch.Tensor:
    '''
        Kullback-Leibler (KL) divergence between a Bernoulli random variable with mean
        p and a Bernoulli random variable with mean q.

        For convolutional output tensor (shape B,C,H,W) the kl divergence is estimated per
        channel.

        Params
        ------
            p: float
                Sparsity parameter, typically a small value close to zero (i.e 0.05).

            q: torch.Tensor
                The output of a layer.

            apply_sigmoid: Bolean
                Indicate if it is necessary to apply sigmoid function to q in order to
                obtain the probability distribution.
        Return
        ------
            kl divergence estimation: torch.Tensor
                In general return a unique value but in convolutional output the tensor
                shape is defined by the number of Channels, i.e shape [1, C].
    '''
    # check if tensor belong to a convolutional output or not
    dim = 2 if len(q.shape) == 4 else 1

    q = sigmoid(q) if apply_sigmoid else q # sigmoid because we need the probability distributions

    rho_hat = torch.mean(q.flatten(dim), dim) 
    rho = torch.ones(rho_hat.shape).to(q.device) * p
    return torch.sum(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)), axis=0)