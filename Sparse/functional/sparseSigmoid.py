import torch
from torch import autograd
from .utils import kl_divergence

class sparse_sigmoid(autograd.Function):
    '''
        A Sigmoid which take all autograd function for creating layers with sparse
        output.
        .. note::
            Code adapted from:
            https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    '''

    @staticmethod
    def forward(ctx, x, rho, beta):
        '''
            Params
            ------
                ctx: 
                    ctx is a context object that can be used to stash information for
                    backward computation.
                
                x: Tensor
                    A tensor which contains the input.
                
                rho: float
                    Sparsity parameter, typically a small value close to zero (i.e 0.05), which
                    constraint the activation of each neuron. To satisfy this constraint, the
                    hidden unit’s activations must mostly be near 0.

                beta: float
                    Controls the weight of the sparsity penalty term.
        '''
        output = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(output, rho, beta)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        '''
            In the backward pass we receive a Tensor containing the gradient of the loss
            with respect to the output, and we need to compute the gradient of the loss
            with respect to the input.
        '''
        output, rho, beta, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = (grad_input * (output*(1-output))) + (beta*kl_divergence(rho, output, apply_sigmoid=False))

        return grad_input, None, None


class sparse_sigmoid_2d(autograd.Function):
    '''
        A Sigmoid which take all autograd function for creating layers with sparse
        output applied in 2D.
    '''

    @staticmethod
    def forward(ctx, x, rho, beta):
        '''
            Params
            ------
                ctx: 
                    ctx is a context object that can be used to stash information for
                    backward computation.
                
                x: Tensor
                    A tensor which contains the input.
                
                rho: float
                    Sparsity parameter, typically a small value close to zero (i.e 0.05), which
                    constraint the activation of each neuron. To satisfy this constraint, the
                    hidden unit’s activations must mostly be near 0.

                beta: float
                    Controls the weight of the sparsity penalty term.
        '''
        output = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(output, rho, beta)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        '''
            In the backward pass we receive a Tensor containing the gradient of the loss
            with respect to the output, and we need to compute the gradient of the loss
            with respect to the input.
        '''
        output, rho, beta, = ctx.saved_tensors
        B,_,H,W = output.shape

        grad_input = grad_output.clone()
        grad_input *= output*(1-output)
        kl_loss = kl_divergence(rho, input)[None,:,None,None].expand(B,-1,H,W)
        grad_input += beta*kl_loss

        return grad_input, None, None