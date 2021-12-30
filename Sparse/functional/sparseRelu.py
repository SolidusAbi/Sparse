import torch
from torch import autograd
from .utils import kl_divergence

class sparse_relu(autograd.Function):
    '''
        A ReLU which take all autograd function for creating layers with sparse
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
        ctx.save_for_backward(x, rho, beta)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        '''
            In the backward pass we receive a Tensor containing the gradient of the loss
            with respect to the output, and we need to compute the gradient of the loss
            with respect to the input.
        '''
        input, rho, beta, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[input > 0] = grad_input[input > 0] + (beta*kl_divergence(rho, input))
        return grad_input, None, None

class sparse_relu_2d(autograd.Function):
    '''
        A ReLU which take all autograd function for creating layers with sparse
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
        ctx.save_for_backward(x, rho, beta)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        '''
            In the backward pass we receive a Tensor containing the gradient of the loss
            with respect to the output, and we need to compute the gradient of the loss
            with respect to the input.
        '''
        input, rho, beta, = ctx.saved_tensors
        B,_,H,W = input.shape

        grad_input = grad_output.clone()
        kl_loss = kl_divergence(rho, input)[None,:,None,None].expand(B,-1,H,W)

        grad_input[input < 0] = 0
        grad_input[input > 0] = grad_input[input > 0] + (beta*kl_loss)[input > 0]
        return grad_input, None, None