# Sparse Network

# Requiriment

* PyTorch, Torchvision...

# Reference
1. **How do neurons operate on sparse distributed representations? A mathematical theory of sparsity, neurons and active dendrites**
    - [Paper](https://arxiv.org/pdf/1601.00720.pdf)
1. **How Can We Be So Dense? The Benefits of Using Highly Sparse Representations**
    - [Paper](https://arxiv.org/pdf/1903.11257.pdf)
    - [Code](https://github.com/numenta/htmpapers/tree/master/arxiv/how_can_we_be_so_dense)

1. K-Winner implementation:
    - [Discussion](https://discuss.pytorch.org/t/k-winner-take-all-advanced-indexing/24348)

# Miscellaneous
1. [Sparse Coding](http://ufldl.stanford.edu/tutorial/unsupervised/SparseCoding/)
1. [Sparse Distributed Representations](https://discuss.pytorch.org/t/k-winner-take-all-advanced-indexing/24348)
1. [ISTA Implementation](https://github.com/lpjiang97/sparse-coding/blob/master/src/model/SparseNet.py)
1. [Bayesian Bits](https://arxiv.org/pdf/2005.07093.pdf)
1. [Sparse AutoEncoder](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf)
1. Sparse AutoEncoder examples
    * [Using L1](https://debuggercafe.com/sparse-autoencoders-using-l1-regularization-with-pytorch/)
    * [Using KL Divergence](https://debuggercafe.com/sparse-autoencoders-using-kl-divergence-with-pytorch/)
1. [Understanding Pytorch hooks](https://www.kaggle.com/sironghuang/understanding-pytorch-hooks)

## ...
$ KL(\rho||\hat{\rho})_{Ber} = \rho \log\left(\dfrac{\rho}{\hat{\rho}}\right) + (1-\rho)\log\left(\dfrac{1-\rho}{1-\hat{\rho}}\right)$

### Gradient based on $\hat{\rho}$
$\dfrac{\partial KL(\rho||\hat{\rho})_{Ber}}{\partial\hat{\rho}} = \dfrac{1-\rho}{1-\hat{\rho}}-\dfrac{\rho}{\hat{\rho}} = -\dfrac{\hat{\rho}-\rho}{\left(\hat{\rho}-1\right)\hat{\rho}}$



# TODO
- [ ] Blind Spot Convolution
     - Just observe the noisy context of a pixel
     - 'Efficient Blind-Spot Neural Network Architecture for Image Denoising' [\[Ref\]](https://arxiv.org/pdf/2008.11010.pdf)
- [ ] Learning Hybrid Sparsity Prior for Image Restoration
     - [Paper](https://arxiv.org/pdf/1807.06920.pdf)
- [ ] Sparse Linear
    - Instead of activation functions
    - Reference: [Extending PyTorch](https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd)
- [ ] Important! Include a configuration where you can set the sparse property with a 'constant prunning' or 'gradual prunning'.

