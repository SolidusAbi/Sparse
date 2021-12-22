from torch import nn
import torch
import math
import abc

from functional import k_winners
from .utils import binaryEntropy, maxEntropy

class KWinnersBase(nn.Module):
  """
  Base KWinners class
  """
  __metaclass__ = abc.ABCMeta


  def __init__(self, n, k, kInferenceFactor=1.0, boostStrength=1.0,
               boostStrengthFactor=1.0, dutyCyclePeriod=1000):
    """
    :param n:
      Number of units
    :type n: int
    :param k:
      The activity of the top k units will be allowed to remain, the rest are set
      to zero
    :type k: int
    :param kInferenceFactor:
      During inference (training=False) we increase k by this factor.
    :type kInferenceFactor: float
    :param boostStrength:
      boost strength (0.0 implies no boosting).
    :type boostStrength: float
    :param boostStrengthFactor:
      Boost strength factor to use [0..1]
    :type boostStrengthFactor: float
    :param dutyCyclePeriod:
      The period used to calculate duty cycles
    :type dutyCyclePeriod: int
    """
    super(KWinnersBase, self).__init__()
    assert 0 < k < n
    assert (boostStrength >= 0.0)

    self.n = n
    self.k = k
    self.kInferenceFactor = kInferenceFactor
    self.learningIterations = 0

    # Boosting related parameters
    self.boostStrength = boostStrength
    self.boostStrengthFactor = boostStrengthFactor
    self.dutyCyclePeriod = dutyCyclePeriod

  def getLearningIterations(self):
    return self.learningIterations

  @abc.abstractmethod
  def updateDutyCycle(self, x):
    '''
        Running average of each unit’s duty cycle (i.e. how frequently 
        it has been one of the top k units). It is necessary in order to
        estimate the boosting coefficient.

        Duty cycles are updated according to the following formula:
        .. math::
            dutyCycle = \\frac{dutyCycle \\times \\left( period - batchSize \\right)
                                + newValue}{period}

        Parameters
        ----------
          x: Tensor
            Current activity of each unit
    '''
    raise NotImplementedError


  def updateBoostStrength(self):
    """
    Update boost strength using given strength factor during training
    """
    if self.training:
      self.boostStrength = self.boostStrength * self.boostStrengthFactor


  def entropy(self):
    """
    Returns the current total entropy of this layer
    """
    if self.k < self.n:
      _, entropy = binaryEntropy(self.dutyCycle)
      return entropy
    else:
      return 0


  def maxEntropy(self):
    '''
        The maximum entropy we could get with n units and k winners.
        Returns the maximum total entropy we can expect from this layer.
    '''
    return maxEntropy(self.n, self.k)


class KWinners(KWinnersBase):
  """
  Applies K-Winner function to the input tensor
  See :class:`htmresearch.frameworks.pytorch.functions.k_winners`
  """


  def __init__(self, n, k, kInferenceFactor=1.0, boostStrength=1.0,
               boostStrengthFactor=1.0, dutyCyclePeriod=1000):
    '''
    Parameters
    ----------
        n: Int
            Number of units.
        
        k: Int
            The activity of the top k units will be allowed to remain, the
            rest are set to zero. It must to be lower than n.
        
        kInferenceFactor: float
            During inference (training=False) we increase k by this factor.

        boostStrength: float
            Boost strength (0.0 implies no boosting). This values must be 
            greater than 0.

        boostStrengthFactor: float
            Boost strength factor to use [0..1].

        dutyCyclePeriod: Int
            The period used to calculate duty cycles
    
    '''

    super(KWinners, self).__init__(n=n, k=k,
                                   kInferenceFactor=kInferenceFactor,
                                   boostStrength=boostStrength,
                                   boostStrengthFactor=boostStrengthFactor,
                                   dutyCyclePeriod=dutyCyclePeriod)
    
    self.register_buffer("dutyCycle", torch.zeros(self.n))


  def forward(self, x):
    # Apply k-winner algorithm if k < n, otherwise default to standard RELU
    if self.training:
      k = self.k
    else:
      k = min(int(round(self.k * self.kInferenceFactor)), self.n)

    x = k_winners.apply(x, self.dutyCycle, k, self.boostStrength)

    if self.training:
      self.updateDutyCycle(x)

    return x


  def updateDutyCycle(self, x):
    '''
        Running average of each unit’s duty cycle (i.e. how frequently 
        it has been one of the top k units). It is necessary in order to
        estimate the boosting coefficient.

        Duty cycles are updated according to the following formula:
        .. math::
            dutyCycle = \\frac{dutyCycle \\times \\left( period - batchSize \\right)
                                + newValue}{period}

        Parameters
        ----------
          x: Tensor, shape (batch_size, features)
            Current activity of each unit
    '''
    batchSize = x.shape[0]
    self.learningIterations += batchSize
    period = min(self.dutyCyclePeriod, self.learningIterations)
    
    self.dutyCycle.mul_(period - batchSize)
    self.dutyCycle.add_(x.gt(0).sum(dim=0, dtype=torch.float))
    self.dutyCycle.div_(period)

