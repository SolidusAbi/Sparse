import math
from torch import Tensor
import numpy as np
from matplotlib import pyplot as plt

def binaryEntropy(x: Tensor) -> tuple:
  '''
    Calculate entropy for a list of binary random variables
    
    Parameters
    ----------
        x: Tensor, shape (samples, features)
            the probability of the variable to be 1.

    Return
    ------    
        entropy: Tensor, shape (samples, features)
        
        sum(entropy): float    
  '''
  entropy = - x*x.log2() - (1-x)*(1-x).log2()
  entropy[x*(1 - x) == 0] = 0
  return entropy, entropy.sum()

def maxEntropy(n: int, k: int) -> float:
    '''
        The maximum entropy we could get with n units and k winners
    '''

    s = float(k)/n
    if s > 0.0 and s < 1.0:
        entropy = - s * math.log(s,2) - (1 - s) * math.log(1 - s,2)
    else:
        entropy = 0

    return n*entropy

def plotDutyCycles(dutyCycle, filePath):
  '''
    Create plot showing histogram of duty cycles.

    Parameters
    ----------
        dutyCycle: Tensor, shape (n_features)
            the duty cycle estimation of each unit
        
        filePath: Str
            Full filename of image file
  '''
  _,entropy = binaryEntropy(dutyCycle)
  bins = np.linspace(0.0, 0.3, 200)
  plt.hist(dutyCycle, bins, alpha=0.5, label='All cols')
  plt.title("Histogram of duty cycles, entropy=" + str(float(entropy)))
  plt.xlabel("Duty cycle")
  plt.ylabel("Number of units")
  plt.savefig(filePath)
  plt.close()