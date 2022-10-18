import numpy as np
from lssutils.utils import histogram_cell


def gelman_rubin(chain):
    ssq = np.var(chain, axis=1, ddof=1)
    W = np.mean(ssq, axis=0)
    pb = np.mean(chain, axis=1)
    pbb = np.mean(pb, axis=0)
    m = chain.shape[0]
    n = chain.shape[1]
    B = n / (m - 1) * np.sum((pbb - pb)**2, axis=0)
    var_p = (n - 1) / n * W + 1 / n * B
    rhat = np.sqrt(var_p / W)
    return rhat



class Posterior:
    """ Log Posterior for PNGModel
    """
    def __init__(self, model, y, invcov, x):
        self.model = model
        self.y = y
        self.invcov = invcov
        self.x = x
        self.x_ = np.arange(x.min(), x.max()+1)

    def logprior(self, theta):
        ''' The natural logarithm of the prior probability. '''
        lp = 0.
        # unpack the model parameters from the tuple
        fnl, b, noise = theta
        
        # uniform prior on fNL
        fmin = -1000. # lower range of prior
        fmax = 1000.  # upper range of prior
        # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
        lp += 0. if fmin < fnl < fmax else -np.inf

        # uniform prior on noise
        b_min = 0.0
        b_max = 5.0
        lp += 0. if b_min < b < b_max else -np.inf        
        
        # uniform prior on noise
        noise_min = -0.001
        noise_max =  0.001
        lp += 0. if noise_min < noise < noise_max else -np.inf
        
        ## Gaussian prior on ?
        #mmu = 3.     # mean of the Gaussian prior
        #msigma = 10. # standard deviation of the Gaussian prior
        #lp += -0.5*((m - mmu)/msigma)**2

        return lp

    def loglike(self, theta):
        '''The natural logarithm of the likelihood.'''
        # unpack the model parameters
        fnl, b, noise = theta
        
        # evaluate the model
        md_ = self.model(self.x_, fnl=fnl, b=b, noise=noise)
        md = histogram_cell(self.x_, md_, bins=self.x)[1]
        # return the log likelihood
        return -0.5*(self.y-md).dot(self.invcov.dot(self.y-md))

    def logpost(self, theta):
        '''The natural logarithm of the posterior.'''
        return self.logprior(theta) + self.loglike(theta)



class LogPosterior:
    """ Log Posterior for PNGModel
    """
    def __init__(self, model, y, invcov, x):
        self.model = model
        self.y = y
        self.invcov = invcov
        self.x = x
        self.x_ = np.arange(x.min(), x.max()+1)

    def logprior(self, theta):
        ''' The natural logarithm of the prior probability. '''
        lp = 0.
        # unpack the model parameters from the tuple
        fnl, b, noise = theta
        
        # uniform prior on fNL
        fmin = -1000. # lower range of prior
        fmax = 1000.  # upper range of prior
        # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
        lp += 0. if fmin < fnl < fmax else -np.inf

        # uniform prior on noise
        b_min = 0.0
        b_max = 5.0
        lp += 0. if b_min < b < b_max else -np.inf        
        
        # uniform prior on noise
        noise_min = -0.001
        noise_max =  0.001
        lp += 0. if noise_min < noise < noise_max else -np.inf
        
        ## Gaussian prior on ?
        #mmu = 3.     # mean of the Gaussian prior
        #msigma = 10. # standard deviation of the Gaussian prior
        #lp += -0.5*((m - mmu)/msigma)**2

        return lp

    def loglike(self, theta):
        '''The natural logarithm of the likelihood.'''
        # unpack the model parameters
        fnl, b, noise = theta
        
        # evaluate the model
        md_ = self.model(self.x_, fnl=fnl, b=b, noise=noise)
        md_b = histogram_cell(self.x_, md_, bins=self.x)[1]
        if np.any(md_b < 0):
            return -np.inf
        else:            
            md = np.log10(md_b)
            # return the log likelihood
            return -0.5*(self.y-md).dot(self.invcov.dot(self.y-md))

    def logpost(self, theta):
        '''The natural logarithm of the posterior.'''
        return self.logprior(theta) + self.loglike(theta)
