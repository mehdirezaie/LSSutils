



class Posterior:
    """ Log Posterior for PNGModel
    """
    def __init__(self, model, y, invcov, x):
        self.model = model
        self.y = y
        self.invcov = invcov
        self.x = x

    def logprior(self, theta):
        ''' The natural logarithm of the prior probability. '''
        lp = 0.
        # unpack the model parameters from the tuple
        fnl, noise = theta
        
        # uniform prior on fNL
        fmin = -1000. # lower range of prior
        fmax = 1000.  # upper range of prior
        # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
        lp += 0. if fmin < fnl < fmax else -np.inf
        
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
        fnl, noise = theta
        
        # evaluate the model
        md = self.model(self.x, fnl=fnl, noise=noise)
        # return the log likelihood
        return -0.5 * (self.y-md).dot(self.invcov.dot(self.y-md))

    def logpost(self, theta):
        '''The natural logarithm of the posterior.'''
        return self.logprior(theta) + self.loglike(theta)