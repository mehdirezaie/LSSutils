""" Kernel Smoother SN Hubble Diagram

"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from lssutils.utils import Cosmology

def kernel(z_i, z, delta=0.05, **kw):
    arg = (np.log10((1.+z_i)/(1.+z))/delta)
    return np.exp(-0.5*arg*arg)

def chi2(y1, y2, sigma):
    # computes RMSE
    # change mean to sum, MSE to chi2
    chi = ((y1-y2)/(sigma))
    return np.sqrt((chi*chi).mean())

class KernelSmoother(object):
    #
    def __init__(self, fn='./data/union.txt', test_size=0.33, random_state=0):
        data = np.loadtxt(fn)
        #
        # add train test split
        train, test   = train_test_split(data, 
                                         random_state=random_state,
                                         test_size=test_size)
        #
        #
        self.z_data  = train[:,0]
        self.mu_data = train[:,1]
        self.mu_err  = train[:,2]       
        
        self.z_data_t  = test[:,0]
        self.mu_data_t = test[:,1]
        self.mu_err_t  = test[:,2]       
        
        self.mu_guess  = None
        
    def _init_cosmology(self, om_m=0.26, om_L=0.7, h=0.69, zmin=0.01, zmax=1.5, nbin=30):
        # 0.69
        # 0.26
        # theoretical mu
        self.z_grid  = np.linspace(zmin, zmax, nbin)
        #self.z_grid  = np.logspace(np.log10(zmin), np.log10(zmax), nbin) # logarithmic grid
        theory       = Cosmology(om_m=om_m, om_L=om_L, h=h)        
        self.mu_th   = 5*np.log10(np.vectorize(theory.DL)(self.z_grid)) + 25
        
        # interpolate the theory on data points
        self.mu_th_spl  = IUS(self.z_grid, self.mu_th)
        
        # guess
        self.mu_g_grid    = self.mu_th        
        self.mu_g_data    = self.mu_th_spl(self.z_data)
        self.mu_g_data_t  = self.mu_th_spl(self.z_data_t)
        
        # chi2
        self.chi2     = [chi2(self.mu_g_data,   self.mu_data,   self.mu_err)]
        self.err      = [chi2(self.mu_g_data_t, self.mu_data_t, self.mu_err_t)]
        self.baseline = [chi2(np.mean(self.mu_data), self.mu_data, self.mu_err),
                         chi2(np.mean(self.mu_data), self.mu_data_t, self.mu_err_t)]

    
    def _init_weights(self, **kw):        
        # kernel on data/grid points
        weights_zdata = []
        for zi in self.z_data:
            kr  = kernel(self.z_data, zi, **kw)
            krn = kr.sum()  # normalization
            weights_zdata.append(kr/krn)
        self.weights_zdata = np.array(weights_zdata) 

        #
        weights_zgrid = []
        for zi in self.z_grid:
            kr  = kernel(self.z_data, zi, **kw)
            krn = kr.sum()  # normalization
            weights_zgrid.append(kr/krn)
        self.weights_zgrid = np.array(weights_zgrid)
        
        weights_zdata_t = []
        for zi in self.z_data_t:
            kr  = kernel(self.z_data, zi, **kw)
            krn = kr.sum()  # normalization
            weights_zdata_t.append(kr/krn)
        self.weights_zdata_t = np.array(weights_zdata_t)
        
        
    def smooth(self, marginalize=False, verbose=False):
        # smooth on data points
        self.delta_mu_data    = self.mu_data - self.mu_g_data 
        self.smooth_deltamu_d = self.weights_zdata.dot(self.delta_mu_data)
        self.smooth_deltamu_g = self.weights_zgrid.dot(self.delta_mu_data)
        self.smooth_deltamu_t = self.weights_zdata_t.dot(self.delta_mu_data)
        #
            
        self.smooth_mu_data    = self.smooth_deltamu_d + self.mu_g_data
        self.smooth_mu_grid    = self.smooth_deltamu_g + self.mu_g_grid
        self.smooth_mu_data_t  = self.smooth_deltamu_t + self.mu_g_data_t
        #
        
        #
        self.mu_g_data       = self.smooth_mu_data
        self.mu_g_grid       = self.smooth_mu_grid
        self.mu_g_data_t     = self.smooth_mu_data_t
        #
        if marginalize:
            ''' results in poor performance '''
            raise RuntimeWarning('Not implemented yet')
            #offset1 = np.mean((self.mu_g_data-self.mu_data)/self.mu_err)
            #self.mu_g_data += offset1                        
            #self.mu_g_grid += offset1                        
            #self.smooth_deltamu_g -= offset1
            
        chi2_train = chi2(self.mu_g_data,   self.mu_data,   self.mu_err)
        chi2_test = chi2(self.mu_g_data_t, self.mu_data_t, self.mu_err_t)
        self.chi2.append(chi2_train)
        self.err.append(chi2_test)
        return chi2_train, chi2_test
        
    def plot_mu_rmse(self, ax=None):
        if ax is None:fig, ax = plt.subplots(ncols=2, figsize=(12,4))
        #
        ax[0].plot(self.z_data,   self.mu_data, '.',   color='k',    alpha=0.1, label='Union')
        ax[0].plot(self.z_data_t, self.mu_data_t, '.', color='navy', alpha=0.5, label=None)
        ax[0].plot(self.z_grid,   self.smooth_mu_grid, 'r-', label='Smoothed')
        ax[0].plot(self.z_grid,   self.mu_th, 'k--', label=r'$\Lambda$CDM')
        #ax[1].axhline(chi2(SN.mu_th_spl(SN.z_data), SN.mu_data, SN.mu_err))
        ax[1].text(0, self.chi2[0]*1.01, 'LCDM', color='k')
        ax[1].scatter(0, self.chi2[0], marker='.', color='k')
        ax[1].scatter(0, self.err[0], marker='.', color='r')
        ax[1].plot(self.chi2, ls='-',  label='train RMSE', color='k')
        ax[1].plot(self.err,  ls='--', label='test RMSE', color='r')
        #ax[1].axhline(1, color='k', ls=':')
        
        # annotation
        ax[0].set_xscale('log')
        ax[0].legend(loc=4)
        ax[0].set_xlabel('redshift')
        ax[0].set_ylabel(r'$\mu$')
        ax[1].set_xlabel('iteration')
        ax[1].set_ylabel(r'RMSE')
        ax[1].set_ylim(0.8, 1.2)
        # ax[1].set_ylim(0.99, 1.01)
        ax[1].legend()
        
        
if __name__ == '__main__':
    

    N_iteration = 5
    SN = KernelSmoother(fn='../../data/union.txt')
    SN._init_cosmology(nbin=100)
    SN._init_weights(delta=0.05)
    
    for i in range(N_iteration):
        chi2s = SN.smooth()
        print(f'{i}, {chi2s}')
        
    #fig, ax = plt.subplots(nrows=2, figsize=(6,8))        
    #SN.plot_mu_rmse(ax=ax)