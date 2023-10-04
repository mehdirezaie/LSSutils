#
# Compute window matrix
# Hivon et al 2001
#
import sys
import fitsio as ft
import healpy as hp
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sympy.physics.wigner import wigner_3j
from lssutils import setup_logging, CurrentMPIComm
from lssutils import utils as ut

def model(l, *p):
    return p[0]*np.log10(l)+p[1]

def get_cl_wind(weight, mask, plot=False, nside=256):

    fsky = mask.mean()
    print(fsky)
    weight[~mask] = hp.UNSEEN
    cl_wind = hp.anafast(weight, lmax=2*nside) / fsky

    el_p = 2
    el = np.arange(5000)
    is_small = el < el_p

    lmin = 10
    lmax = 200 #2*nside-1
    x = np.arange(lmin, lmax+1)
    y = cl_wind[lmin:lmax+1]
    res = curve_fit(model, x, np.log10(y), p0=[1, 1])

    cl_window = np.zeros(el.size)
    cl_window[:el_p] = cl_wind[:el_p]
    cl_window[~is_small] = 10**model(el[~is_small], *res[0])
    
    if plot:
        x_g = np.arange(el_p, 5000)
        plt.plot(cl_wind, alpha=0.2, lw=4)
        plt.plot(cl_window, alpha=0.4, lw=2, ls='-')
        #plt.plot(x_g, 10**model(x_g, *res[0]), lw=1)
        plt.xscale('log')
        plt.yscale('log')    
        
    return cl_window


def get_mixm(l1_, cl_w, l2_max=400, l2_min=0):
    
    #l2 = np.arange(np.maximum(0, l1_-50), l1_+50)    
    l2 = np.arange(l2_min, l2_max+1)
    mixm = []
    for l2_ in l2:
        
        lmin = abs(l2_ - l1_)
        lmax = l2_ + l1_
        
        sum_ = 0.0                        
        for l3_ in np.arange(lmin, lmax+1):            
            wj = wigner_3j(l1_, l2_, l3_, 0, 0, 0)
            sum_ += (2.*l3_+1.)*cl_w[l3_]*wj*wj        
            
        mixm.append((2.*l2_+1.)*sum_)
        
    return mixm


def main(l1_):
    # get window power
    nside = 256
    
    data = ft.read(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_desic_{nside}.fits')
    weight = ut.make_hp(nside, data['hpix'], data['fracgood'])
    mask = weight > 0

    cl_w = get_cl_wind(weight, mask, nside=nside)
    mixm = get_mixm(l1_, cl_w, l2_max=500)
    fpi = 0.25/np.pi # 1/4pi
    mixm = fpi*np.array(mixm)
    np.save(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/window/window_desic_{nside}_{l1_}.npy', mixm)
    print('done')


if __name__ == '__main__':

    l1 = int(sys.argv[1])
    main(l1)
