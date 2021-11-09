"""
   tools for handing pixels
   a bunch of useful functions & classes for calculating
   cosmological quantities

   (c) Mehdi Rezaie medirz90@icloud.com
   Last update: Jul 5, 2020

"""
import os
import sys
import logging
import numpy as np
import healpy as hp
import fitsio as ft
import pandas as pd

from astropy.table import Table
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from scipy.stats import (binned_statistic, spearmanr, pearsonr)

import scipy.special as scs
from scipy.constants import c as clight
from scipy import integrate






ud_grade = hp.ud_grade

# columns
maps_eboss_v7p2 = ['star_density', 'ebv', 'loghi',
                'sky_g', 'sky_r', 'sky_i', 'sky_z',
                'depth_g_minus_ebv','depth_r_minus_ebv', 
                'depth_i_minus_ebv', 'depth_z_minus_ebv', 
                'psf_g', 'psf_r', 'psf_i', 'psf_z',
                 'run', 'airmass']

maps_dr9sv3 = ['stardens', 'ebv', 'loghi',
           'psfdepth_g', 'psfdepth_r', 'psfdepth_z',
           'galdepth_g', 'galdepth_r', 'galdepth_z', 
           'psfsize_g', 'psfsize_r', 'psfsize_z', 
           'psfdepth_w1', 'psfdepth_w2']

maps_dr9 = ['EBV', 'STARDENS']\
          + [f'galdepth_{b}mag_ebv' for b in ['r', 'g', 'z']]\
          + [f'psfdepth_{b}mag_ebv' for b in ['r', 'g', 'z', 'w1', 'w2']] \
          + [f'PSFSIZE_{b}' for b in ['R', 'G', 'Z']]


# z range
z_bins = {'main':(0.8, 2.2),
         'highz':(2.2, 3.5),
         'low':(0.8, 1.5),
         'mid':(1.5, 2.2),
         'z1':(0.8, 1.3),
         'z2':(1.3, 1.7),
         'z3':(1.7, 2.2)}


def chi2_fn(residual, invcov):
    return np.dot(residual, np.dot(invcov, residual))  


def get_inv(err_tot, return_cov=False):
    
    nmocks, nbins = err_tot.shape
    hartlapf = (nmocks - 1.) / (nmocks - nbins - 2.)
    covmax = np.cov(err_tot, rowvar=False)*hartlapf
    invcov = np.linalg.inv(covmax)
    
    ret = (invcov, )
    if return_cov:
        ret += (covmax, )
    
    print(f'Hartlap factor: {hartlapf}')
    print(f'with nmocks: {nmocks} and nbins: {nbins}')
    return ret


def get_chi2pdf(err_tot):
    
    nmocks, nbins = err_tot.shape
    hartlapf = (nmocks-1. - 1.) / (nmocks-1. - nbins - 2.) # leave-one-out
    print(f'nmocks: {nmocks}, nbins: {nbins}')
    
    indices = np.arange(nmocks).tolist()
    chi2s = []
    
    for i in range(nmocks):
        
        indices_ = indices.copy()    
        indices_.pop(i)
        
        nbar_ = err_tot[i, :]
        err_ = err_tot[indices_, :]    
        
        covmax_ = np.cov(err_, rowvar=False)
        invcov_ = np.linalg.inv(covmax_*hartlapf)
        
        chi2_ = chi2_fn(nbar_, invcov_)
        chi2s.append(chi2_)       
        
    return chi2s


def to_numpy(label, features, frac, hpix):

    dtype = [('features', ('f8', features.shape[1])), 
             ('label', 'f8'),
             ('fracgood', 'f8'),
             ('hpix', 'i8')]    

    d = np.zeros(label.size, dtype=dtype)
    
    d['label'] = label
    d['fracgood'] = frac
    d['features'] = features
    d['hpix'] = hpix

    return d    


def make_hp(nside, hpix, value, fill_nan=False):
    """ 
        Create a healpix map given nside, hpix, and value
        
    """
    map_ = np.zeros(12*nside*nside)
    if fill_nan:
        map_[:] = np.nan
    
    map_[hpix] = value
    return map_

def D(z, omega0):
    """
        Growth Function
    """
    a = 1/(1+z)
    v = scs.cbrt(omega0/(1.-omega0))/a
    return a*d1(v)

def d1(v):
    """
        d1(v) = D(a)/a where D is growth function see. Einsenstein 1997
    """
    beta  = np.arccos((v+1.-np.sqrt(3.))/(v+1.+np.sqrt(3.)))
    sin75 = np.sin(75.*np.pi/180.)
    sin75 = sin75**2
    ans   = (5./3.)*(v)*(((3.**0.25)*(np.sqrt(1.+v**3.))*(scs.ellipeinc(beta,sin75)\
            -(1./(3.+np.sqrt(3.)))*scs.ellipkinc(beta,sin75)))\
            +((1.-(np.sqrt(3.)+1.)*v*v)/(v+1.+np.sqrt(3.))))
    return ans

def growthrate(z,omega0):
    """
        growth rate f = dln(D(a))/dln(a)

    """
    a = 1/(1+z)
    v = scs.cbrt(omega0/(1.-omega0))/a
    return (omega0/(((1.-omega0)*a**3)+omega0))*((2.5/d1(v))-1.5)

def invadot(a, om_m=0.3, om_L=0.0, h=.696):
    om_r = 4.165e-5*h**-2 # T0 = 2.72528K
    answ = 1/np.sqrt(om_r/(a * a) + om_m / a\
            + om_L*a*a + (1.0-om_r-om_m-om_L))
    return answ

def invaadot(a, om_m=0.3, om_L=0.0, h=.696):
    om_r = 4.165e-5*h**-2 # T0 = 2.72528K
    answ = 1/np.sqrt(om_r/(a * a) + om_m / a\
            + om_L*a*a + (1.0-om_r-om_m-om_L))
    return answ/a

class Cosmology(object):
    '''
       cosmology
       # see
       # http://www.astro.ufl.edu/~guzman/ast7939/projects/project01.html
       # or
       # https://arxiv.org/pdf/astro-ph/9905116.pdf
       # for equations, there is a typo in comoving-volume eqn
    '''
    def __init__(self, om_m=1.0, om_L=0.0, h=.696):
        self.om_m = om_m
        self.om_L = om_L
        self.h    = h
        self.om_r = 4.165e-5*h**-2 # T0 = 2.72528K
        self.tH  = 9.778/h         # Hubble time : 1/H0 Mpc --> Gyr
        self.DH  = clight*1.e-5/h       # Hubble distance : c/H0

    def age(self, z=0):
        '''
            age of universe at redshift z [default z=0] in Gyr
        '''
        az = 1 / (1+z)
        answ,_ = integrate.quad(invadot, 0, az,
                               args=(self.om_m, self.om_L, self.h))
        return answ * self.tH

    def DCMR(self, z):
        '''
            comoving distance (line of sight) in Mpc
        '''
        az = 1 / (1+z)
        answ,_ = integrate.quad(invaadot, az, 1,
                               args=(self.om_m, self.om_L, self.h))
        return answ * self.DH

    def DA(self, z):
        '''
            angular diameter distance in Mpc
        '''
        az = 1 / (1+z)
        r = self.DCMR(z)
        om_k = (1.0-self.om_r-self.om_m-self.om_L)
        if om_k != 0.0:DHabsk = self.DH/np.sqrt(np.abs(om_k))
        if om_k > 0.0:
            Sr = DHabsk * np.sinh(r/DHabsk)
        elif om_k < 0.0:
            Sr = DHabsk * np.sin(r/DHabsk)
        else:
            Sr = r
        return Sr*az

    def DL(self, z):
        '''
            luminosity distance in Mpc
        '''
        az = 1 / (1+z)
        da = self.DA(z)
        return da / (az * az)

    def CMVOL(self, z):
        '''
            comoving volume in Mpc^3
        '''
        Dm = self.DA(z) * (1+z)
        om_k = (1.0-self.om_r-self.om_m-self.om_L)
        if om_k != 0.0:DHabsk = self.DH/np.sqrt(np.abs(om_k))
        if om_k > 0.0:
            Vc = DHabsk**2 * np.sqrt(1 + (Dm/DHabsk)**2) * Dm \
                 - DHabsk**3 * np.sinh(Dm/DHabsk)
            Vc *= 4*np.pi/2.
        elif om_k < 0.0:
            Vc = DHabsk**2 * np.sqrt(1 + (Dm/DHabsk)**2) * Dm \
                 - DHabsk**3 * np.sin(Dm/DHabsk)
            Vc *= 4*np.pi/2.
        else:
            Vc = Dm**3
            Vc *= 4*np.pi/3
        return Vc

def gaulegf(a, b, n):
    """
    Gauss Legendre numerical quadrature, x and w computation 
    integrate from a to b using n evaluations of the function f(x)  
    usage: from gauleg import gaulegf         
           x,w = gaulegf( a, b, n)                                
           area = 0.0                                            
           for i in range(1,n+1): #  yes, 1..n                   
             area += w[i]*f(x[i]) 
    
    """
    x = range(n+1) # x[0] unused
    w = range(n+1) # w[0] unused
    eps = 3.0E-14
    m = (n+1)/2
    xm = 0.5*(b+a)
    xl = 0.5*(b-a)
    for i in range(1,m+1):
        z = np.cos(3.141592654*(i-0.25)/(n+0.5))
        while True:
            p1 = 1.0
            p2 = 0.0
            for j in range(1,n+1):
                p3 = p2
                p2 = p1
                p1 = ((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j

            pp = n*(z*p1-p2)/(z*z-1.0)
            z1 = z
            z = z1 - p1/pp
            if abs(z-z1) <= eps:
                break

    x[i] = xm - xl*z
    x[n+1-i] = xm + xl*z
    w[i] = 2.0*xl/((1.0-z*z)*pp*pp)
    w[n+1-i] = w[i]
    return x, w    
    
def nside2npix(nside):
    """ get npix from nside """
    return 12 * nside * nside

def npix2nside(npix):
    """ Determine nside from npix """
    return int(np.sqrt(npix / 12.0))

def nside2pixarea(nside, degrees=False):
    """ Determine pix area given nside """
    pixarea = 4 * np.pi / nside2npix(nside)
    if degrees:
        pixarea = np.rad2deg(np.rad2deg(pixarea))
    
    return pixarea

def cutphotmask(aa, bits, return_indices=False):
    print(f'{len(aa)} before imaging veto')
    
    keep = (aa['NOBS_G']>0) & (aa['NOBS_R']>0) & (aa['NOBS_Z']>0)
    for biti in bits:
        keep &= ((aa['MASKBITS'] & 2**biti)==0)
    print(f'{keep.sum()} {keep.mean()} after imaging veto')
    #print(keep)
    #return keep
    if return_indices:
        return (aa[keep], keep)
    else:
        return aa[keep]

def rolling(x, y, width=3):
    """ compute moving average given width """
    size = y.size
    assert width%2 != 0, "width must be odd"
    step = width//2

    x_ = []
    y_ = []
    for i in range(step, size-step):
        
        x_.append(np.mean(x[i-step:i+step]))
        y_.append(np.mean(y[i-step:i+step]))
    
    return np.array(x_), np.array(y_)


def split_NtoM(N, M, rank):
    """
    split N to M pieces
    
    see https://stackoverflow.com/a/26554699/9746916
    """
    chunk = N // M
    remainder = N % M
    
    if rank < remainder:
        start = rank * (chunk + 1)
        stop = start + chunk
    else:
        start = rank * chunk + remainder
        stop = start + (chunk -1)
    
    return start, stop



class SphericalKMeans(KMeans):
    """
    Class provides K-Means clustering on the surface of a sphere.
    
    
    attributes
    ----------
    centers_radec : array_like
        the center ra and dec of the clusters
    
    
    methods
    -------
    fit_radec(ra, dec, sample_weight=None)
        compute K-Means clustering 
        
    predict_radec(ra, dec)
        return the cluster index the ra and dec belong to
    
    see also
    --------
    sklearn.cluster.KMeans
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        
    """
    def __init__(self, n_clusters=40, random_state=42, **kwargs):
        """ initialize self
        
            parameters
            ----------
            n_clusters : int, optional
            the number of clusters to form as well as the number of centroids to generate.
        
            random_state : int, optional
            the seed determines random generator for centroid initialization. 
            Use an int to make the randomness deterministic
            
            kwargs: dict
            optional arguments to sklearn.cluster.Kmeans
        """
        super(SphericalKMeans, self).__init__(n_clusters=n_clusters,
                                             random_state=random_state,
                                             **kwargs)

    def fit_radec(self, ra, dec, sample_weight=None):
        """ Compute K-Means clustering
        
        parameters
        ----------
        ra : array_like
            right ascention of the examples in degrees
        
        dec : array_like
            declination of the examples in degrees
            
            
        """
        r = radec2r(ra, dec)
        self.fit(r, sample_weight=sample_weight)
        self.centers_radec = r2radec(self.cluster_centers_)

    def predict_radec(self, ra, dec):
        """ Predict the closest cluster each ra and dec belong to
        
        
        parameters
        ----------
        ra : array_like
            right ascention of the examples in degrees
            
        dec : array_like
            declination of the examples in degrees
            
        returns
        -------
        labels : array_like
            index of the cluster each ra and dec belong to
        
        """
        r = radec2r(ra, dec)
        return self.predict(r)


def r2radec(r):
    """
    Function transforms r to ra and dec
    
    parameters
    ----------
    r : array_like (N, 3)
        x, y, z coordinates in the Cartesion coordinate system
        
    returns
    -------
    ra : array_like
        right ascention in degrees
        
    dec : array_like
        declination in degrees
        
        
    see also
    --------
    radec2r 
        
    """    
    rad2deg = 180./np.pi
    dec = rad2deg*np.arcsin(r[:, 2])
    ra = rad2deg*np.arctan(r[:, 1]/r[:, 0])
    ra[r[:, 0]<0] += 180. # if x < 0, we are in the 2nd or 3rd quadrant
    return ra, dec


def radec2r(ra, dec):
    """
    Function transforms ra and dec to r
    
    x = cos(phi)sin(theta) or cos(ra)cos(dec)
    y = sin(phi)sin(theta) or sin(ra)cos(dec)
    z = cos(theta) or sin(dec)
    
    parameters
    ----------
    ra : array_like
        right ascention in degrees
    
    dec : array_like
        declination in degrees
        
    returns
    --------
    r : array_like
        Euclidean `distance` from the center

    """
    ra_rad, dec_rad = np.deg2rad(ra), np.deg2rad(dec)
    x = np.cos(dec_rad)*np.cos(ra_rad)
    y = np.cos(dec_rad)*np.sin(ra_rad)
    z = np.sin(dec_rad)
    r = np.column_stack([x, y, z])
    return r


class KMeansJackknifes:
    """
    Class constructs K-Means clusters for Jackknife resampling
    
    
    attributes
    ----------
    mask : array_like
        boolean mask represents the footprint
        
    hpix : array_like
        HEALPix pixels indices represent the footprint
        
    weight : array_like
        weight associated with each pixel
        
    radec : (array_like, array_like)
        right ascention and declination of the footprint
        
    centers : (array_like, array_like)
        ra and dec of the cluster centers
        
    masks : dict
        Jackknife masks
        
        
    methods
    -------
    build_masks(self, njack, seed=42, kmeans_kw={'n_init':1})
        build the jackknife masks
    
    
    examples
    --------
    >>> mask = hp.read_map('mask_NGC.hp512.ran.fits') > 0
    >>> jk = KMeansJackknifes(mask, mask.astype('f8'))
    >>> jk.build_masks(20)
    >>> jk.centers
    array([2, 2, 2, ..., 8, 8, 8], dtype=int32)
    >>> jk.visualize()
    >>> jk[0]            # to get mask 0
    
    """
    def __init__(self, mask, weight):
        """ initialize self
        
        parameters
        ----------
        mask : array_like
            boolean mask of the footprint
            
        weight : array_like
            weight associated with the footprint
            
        """
        
        self.nside = hp.get_nside(mask)
        assert hp.get_nside(weight)==self.nside
        
        self.mask = mask
        self.hpix = np.argwhere(mask).flatten()
        self.weight = weight[mask]
        self.radec = hpix2radec(self.nside, self.hpix)

    def build_masks(self, njack, seed=42, kmeans_kw={'n_init':1, 'n_jobs':1}):
        """ 
        function creates Jackknife masks
        
        parameters
        ----------
        njack : int
            the number of jackknife regions
            
        seed : int
            the seed for the random number generator
            
        kmeans_kw: dict
            optional parameters for SphericalKMeans
        
        
        """        
        
        np.random.seed(seed) # for KMeans centeroid initialization
        
        km = SphericalKMeans(n_clusters=njack, **kmeans_kw)
        km.fit_radec(*self.radec, sample_weight=self.weight)        
        self.centers = km.predict_radec(*self.radec)
        
        self.masks = {-1:self.mask}        
        for i in range(njack):            
            mask_i = self.mask.copy()
            mask_i[self.hpix[self.centers == i]] = False            
            self.masks[i] = mask_i        

    def visualize(self):        
        """ function plots K-Means clusters
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        
        njack = len(self.masks)-1 # drop -1 as the global mask        
        for i in range(njack): 
            mask_i = self.centers == i
            ax.scatter(shiftra(self.radec[0][mask_i]), 
                       self.radec[1][mask_i],
                       s=1, marker='o',
                       alpha=0.2, 
                       color=plt.cm.rainbow(i/njack),
                      rasterized=True) 
        ax.set(xlabel='RA [deg]', ylabel='DEC [deg]')        
        return fig, ax

    
    def __getitem__(self, item):
        """ function returns i'th Jackknife mask
        """
        return self.masks[item]



def corrmat(matrix, estimator='pearsonr'):
    '''
    The corrmatrix function.

    The function computes the correlation matrix.

    Parameters
    ----------
    matrix : 2-D Array with shape (n,m)
        2-D array of the attributes (n, m)

    estimator : string, optional
        String to determine the correlation coefficient estimator
        options are pearsonr and spearmanr

    Returns
    -------
    corr : 2-D array with shape (m, m)
        2-D array of the correlation matrix

    Examples
    --------
    >>> # example 1
    >>> x = np.random.multivariate_normal([1, -1], 
                                  [[1., 0.9], [0.9, 1.]], 
                                  size=1000)                                  
    >>> corr = corrmat(x, estimator='pearsonr')
    >>> assert np.allclose(corr, [[1., 0.9], [0.9, 1.]], rtol=1.0e-2)
    >>>
    >>> # example 2
    >>> df = pd.read_hdf('SDSS_WISE_HI_imageprop_nside512.h5', key='templates')
    >>> df.dropna(inplace=True)
    >>> corr = corrmat(df.values)
    >>> plt.imshow(corr)

    '''
    if estimator == 'pearsonr':
        festimator = pearsonr
    elif estimator == 'spearmanr':
        festimator = spearmanr
    else:
        raise ValueError(f'{estimator} is pearsonr or spearmanr')

    n_examples, n_features = matrix.shape
    
    corr = np.ones((n_features, n_features))

    for i in range(n_features):
        column_i = matrix[:,i]

        for j in range(i+1, n_features):
            corr_ij = festimator(column_i, matrix[:,j])[0]
            corr[i,j] = corr_ij  # corr matrix is symmetric
            corr[j,i] = corr_ij

    return corr


def read_dr8density(df, n2r=False, perpix=True):
    """ Funtion reads DR8 ELG Density, Colorbox selection

        credit: Pauline Zarrouk
        
        parameters
        ----------
        df : array_like
            dataframe with ELG column densities
        
        n2r : boolean
            convert from nest to ring ordering
            
        perpix : boolean
            convert the density to per pixel unit
            
            
        returns
        -------
        density : array_like
            density of ELGs
        
    """

    density = np.zeros(df.size)
    for colorcut in ['ELG200G228', 'ELG228G231',\
                 'ELG231G233', 'ELG233G234',\
                 'ELG234G236']: # 'ELG200G236'
        density += df[colorcut]

    nside = hp.get_nside(density)
    
    if perpix:
        # it's already per sq deg
        density *= df['FRACAREA']*hp.nside2pixarea(nside, degrees=True)

    if n2r:
        density = hp.reorder(density, n2r=n2r)

    return density

def steradian2sqdeg(steradians):
    """ 
    Steradians to sq. deg

    parameters
    ----------
    steradians : float
        area in steradians

    returns
    -------
    area : float
        area in steradians
        
    """
    return steradians*(180/np.pi)**2

def shiftra(ra):
    """ 
    (c) Julian Bautista Hack to shift RA for plotting
    
    
    parameters
    ----------
    ra : array_like
        right ascention in degrees
        
        
    returns
    -------
    ra' : array_like
        shifted right ascention in degrees
        
    """
    return ra-360*(ra>300)


def flux_to_mag(flux, band, ebv=None):
    """ 
    Converts SDSS fluxes to magnitudes,
    correcting for extinction optionally (EBV)

    credit: eBOSS pipeline (Ashley Ross, Julian Bautista et al.)
    
    parameters
    ----------
    flux : array_like
        SDSS fluxes
        
    band : string
        one of ugriz
        
    returns
    -------
    mag : array_like
        magnitudes corrected for the EBV
    
    """
    index_b = dict(zip(['u', 'g', 'r', 'i', 'z'], np.arange(5)))
    index_e = dict(zip(['u', 'g', 'r', 'i', 'z'], [4.239,3.303,2.285,1.698,1.263]))
    #-- coefs to convert from flux to magnitudes
    iband = index_b[band]
    ext_coeff = index_e[band]
    b   = np.array([1.4, 0.9, 1.2, 1.8, 7.4])[iband]*1.e-10
    mag = -2.5/np.log(10.)*(np.arcsinh((flux/1.e9)/(2*b)) + np.log(b))
    if ebv is not None:
        #-- extinction coefficients for SDSS u, g, r, i, and z bands
        #ext_coeff = np.array([4.239, 3.303, 2.285, 1.698, 1.263])[band]
        mag -= ext_coeff*ebv
    return mag


def radec2hpix(nside, ra, dec):
    """ 
    Function transforms RA,DEC to HEALPix index in ring ordering
    
    parameters
    ----------
    nside : int
    
    ra : array_like
        right ascention in deg
    
    dec : array_like
        declination in deg
    
    
    returns
    -------
    hpix : array_like
        HEALPix indices
    
    """
    hpix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
    return hpix

def hpix2radec(nside, hpix):
    """
    Function transforms HEALPix index (ring) to RA, DEC
    
    parameters 
    ----------
    nside : int
        
    hpix : array_like
        HEALPix indices
    
    returns
    -------
    ra : array_like
        right ascention in deg
        
    dec : array_like
        declination in deg
        
    """
    theta, phi = hp.pixelfunc.pix2ang(nside, hpix)
    return np.degrees(phi), 90-np.degrees(theta)


def select_region(ra, dec, reg):
    
    wra = (ra > 100.-dec)
    wra &= (ra < 280. +dec)
    
    if reg == 'ndecals':
        w = dec < 32.375
        w &= wra
    elif reg == 'sdecals':
        w = ~wra
        w &= dec > -30.0
    else:
        raise ValueError(f'{reg} not implemented')
    return w


def radec2regions(ra, dec, nside=256, min_dec_mzls=32.375, min_dec_decals=-30.0):
    """
    Function splits RA and DEC to DECaLS North, South, and BASS/MzLS

    parameters
    ----------
    ra : array_like [in deg]
    
    ra : array_like [in deg]
    
    nside : int
    
    min_dec_mzls : float
    
    min_dec_decals : float


    returns
    -------
    is_decaln : array_like
    
    is_decals : array_like
    
    is_mzls : array_like
    
    
    examples
    --------
    """
    theta = np.radians(90.-dec)
    phi = np.radians(ra)      
    r = hp.Rotator(coord=['C', 'G'])
    theta_g, phi_g = r(theta, phi)

    is_north  = theta_g < np.pi/2
    is_mzls   = (dec > min_dec_mzls) & is_north
    is_decaln = (~is_mzls) & is_north
    is_decals = (~is_mzls) & (~is_north) & (dec > min_dec_decals)
    
    return is_decaln, is_decals, is_mzls

   

def hpix2regions(hpix, nside=256, min_dec_mzls=32.375, min_dec_decals=-30.0):
    """
    Function splits HEALPix indices to DECaLS North, South, and BASS/MzLS

    parameters
    ----------
    hpix : array_like
    
    nside : int
    
    min_dec_mzls : float
    
    min_dec_decals : float


    returns
    -------
    is_decaln : array_like
    
    is_decals : array_like
    
    is_mzls : array_like
    
    
    examples
    --------
    >>> mask = hp.read_map('mask_elg_256.cut.fits') > 0
    >>> hpix = np.argwhere(mask).flatten()
    >>> regions = hpix2regions(hpix, 256)
    >>> for region in regions:
            ra, dec = hpix2radec(256, hpix[region])
            plt.scatter(shiftra(ra), dec)
            
    >>> plt.show()
    """
    theta, phi = hp.pixelfunc.pix2ang(nside, hpix)    
    dec = 90-np.degrees(theta)
    r = hp.Rotator(coord=['C', 'G'])
    theta_g, phi_g = r(theta, phi)

    is_north  = theta_g < np.pi/2
    is_mzls   = (dec > min_dec_mzls) & is_north
    is_decaln = (~is_mzls) & is_north
    is_decals = (~is_mzls) & (~is_north) & (dec > min_dec_decals)
    
    return is_decaln, is_decals, is_mzls

def mask2regions(mask, min_dec_mzls=32.375, min_dec_decals=-30.0):
    """    
    Function splits a binary mask into DECaLS North, South, and BASS/MzLS
    
    parameters
    ----------
    mask : array_like, boolean
    
    hpix2caps_kwargs : dict
        optional arguments for `hpix2caps'
         
         
    see also
    --------
    'hpix2regions'


    examples
    --------
    >>> mask = hp.read_map('mask_elg_256.cut.fits') > 0
    >>> regions = mask2regions(mask)
    >>> for region in regions:
            hp.mollview(region)

    """
    nside = hp.get_nside(mask)
    hpix = np.argwhere(mask).flatten()
    regions = hpix2regions(hpix, nside, 
                           min_dec_mzls=min_dec_mzls, 
                           min_dec_decals=min_dec_decals)

    ngc = np.zeros_like(mask)
    ngc[hpix[regions[0]]] = True

    sgc = np.zeros_like(mask)
    sgc[hpix[regions[1]]] = True

    bmzls = np.zeros_like(mask)
    bmzls[hpix[regions[2]]] = True

    return ngc, sgc, bmzls

def histogram_cell(cell, return_err=False, method='nmodes', bins=None, fsky=1.0, **kwargs):  
    """
    Function bins C_ell and estimates the error
    
    
    parameters
    ----------
    cell : array_like, or dict of array_like
    
    return_err : boolean
    
    method : str
        nmodes : error based on mode counting
        
        jackknife : error based on Jackknife sub-sampling
        
    bins : array_like
    
    fsky : float
        fraction of sky covered
        
    kwargs : dict
        optional arguments for `__histogram_cell`
        
        
        
    returns
    -------
    ell_bin : array_like
    
    cell_bin : array_like
    
    cell_bin_err : array_like 
        (optional)
    
    """
    if return_err:
        
        if method=='nmodes':     
            assert isinstance(cell, np.ndarray)
            ell_bin, cell_bin, cell_bin_err = __get_nmodes_cell_err(cell, bins=bins, fsky=fsky, **kwargs)                

        elif method=='jackknife':
            assert isinstance(cell, dict)
            ell_bin, cell_bin, cell_bin_err = __get_jackknife_cell_err(cell['cl_jackknifes'], bins=bins, **kwargs)        

        else:
            raise ValueError(f'{method} is not implemented, choices are nmodes and jackknifes.')

        return ell_bin, cell_bin, cell_bin_err

    else:
        
        assert isinstance(cell, np.ndarray)
        
        ell = np.arange(cell.size)
        ell_bin, cell_bin = __histogram_cell(ell, cell, bins=bins, **kwargs)

        return ell_bin, cell_bin
    
    
def __histogram_cell(ell, cell, bins=None, return_weights=False, log=True):
    """
    Function computes the histogram of the C_ell weighted by 2\ell+1  
    
    C_{ell} = \sum C_{ell} (2\ell+1) / \sum (2\ell+1)
                            
              
    parameters
    ----------
    ell : array_like 
    
    cell : array_like
    
    bins : array_like
    
    return_weights : boolean
    
    log : boolean
        logarithmic binning if bins not provided

    returns
    -------    
    ell_bin : array_like
    
    cell_bin : array_like
    
    weights_bin : array_like    
        (optional)
    
    """
    
    # set the bins, ell_min = 0 (or 1 in log)
    if bins is None:                
        if log:
            bins = np.logspace(0, np.log10(ell.max()+1), 10)
        else:
            bins = np.linspace(0, ell.max()+1, 10)
                    
    kwargs  = dict(bins=bins, statistic='sum')    
    bins_mid = 0.5*(bins[1:]+bins[:-1])
    weights = 2*ell + 1
    
    ell_weights_bin = binned_statistic(ell, weights*ell, **kwargs)[0] # first output is needed
    cell_weights_bin = binned_statistic(ell, weights*cell, **kwargs)[0] # first output is needed
    weights_bin = binned_statistic(ell, weights, **kwargs)[0]
    cell_bin = cell_weights_bin / weights_bin
    ell_bin = ell_weights_bin / weights_bin
    
    ret = (ell_bin, cell_bin, )
    
    if return_weights:
        ret += (weights_bin, )

    return ret

        
def __get_nmodes_cell_err(cell, bins=None, fsky=1.0, **kwargs):
    """
    Function computes the mode counting error estimate
    
    parameters
    ----------
    cell : array_like
    
    bins : array_like
    
    fsky : float
    
    kwargs : dict
        optional arguments for 'histogram_cell'
        
        
    returns
    -------
    ell_bin : array_like
    
    cell_bin : array_like
    
    cell_bin_err : array_like    
        
    """
    
    ell = np.arange(cell.size)    
    ell_bin, cell_bin, weight_bin = __histogram_cell(ell, cell, bins=bins, return_weights=True, **kwargs)  
    cell_bin_err = (cell_bin/weight_bin)/(np.sqrt(0.5*fsky*weight_bin))
    
    return ell_bin, cell_bin, cell_bin_err


def __get_jackknife_cell_err(cljks, bins=None, **kwargs):
    """
    Function computes jackknife C_ell measurements and get the error estimate
        
    parameters
    ----------
    cljks : dict of array_like
    
    bins : array_like
    
    kwargs : dict
        optional arguments for `histogram_cell`
        
    
    returns
    -------
    ell_bin : array_like
    
    cell_bin : array_like    
    
    cell_bin_err : array_like
    
    
    """    
    njacks = len(cljks) - 1 # -1 for the global measurement    
    ell = np.arange(cljks[0].size)
    
    ell_bin, cell_bin = __histogram_cell(ell, cljks[-1], bins=bins, **kwargs)
    
    cell_var = np.zeros(cell_bin.size)
    
    for i in range(njacks):
        
        cell_bin_i = __histogram_cell(ell, cljks[i], bins=bins, **kwargs)[1] # only need cell_bin
        delta_cell = cell_bin - cell_bin_i
        
        cell_var += delta_cell*delta_cell 
        
    cell_var *= (njacks-1.)/njacks
    cell_bin_err = np.sqrt(cell_var)
    
    return ell_bin, cell_bin, cell_bin_err


def value2hpix(nside, ra, dec, value, statistic='mean'):
    """
    Aggregates a quantity (value) onto HEALPix with nside and ring ordering
    using `statistic`
    
    parameters
    ----------
    nside : int
    
    ra : array_like
    
    dec : array_like
    
    value : array_like
    
    statistic : str
        (optional), default is 'mean', but can work with 'min', 'max', etc


    returns
    -------
    value_hp : array_like

    """
    hpix = radec2hpix(nside, ra, dec)
    npix = hp.nside2npix(nside)
    value_hp = binned_statistic(hpix, value, statistic=statistic,
                                bins=np.arange(0, npix+1, 1))[0]
    return value_hp


def hpixsum(nside, ra, dec, weights=None):
    """
    Aggregates ra and dec onto HEALPix with nside and ring ordering.

    credit: Yu Feng, Ellie Kitanidis, ImagingLSS, UC Berkeley


    parameters
    ----------
    nside: int
    
    ra: array_like
        right ascention in degree.
    dec: array_like
        declination in degree.

    returns
    -------
    weight_hp: array_like
            
    """
    hpix = radec2hpix(nside, ra, dec)
    npix = hp.nside2npix(nside)
    weight_hp = np.bincount(hpix, weights=weights, minlength=npix)
    return weight_hp


def make_overdensity(ngal, nran, mask, selection_fn=None, is_sys=False, nnbar=False):
    """
    Constructs the density contrast field, \delta


    parameters
    ----------
    ngal : array_like
        galaxy counts map in HEALPix
    nran : array_like
        random counts map in HEALPix
    mask : array_like
        footprint boolean mask in HEALPix
    selection_fn : array_like
        selection function in HEALpix
    is_sys : boolean
        whether the input 'galmap' is a systematic template.
    nnbar : boolean
        whether subtract one to make density contrast.
        
        
    returns
    -------
    delta : array_like
        density contrast field
    
    """
    assert mask.dtype=='bool', "mask must be boolean" #MR: how about mask has indices e.g., 0, 1, 3
    assert np.all(nran[mask]>1.0e-8), "'weight' must be > 0"
    
    delta = np.empty_like(ngal)
    delta[:] = np.nan    
    ngal_ = ngal.copy()
    
    if selection_fn is not None:
        assert np.all(selection_fn[mask]>1.0e-8), "'selection_mask' must be > 0"
        ngal_[mask] = ngal_[mask] / selection_fn[mask]

    if is_sys:
        sf = (ngal_[mask]*nran[mask]).sum() / nran[mask].sum()
        delta[mask] = ngal_[mask] / sf
    else:
        sf = ngal_[mask].sum()/nran[mask].sum()
        delta[mask] = ngal_[mask]/(nran[mask]*sf)
        
    if not nnbar:
        delta[mask] -= 1.0

    return delta
    

def make_sysmaps(ran, path_lenz, path_gaia, nside=256):
    """
    Creates templates for systematics
    
    
    parameters
    ----------
    ran : numpy structured array
    
    path_lenz : str
        path to HI column density (Lens et al.)
    path_gaia : str
        path to Gaia Stellar density
    nside : int
    
    
    returns
    -------
    pandas.DataFrame
    
    
    """
    from lssutils.extrn.galactic.hpmaps import Gaia, logHI
    
    maps = {'sky_g':ran['skyflux'][:,1],
            'sky_r':ran['skyflux'][:,2],
            'sky_i':ran['skyflux'][:,3],
            'sky_z':ran['skyflux'][:,4],
            'airmass':ran['airmass'],
            'ebv':ran['eb_minus_v'],
            'depth_g':ran['image_depth'][:,1],
            'depth_r':ran['image_depth'][:,2],
            'depth_i':ran['image_depth'][:,3],
            'depth_z':ran['image_depth'][:,4],
            'psf_g':ran['psf_fwhm'][:,1],
            'psf_r':ran['psf_fwhm'][:,2],
            'psf_i':ran['psf_fwhm'][:,3],
            'psf_z':ran['psf_fwhm'][:,4],
            'run':ran['run']}
    
    hpmaps = {}
    for name in maps:
        print('.', end='')
        hpmaps[name] = value2hpix(nside, ran['ra'], ran['dec'], maps[name])
    
    lenz = logHI(nside_out=nside, path=path_lenz)
    nstar = Gaia(nside_out=nside, path=path_gaia)
    hpmaps['loghi'] = lenz.loghi
    hpmaps['star_density'] = nstar.nstar
    for band in 'rgiz':
        hpmaps[f'depth_{band}_minus_ebv'] = flux_to_mag(hpmaps[f'depth_{band}'], band, ebv=hpmaps['ebv'])
    hpmaps['w1_med'] = np.ones(12*nside*nside)
    hpmaps['w1_covmed'] = np.ones(12*nside*nside)
    return pd.DataFrame(hpmaps)


def split2kfolds(data, k=5, shuffle=True, seed=42):
    """
    Splits data into k randomly chosen folds 
    for training (3x), validation (1x) and testing (1x)
       
    
    parameters
    ----------
    data : numpy structured array
    
    k : int
    
    shuffle : boolean
    
    seed : int
    
    
    returns
    -------
    kfold_data : dict
        k partitions of training, validation, and test sets
    
    """
    np.random.seed(seed)
    kfold = KFold(k, shuffle=shuffle, random_state=seed)
    index = np.arange(data.size)
    
    kfold_data = {'test':{}, 'train':{}, 'validation':{}}
    for i, (nontestID, testID) in enumerate(kfold.split(index)):
        #
        #
        foldname = 'fold'+str(i)
        validID  = np.random.choice(nontestID, size=testID.size, replace=False)
        trainID  = np.setdiff1d(nontestID, validID)
        #
        #
        kfold_data['test'][foldname]       = data[testID]
        kfold_data['train'][foldname]      = data[trainID]
        kfold_data['validation'][foldname] = data[validID]
        
    return kfold_data


def ivar2depth(ivar):
    """ change IVAR to DEPTH """
    depth = nmag2mag(5./np.sqrt(ivar))
    return depth

def nmag2mag(nmag):
    """ nano maggies to magnitude """
    return -2.5 * (np.log10(nmag) - 9.)

def mag2nmag(m):
    """ Magnitude to nano maggies """
    return 10.**(-m/2.5+9.)


def read_partialmap(filename, nside=256):    
    """ read partial systematic map """
    data   = ft.read(filename, lower=True)
    if 'ivar' in filename.lower(): # in case there was IVAR
        signal = IvarToDepth(data['signal'])
    else:
        signal = data['signal']

    output = np.empty(12*nside*nside)
    output[:] = np.nan
    output[data['pixel']] = signal
    return output


def make_clustering_catalog(mock, comp_min=0.5):
    """
    make clustering catalogs for SDSS-IV eBOSS
    
    (c) Julien Bautista 
    """
    w = ((mock['IMATCH']==1) | (mock['IMATCH']==2))
    w &= (mock['COMP_BOSS'] > comp_min)
    w &= (mock['sector_SSR'] > comp_min)

    names = ['RA', 'DEC', 'Z', 'WEIGHT_FKP', 'WEIGHT_SYSTOT', 'WEIGHT_CP']
    names += ['WEIGHT_NOZ', 'NZ', 'QSO_ID']

    mock = mock[w]

    fields = []
    for name in names:
        fields.append(mock[name])
    mock_clust = Table(fields, names=names)
    return mock_clust


def reassign(randoms, data, seed=42, comp_min=0.5):
    """
    This function re-assigns the attributes from data to randoms
    
    Parameters
    ----------
    randoms : numpy structured array for randoms
    
    data : numpy structured array for data
        
    seed : int
    
        
    Returns
    -------
    rand_clust : numpy structured array for randoms
        

    (c) Julien Bautista
    
    Updates
    --------
    March 9, 20: Z, NZ, FKP must be assigned from data
    
    Examples    
    --------
    >>> ???
    """

    rand_clust = Table()
    rand_clust['RA'] = randoms['RA']*1
    rand_clust['DEC'] = randoms['DEC']*1
    rand_clust['COMP_BOSS'] = randoms['COMP_BOSS']*1
    rand_clust['sector_SSR'] = randoms['sector_SSR']*1

    np.random.seed(seed)    
    index = np.arange(len(data))
    indices = np.random.choice(index, size=len(randoms), replace=True)
    
    fields = ['WEIGHT_NOZ', 'WEIGHT_CP', 'WEIGHT_SYSTOT', 'WEIGHT_FKP', 'Z', 'NZ'] 
    for f in fields:
        rand_clust[f] = data[f][indices]


    rand_clust['WEIGHT_SYSTOT'] *= rand_clust['COMP_BOSS']
    good = (rand_clust['COMP_BOSS']>comp_min) & (rand_clust['sector_SSR']>comp_min) 

    return rand_clust[good]


class EbossCat:
    """
    Reads SDSS-IV eBOSS catalogs
    
    
    attributes
    ----------
    kind : str
        kind of the catalog, galaxy or randoms
    columns : list of str
        names of the columns in the catalog
    
    
    methods
    -------
    tohp(nside, zmin, zmax, raw=0)
        projects objects to HEALPix with nside and ring ordering
        
    
    see also
    --------
    ???
    """    
    
    logger = logging.getLogger('EbossCat')
    
    columns = ['RA', 'DEC', 'Z', 
               'WEIGHT_FKP', 'WEIGHT_SYSTOT', 'WEIGHT_CP',
               'WEIGHT_NOZ', 'NZ', 'QSO_ID', 'IMATCH',
               'COMP_BOSS', 'sector_SSR']  
    
    __comp_min = 0.5
    
    def __init__(self, filename, kind='data', **clean_kwargs):
        """ 
        Initialize the EbossCat object 
        
        
        parameters
        ----------
        filename : str
        
        kind : str
        
        clean_kwargs : dict
            zmin : float
                minimum redshift (=0.8)
            zmax : float
                maximum redshift (=2.2)
        
        """
        assert kind in ['data', 'randoms'], "kind must be either 'data' or 'randoms'"
        self.kind  = kind
        self.__read(filename)
        self.__clean(**clean_kwargs)
        
    def __read(self, filename):
        """ 
        Read the catalog 
        
        
        parameters
        ----------
        filename : str
        
        """
        if filename.endswith('.fits'):
            self.data  = Table.read(filename)
        else:
            raise NotImplementedError(f'file {filename} not implemented')
            
        self.data_is_clean = False
    
    def __clean(self, zmin=0.8, zmax=2.2):
        """ 
        Clean data and randoms catalogs, change the `Full` to `Clustering` catalog
        
        parameters
        ----------
        zmin : float
        
        zmax : float
        
        """           
        columns = []
        for i, column in enumerate(self.columns):
            if column not in self.data.columns:
                self.logger.warning(f'column {column} not in the {self.kind} file')
            else:
                columns.append(column)
                
        self.columns = columns
        self.data  = self.data[self.columns]        
        
        self.logger.info(f'{zmin} < z < {zmax}')        
        good = (self.data['Z'] > zmin) & (self.data['Z'] < zmax)
        
        for column in ['COMP_BOSS', 'sector_SSR']:            
            if column in self.data.columns:                                
                self.logger.info(f'{column} > {self.__comp_min}')
                good &= (self.data[column] > self.__comp_min)
                
        if self.kind=='data':            
            if 'IMATCH' in self.data.columns:
                self.logger.info(f'IMATCH = 1 or 2 for {self.kind}')                
                is_eboss = (self.data['IMATCH']==1)
                is_legacy = (self.data['IMATCH']==2)                 
                good &=  is_eboss | is_legacy
                                                
        self.logger.info(f'{good.sum()} ({100*good.mean():3.1f}%) {self.kind} pass the cuts')
        self.data = self.data[good]
        self.data_is_clean = True
        
    def __prepare_weights(self, raw=0):    
        """
        prepare the weights
        
        parameters
        ----------
        raw : int
            0: raw number of objects
            1: data weighted by FKPxCPxNOZ
               randoms weighted by FKPxCOMP_BOSS
            2: data/randoms weighted by FKPxCPxNOZxSYSTOT
        
        """
        self.logger.info(f'raw: {raw}')        
        
        if raw==0:
            self.data['WEIGHT'] = 1.0
            
        elif raw==1:             
            if self.kind == 'data':
                self.data['WEIGHT'] = self.data['WEIGHT_FKP']*1.
                self.data['WEIGHT'] *= self.data['WEIGHT_CP']
                self.data['WEIGHT'] *= self.data['WEIGHT_NOZ']                
            elif self.kind == 'randoms':                
                self.data['WEIGHT'] = self.data['WEIGHT_FKP']*1.
                self.data['WEIGHT'] *= self.data['COMP_BOSS']                
            else:
                raise ValueError(f'{self.kind} not defined')
                
        elif raw==2:
            self.data['WEIGHT'] = self.data['WEIGHT_FKP']*1.
            self.data['WEIGHT'] *= self.data['WEIGHT_CP']            
            self.data['WEIGHT'] *= self.data['WEIGHT_NOZ']
            self.data['WEIGHT'] *= self.data['WEIGHT_SYSTOT']
            
        else:
            raise ValueError(f'{raw} should be 0, 1, or 2!')
            
            
    def to_hp(self, nside, zmin, zmax, raw=0):
        """
        Project to HEALPix
        
        parameters
        ----------
        nside : int
        
        zmin : float
        
        zmax : float
        
        raw : int
        
        
        returns
        -------
        nobjs : array_like
            the number of objects in HEALPix
        
        """
        assert self.data_is_clean, "`data` is not clean"
        self.__prepare_weights(raw=raw)
        
        self.logger.info(f'Projecting {self.kind}  to HEALPix with {nside}')
        good = (self.data['Z'] > zmin) & (self.data['Z'] < zmax)
        self.logger.info((f'{good.sum()} ({100*good.mean():3.1f}%)'
                          f' {self.kind} pass ({zmin:.1f} < z < {zmax:.1f})'))
        
        return hpixsum(nside, 
                       self.data['RA'][good], 
                       self.data['DEC'][good], 
                       weights=self.data['WEIGHT'][good])

    def swap(self, mappers, column='WEIGHT_SYSTOT'):
        """ 
        Swap 'column' using mappers
        
        
        parameters
        ----------
        mappers : dict
        
        column : str
        
        
        """
        # precompute three weights
        w_tot = self.data['WEIGHT_CP']*1
        w_tot *= self.data['WEIGHT_NOZ']
        w_tot *= self.data['WEIGHT_FKP'] 
        
        
        for sample, mapper in mappers.items():

            zmin, zmax = mapper[0]
            
            good = (self.data['Z'] > zmin) & (self.data['Z'] < zmax)
            w_sys = mapper[1](self.data['RA'][good], self.data['DEC'][good])  
            
            # normalize and clip extremes
            norm_factor = w_tot[good].sum() / (w_tot[good]*w_sys).sum() # normalize w_sys
            w_sys = norm_factor*w_sys    
            
            extremes = (w_sys < 0.5) | (w_sys > 2.0)
            self.logger.info(f'number of extreme w_sys (0.5< or > 2.0): {extremes.sum()}')
            w_sys = w_sys.clip(0.5, 2.0)

            self.data[column][good] = w_sys
            
            self.logger.info(f'number of {sample} objects passed {zmin}<z<{zmax} : {good.sum()}')
            self.logger.info(f'w_sys: [{w_sys.min():.2f}, {w_sys.max():.2f}]')
                                
    def to_fits(self, filename):
        assert self.data_is_clean, "`data` is not clean"
        
        if os.path.isfile(filename):
            raise RuntimeError('%s exists'%filename)
            
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            
        self.data.write(filename, overwrite=True)    
    
    def reassign_zattrs(self, source, seed=42):
        """ Reassign z-related attributes from 'source' to 'data'
        """
        assert self.kind=='randoms', "reassignment only performed for 'data'"
        self.data = reassign(self.data, source, seed=seed)
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        msg = f"catalog : {self.kind}\n"
        msg += f'# of objects : {len(self.data)}\n'
        msg += f"z : {self.data['Z'].min(), self.data['Z'].max()}\n"
        msg += f"columns : {self.columns}"
        return msg
    
    

class HEALPixDataset:
    logger = logging.getLogger('HEALPixDataset')
    
    def __init__(self, data, randoms, templates, columns):

        self.data = data
        self.randoms = randoms        
        self.features = templates[columns].values
        self.nside = hp.get_nside(self.features[:, 0])
        self.mask = np.ones(self.features.shape[0], '?')
        for i in range(self.features.shape[1]):
            self.mask &= np.isfinite(self.features[:, i])
        self.logger.info(f'{self.mask.sum()} pixels ({self.mask.mean()*100:.1f}%) have imaging')
        
    def prepare(self, nside, zmin, zmax, label='nnbar', frac_min=0, nran_exp=None):        
        assert nside == self.nside, f'template has NSIDE={self.nside}'
               
        if label=='nnbar': # used in Rezaie et al. (2020) "DECaLS DR7 Galaxies"
            return self.__prep_nnbar(nside, zmin, zmax, frac_min, nran_exp)
        elif label=='ngal': # used in Rezaie et al. (2021) "eBOSS QSOs"
            return self.__prep_ngal(nside, zmin, zmax, frac_min, nran_exp)
        elif label=='ngalw':
            return self.__prep_ngalw(nside, zmin, zmax, frac_min, nran_exp)
        else:
            raise ValueError(f'{label} must be nnbar, ngal, or ngalw')
    
    def __prep_nnbar(self, nside, zmin, zmax, frac_min, nran_exp):
        
        ngal = self.data.to_hp(nside, zmin, zmax, raw=1)        
        nran = self.randoms.to_hp(nside, zmin, zmax, raw=1)
        if nran_exp is None:
            nran_exp = np.mean(nran[nran>0])
            self.logger.info(f'using {nran_exp} as nran_exp')            
            
        frac = nran / nran_exp
        
        mask_random = (frac >  frac_min)        
        mask = mask_random & self.mask                
        self.logger.info(f'{mask.sum()} pixels ({mask.mean()*100:.1f}%) have imaging')
        
        nnbar = make_overdensity(ngal, nran, mask, nnbar=True) 
        
        return self._to_numpy(nnbar[mask], self.features[mask, :],
                             frac[mask], np.argwhere(mask).flatten())        
        
    def __prep_ngalw(self, nside, zmin, zmax, frac_min, nran_exp):
        
        ngal = self.data.to_hp(nside, zmin, zmax, raw=1)        
        nran = self.randoms.to_hp(nside, zmin, zmax, raw=1)
        if nran_exp is None:
            nran_exp = np.mean(nran[nran>0])
            self.logger.info(f'using {nran_exp} as nran_exp')            
            
        frac = nran / nran_exp        
        mask_random = (frac >  frac_min)        
        mask = mask_random & self.mask        
        self.logger.info(f'{mask.sum()} pixels ({mask.mean()*100:.1f}%) have imaging')
        
        return self._to_numpy(ngal[mask], self.features[mask, :],
                             frac[mask], np.argwhere(mask).flatten())

    def __prep_ngal(self, nside, zmin, zmax, frac_min, nran_exp):
        
        ngal = self.data.to_hp(nside, zmin, zmax, raw=0)
        ngalw = self.data.to_hp(nside, zmin, zmax, raw=1)        
        
        wratio = np.ones_like(ngal)
        good = (ngal > 0.0) & (ngalw > 0.0)
        wratio[good] = ngalw[good]/ngal[good]        
        
        nran = self.randoms.to_hp(nside, zmin, zmax, raw=1)
        if nran_exp is None:
            nran_exp = np.mean(nran[nran>0])
            self.logger.info(f'using {nran_exp} as nran_exp')            
            
        frac = nran / nran_exp
        
        mask_random = (frac >  frac_min)        
        mask = mask_random & self.mask        
        self.logger.info(f'{mask.sum()} pixels ({mask.mean()*100:.1f}%) have imaging')
        
        #wratio[mask & (~good)] = 1.0 # have randoms but no data
        
        fracw = np.zeros_like(frac)
        fracw[mask] = frac[mask] / wratio[mask]
        
        return self._to_numpy(ngal[mask], self.features[mask, :],
                             fracw[mask], np.argwhere(mask).flatten())    
    
    def _to_numpy(self, t, features, frac, hpix):
        
        dtype = [('features', ('f8', features.shape[1])), 
                 ('label', 'f8'),
                 ('fracgood', 'f8'),
                 ('hpix', 'i8')]    
        
        dataset = np.zeros(t.size, dtype=dtype)
        dataset['label'] = t
        dataset['fracgood'] = frac
        dataset['features'] = features
        dataset['hpix'] = hpix
        
        return dataset    

    
class DR9Data:
    
    features_names = ['STARDENS', 'EBV', 
                      'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 
                      'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 
                      'PSFDEPTH_W1', 'PSFDEPTH_W2',
                      'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

    caps = {'N':'isnorth',
            'S':'issouth'}
    
    targets = {'elg':'elg_dens',
              'lrg':'lrg_dens',
              'qso':'qso_dens'}
    
    fracgoods = {'elg':'elg_fracarea',
                'lrg':'lrg_fracarea',
                'qso':'qso_fracarea'}

    def __init__(self, filename):
        dt = ft.read(filename)
        
        ix_ = hp.reorder(np.arange(dt.size), n2r=True)
        self.dt = dt[ix_]  # reorder to Ring
        
    def run(self, target, region, frac_min=0.0):
        
        ngal = self.dt[self.targets[target]]
        frac = self.dt[self.fracgoods[target]]           
        mask = self.dt[self.caps[region]] 
        
        nside = hp.get_nside(ngal)
        pixarea = hp.nside2pixarea(nside, degrees=True)
        
        print('org. mask:', mask.sum())
        mask = mask & (frac > frac_min)
        print('org. m. & frac > frac_min:', mask.sum())
        
        features = []
        for feature in self.features_names:
            feature_  = self.dt[feature]
            mask &= np.isfinite(feature_)
            features.append(feature_)
        features = np.array(features).T
        print('org. m. & frac. min. & inf features', mask.sum())

        
        hpix = np.argwhere(mask).flatten()
        target = ngal * frac * pixarea
        
        return self._to_numpy(target[mask], features[mask, :], frac[mask], hpix)
        
    
    
        
    def _to_numpy(self, t, features, frac, hpix):
        
        dtype = [('features', ('f8', features.shape[1])), 
                 ('label', 'f8'),
                 ('fracgood', 'f8'),
                 ('hpix', 'i8')]    
        
        dataset = np.zeros(t.size, dtype=dtype)
        dataset['label'] = t
        dataset['fracgood'] = frac
        dataset['features'] = features
        dataset['hpix'] = hpix
        
        return dataset                 

#class SysWeight(object):
#    '''
#    Reads the systematic weights in healpix
#    Assigns them to a set of RA and DEC (both in degrees)
#
#    ex:
#        > Mapper = SysWeight('nn-weights.hp256.fits')
#        > wsys = Mapper(ra, dec)    
#    '''    
#    def __init__(self, filename, ismap=False):
#        if ismap:
#            self.wmap  = filename
#        else:
#            self.wmap  = hp.read_map(filename, verbose=False)            
#        self.nside = hp.get_nside(self.wmap)
#
#    def __call__(self, ra, dec):
#              
#        hpix = radec2hpix(self.nside, ra, dec) # HEALPix index from RA and DEC
#        w_ = self.wmap[hpix]                 # Selection mask at the pixel
#        
#        w_normed = w_ / np.median(w_)
#        w_normed = w_normed.clip(0.5, 2.0)      
#        
#        return 1./w_normed
#        
#class EnsembleWeights(SysWeight):
#    
#    def __init__(self, filename, nside, istable=False):
#        #
#        if istable:
#            wnn = filename
#        else:
#            wnn = ft.read(filename)  
#        
#        wnn_hp = np.ones(12*nside*nside)
#        wnn_hp[wnn['hpix']] = wnn['weight'].mean(axis=1)
#        
#        self.mask = np.zeros_like(wnn_hp, '?')
#        self.mask[wnn['hpix']] = True
#        
#        super(EnsembleWeights, self).__init__(wnn_hp, ismap=True)       
#        
class SysWeight(object):
    '''
    Reads the systematic weights in healpix
    Assigns them to a set of RA and DEC (both in degrees)

    ex:
        > Mapper = SysWeight('nn-weights.hp256.fits')
        > wsys = Mapper(ra, dec)    
    '''
    logger = logging.getLogger('SysWeight')
  
    def __init__(self, filename, ismap=False, fix=True, clip=True):
        if ismap:
            self.wmap  = filename
        else:
            self.wmap  = hp.read_map(filename, verbose=False)
          
        self.nside = hp.get_nside(self.wmap)
        self.fix = fix
        self.clip = clip

    def __call__(self, ra, dec):
      
      
        hpix = radec2hpix(self.nside, ra, dec) # HEALPix index from RA and DEC
        wsys = self.wmap[hpix]                 # Selection mask at the pixel
      
        if self.fix:
          
            NaNs = np.isnan(wsys)                  # check if there is any NaNs
            self.logger.info(f'# NaNs : {NaNs.sum()}')

            NaNs |= (wsys <= 0.0)                  # negative weights
            if self.clip:
                self.logger.info('< or > 2x')
                assert abs(np.median(wsys)-1.0) < 0.1, 'You should not clip the selection function that is not normalized'
                NaNs |= (wsys < 0.5) 
                NaNs |= (wsys > 2.0)
              
            self.logger.info(f'# NaNs or lt 0: {NaNs.sum()}')


            if NaNs.sum() !=0:

                nan_wsys = np.argwhere(NaNs).flatten()
                nan_hpix = hpix[nan_wsys]

                # use the average of the neighbors
                self.logger.info(f'# NaNs (before) : {len(nan_hpix)}')
                neighbors = hp.get_all_neighbours(self.nside, nan_hpix) 
                wsys[nan_wsys] = np.nanmean(self.wmap[neighbors], axis=0)

                # 
                NaNs =  (np.isnan(wsys) | (wsys <= 0.0))
                NNaNs = NaNs.sum()
                self.logger.info(f'# NaNs (after)  : {NNaNs}')

                # set weight to 1 if not available
                if NNaNs != 0:
                    self.logger.info(f'set {NNaNs} pixels to 1.0 (neighbors did not help)')
                    wsys[NaNs] = 1.0

          
        assert np.all(wsys > 0.0),f'{(wsys <= 0.0).sum()} weights <= 0.0!' 
        return 1./wsys # Systematic weight = 1 / Selection mask
    
    
class NNWeight(SysWeight):
  
    def __init__(self, filename, nside, fix=True, clip=False, aggregate='mean', ix=0):
      
        wnn = ft.read(filename)        
        wnn_hp = np.zeros(12*nside*nside)
        
        if aggregate == 'mean':
            wnn_hp[wnn['hpix']] = wnn['weight'].mean(axis=1)
        elif aggregate == 'median':
            wnn_hp[wnn['hpix']] = np.median(wnn['weight'], axis=1)
        else:
            print(f'use {ix}')
            wnn_hp[wnn['hpix']] = wnn['weight'][:, ix]
            #raise ValueError(f'{aggregate} not implemented')
            
        self.mask = np.zeros_like(wnn_hp, '?')
        self.mask[wnn['hpix']] = True
      
        super(NNWeight, self).__init__(wnn_hp, ismap=True, fix=fix, clip=clip)    
    
def extract_keys_dr9(mapi):
    band = mapi.split('/')[-1].split('_')[3]
    sysn = mapi.split('/')[-1].split('_')[6]
    oper = mapi.split('/')[-1].split('_')[-1].split('.')[0]
    return '_'.join((sysn, band, oper)) 

def extract_keys_dr8(mapi):
    band = mapi.split('/')[-1].split('_')[4]
    sysn = mapi.split('/')[-1].split('_')[7]
    oper = mapi.split('/')[-1].split('_')[-1].split('.')[0]
    return '_'.join((sysn, band, oper))

def jointemplates():
    #--- append the CCD based templates to the TS based ones
    ts = pd.read_hdf('/home/mehdi/data/templates/pixweight-dr8-0.32.0.h5')
    ccd = pd.read_hdf('/home/mehdi/data/templates/dr8_combined256.h5')

    # rename the second to last ebv
    combined = pd.concat([ccd[cols_dr8], ts[cols_dr8_ts]], sort=False, axis=1)
    colnames = combined.columns.values
    colnames[-2] = 'ebv2'
    combined.columns = colnames
    return combined


def uniform_sphere(RAlim, DEClim, size=1):
    """Draw a uniform sample on a sphere
    Parameters
    ----------
    RAlim : tuple
        select Right Ascension between RAlim[0] and RAlim[1]
        units are degrees
    DEClim : tuple
        select Declination between DEClim[0] and DEClim[1]
    size : int (optional)
        the size of the random arrays to return (default = 1)
    Returns
    -------
    RA, DEC : ndarray
        the random sample on the sphere within the given limits.
        arrays have shape equal to size.
    """
    zlim = np.sin(np.pi * np.asarray(DEClim) / 180.)

    z = zlim[0] + (zlim[1] - zlim[0]) * np.random.random(size)
    DEC = (180. / np.pi) * np.arcsin(z)
    RA = RAlim[0] + (RAlim[1] - RAlim[0]) * np.random.random(size)

    return RA, DEC

class Readfits(object):
    #
    def __init__(self, paths, extract_keys=extract_keys_dr9, res_out=256):
        files = paths
        print('total number of files : %d'%len(files))
        print('file-0 : %s %s'%(files[0], extract_keys(files[0])))
        self.files        = files
        self.extract_keys = extract_keys
        self.nside_out = res_out
        
    def run(self, add_foreground=False, mkwytemp=None):
        
        self._run()
        if add_foreground:
            self._add_foreground(mkwytemp)
        #
        # replace inf with nan
        self.metadata.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.ready2write  = True
        
    def save(self, path2output, name='metadata'):
        if os.path.isfile(path2output):
            print('%s exists'%path2output)            
        self.metadata.to_hdf(path2output, name, mode='w', format='fixed')
        
    def _run(self):
        metadata = {}
        for file_i in self.files:    
            name_i  = self.extract_keys(file_i)    
            print('working on ... %s'%name_i)
            if 'ivar' in name_i:name_i = name_i.replace('ivar', 'depth')
            if name_i in metadata.keys():
                raise RuntimeError('%s already in metadata'%name_i)
            metadata[name_i] = read_partialmap(file_i, self.nside_out)            
            
        self.metadata = pd.DataFrame(metadata)
        
    def _add_foreground(self, mkwytemp=None):
        # FIXME: 'mkwytemp' will point to the templates 
        from lssutils.extrn.galactic import hpmaps
        # 
        Gaia    = hpmaps.gaia_dr2(nside_out=self.nside_out)
        self.metadata['nstar'] = Gaia.gaia
        
        EBV     = hpmaps.sfd98(nside_out=self.nside_out)
        self.metadata['ebv']   = EBV.ebv
        
        logNHI  = hpmaps.logHI(nside_out=self.nside_out)
        self.metadata['loghi'] = logNHI.loghi            
        
    def make_plot(self, path2fig):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        nmaps = self.metadata.shape[1]
        ncols = 3
        nrows = nmaps // ncols
        if np.mod(nmaps, ncols)!=0:
            nrows += 1
            
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, 
                               figsize=(4*ncols, 3*nrows))
        
        ax=ax.flatten()
        for i,name in enumerate(self.metadata.columns):
            plt.sca(ax[i])
            good = np.isfinite(self.metadata[name])
            vmin, vmax = np.percentile(self.metadata[name][good], [2.5, 97.5])
            hp.mollview(self.metadata[name], hold=True, title=name, rot=-89, min=vmin, max=vmax)
            
        plt.savefig(path2fig, bbox_inches='tight')
        

def split_jackknife_strip(hpix, weight, njack=20):
    f = weight.sum() // njack
    hpix_L = []
    hpix_l = []
    frac_L = []
    frac    = 0
    w_L = []
    w_l = []
    remainder = None
    
    for i in range(hpix.size):
        frac += weight[i]
        hpix_l.append(hpix[i])
        w_l.append(weight[i])

        if frac >= f:
            hpix_L.append(hpix_l)
            frac_L.append(frac)
            w_L.append(w_l)
            frac    = 0
            w_l     = []
            hpix_l = []
        elif (i == hpix.size-1):
            if (frac > 0.9*f):
                hpix_L.append(hpix_l)
                frac_L.append(frac)
                w_L.append(w_l)
            else:
                print('the remaining chunk is less than 90% complete!')
                remainder = [hpix_l, w_l, frac]
            
    return hpix_L, w_L, remainder      
        

def split_continuous(hpix, weight, label, features, njack=20):
    '''
        split_jackknife(hpix, weight, label, features, njack=20)
        split healpix-format data into k equi-area regions
        hpix: healpix index shape = (N,)
        weight: weight associated to each hpix
        label: label associated to each hpix
        features: features associate to each pixel shape=(N,M)
    '''
    f = weight.sum() // njack
    hpix_L = []
    hpix_l = []
    frac_L = []
    frac    = 0
    label_L = []
    label_l = []
    features_L = []
    features_l = []
    w_L = []
    w_l = []
    #
    #
    for i in range(hpix.size):
        frac += weight[i]
        hpix_l.append(hpix[i])
        label_l.append(label[i])
        w_l.append(weight[i])
        features_l.append(features[i])
        #
        #
        if frac >= f:
            hpix_L.append(hpix_l)
            frac_L.append(frac)
            label_L.append(label_l)
            w_L.append(w_l)
            features_L.append(features_l)
            frac    = 0
            features_l  = []
            w_l     = []
            hpix_l = []
            label_l = []
        elif (i == hpix.size-1) and (frac > 0.9*f):
            hpix_L.append(hpix_l)
            frac_L.append(frac)
            label_L.append(label_l)
            w_L.append(w_l)
            features_L.append(features_l)
    return hpix_L, w_L, label_L, features_L #, frac_L

def concatenate(A, ID):
    # combine A[i] regions for i in ID
    AA = [A[i] for i in ID]
    return np.concatenate(AA)

def combine(hpix, fracgood, label, features, DTYPE, IDS):
    # uses concatenate(A,ID) to combine different attributes
    size = np.sum([len(hpix[i]) for i in IDS])
    zeros = np.zeros(size, dtype=DTYPE)
    zeros['hpix']     = concatenate(hpix, IDS)
    zeros['fracgood'] = concatenate(fracgood, IDS)
    zeros['features'] = concatenate(features, IDS)
    zeros['label']    = concatenate(label, IDS)
    return zeros


def split2KfoldsSpatially(data, k=5, shuffle=True, random_seed=42):
    '''
        split data into k contiguous regions
        for training, validation and testing
    '''
    P, W, L, F = split_continuous(data['hpix'],data['fracgood'],
                                data['label'], data['features'],
                                 njack=k)
    DTYPE = data.dtype
    np.random.seed(random_seed)
    kfold = KFold(k, shuffle=shuffle, random_state=random_seed)
    index = np.arange(k)
    kfold_data = {'test':{}, 'train':{}, 'validation':{}}
    arrs = P, W, L, F, DTYPE
    for i, (nontestID, testID) in enumerate(kfold.split(index)):
        foldname = 'fold'+str(i)
        validID  = np.random.choice(nontestID, size=testID.size, replace=False)
        trainID  = np.setdiff1d(nontestID, validID)
        kfold_data['test'][foldname]       = combine(*arrs, testID)
        kfold_data['train'][foldname]      = combine(*arrs, trainID)
        kfold_data['validation'][foldname] = combine(*arrs, validID)
    return kfold_data

class DR8templates:
    
    logger = logging.getLogger('DR8templates')
    
    def __init__(self, inputFile='/home/mehdi/data/pixweight-dr8-0.31.1.fits'):    
        self.logger.info(f'read {inputFile}')
        self.templates = ft.read(inputFile, lower=True)
    
    def run(self, list_maps):
        
        # http://legacysurvey.org/dr8/files/#random-catalogs
        FluxToMag = lambda flux: -2.5 * (np.log10(5/np.sqrt(flux)) - 9.)

        # http://legacysurvey.org/dr8/catalogs/#galactic-extinction-coefficients
        ext = dict(g=3.214, r=2.165, z=1.211)


        
        self.maps = []
        self.list_maps = list_maps
        
        for map_i in self.list_maps:
            
            self.logger.info(f'read {map_i}')
            hpmap_i = self.templates[map_i]
            
            #--- fix depth
            if 'depth' in map_i:                
                self.logger.info(f'change {map_i} units')
                _,band = map_i.split('_')                
                hpmap_i = FluxToMag(hpmap_i)
                
                if band in 'rgz':
                    self.logger.info(f'apply extinction on {band}')
                    hpmap_i -= ext[band]*self.templates['ebv']
                
            #--- rotate
            self.maps.append(hp.reorder(hpmap_i, n2r=True))   
            
    def plot(self, show=True):
        
        import matplotlib.pyplot as plt
        
        nrows = len(self.maps)//2
        if len(self.maps)%2 != 0:nrows += 1
            
        fig, ax = plt.subplots(ncols=2, nrows=nrows, figsize=(8, 3*nrows))
        ax = ax.flatten()

        for i, map_i in enumerate(self.maps):
            fig.sca(ax[i])
            hp.mollview(map_i, title=self.list_maps[i], hold=True, rot=-89)
            
        if show:plt.show()
    
    def to_hdf(self, name,
              key='templates'):
        df = pd.DataFrame(np.array(self.maps).T, columns=self.list_maps)
        df.to_hdf(name, key=key)
    
        
def hd5_2_fits(myfit, cols, fitname=None, hpmask=None, hpfrac=None, fitnamekfold=None, res=256, k=5, 
              logger=None):
        
    from lssutils.utils import split2Kfolds
    
    for output_i in [fitname, hpmask, hpfrac, fitnamekfold]:
        if output_i is not None:
            if os.path.isfile(output_i):raise RuntimeError('%s exists'%output_i)
    #
    hpix    = myfit.index.values
    label    = (myfit.ngal / (myfit.nran * (myfit.ngal.sum()/myfit.nran.sum()))).values
    fracgood = (myfit.nran / myfit.nran.mean()).values
    features = myfit[cols].values

    outdata = np.zeros(features.shape[0], 
                       dtype=[('label', 'f8'),
                              ('hpix', 'i8'), 
                              ('features',('f8', features.shape[1])),
                              ('fracgood','f8')])
    outdata['label']    = label
    outdata['hpix']    = hpix
    outdata['features'] = features
    outdata['fracgood'] = fracgood    

    
    if fitname is not None:
        ft.write(fitname, outdata, clobber=True)
        if logger is not None:
            logger.info('wrote %s'%fitname)

    if hpmask is not None:
        mask = np.zeros(12*res*res, '?')
        mask[hpix] = True
        hp.write_map(hpmask, mask, overwrite=True, fits_IDL=False)
        if logger is not None:
            logger.info('wrote %s'%hpmask)

    if hpfrac is not None:
        frac = np.zeros(12*res*res)
        frac[hpix] = fracgood
        hp.write_map(hpfrac, frac, overwrite=True, fits_IDL=False)
        if logger is not None:
            logger.info('wrote %s'%hpfrac)  
    
    if fitnamekfold is not None:
        outdata_kfold = split2Kfolds(outdata, k=k)
        np.save(fitnamekfold, outdata_kfold)
        if logger is not None:
            logger.info('wrote %s'%fitnamekfold)  
        




class DesiCatalog:

    logger = logging.getLogger('DesiCatalog')

    def __init__(self, filename, bool_mask):
        self.data = ft.read(filename)
        self.bool = ft.read(bool_mask)['bool_index']
        self.data = self.data[self.bool]


    def swap(self, zcuts, slices, clip=False):

        self.z_rsd = self.data['Z_COSMO'] + self.data['DZ_RSD']
        self.wsys = np.ones_like(self.z_rsd)

        for slice_i in slices:
            
            assert slice_i in zcuts.keys(), '%s not available'%slice_i

            my_zcut = zcuts[slice_i][0]
            my_mask = (self.data['Z'] >= my_zcut[0])\
                    & (self.data['Z'] <= my_zcut[1])
            
            mapper = zcuts[slice_i][1]
            self.wmap_data = mapper(self.data['RA'][my_mask], self.data['DEC'][my_mask])
            
            self.logger.info(f'{slice_i}, {self.wmap_data.min()}, {self.wmap_data.max()}')            
            if clip:self.wmap_data = self.wmap_data.clip(0.5, 2.0)
            #
            assert np.all(self.wmap_data > 0.0),'the weights are zeros!'
            self.wsys[my_mask] = self.wmap_data            
            self.logger.info('number of objs w zcut {} : {}'.format(my_zcut, my_mask.sum()))
    
    def export_wsys(self, data_name_out):
        systot = Table([self.wsys], names=['wsys']) 
        systot.write(data_name_out, format='fits')

class RegressionCatalog:
    
    logger = logging.getLogger('SystematicsPrepare')
    
    def __init__(self, 
                 data, 
                random,
                dataframe):
        
        self.data = data
        self.random = random
        self.dataframe = dataframe
        self.columns = self.dataframe.columns        
        self.logger.info(f'available columns : {self.columns}')

        
    def __call__(self, slices, zcuts, output_dir, 
                 nside=512, cap='NGC', efficient=True, columns=None):
        
        if columns is None:
            columns = self.columns
        
        if not os.path.exists(output_dir):            
            os.makedirs(output_dir)
            self.logger.info(f'created {output_dir}')
        
        
        for i, key_i in enumerate(slices):

            if key_i not in slices:
                 raise RuntimeError(f'{key_i} not in {slices}')

            self.logger.info('split based on {}'.format(zcuts[key_i]))  

            # --- prepare the names for the output files
            if efficient:
                #
                # ---- not required for regression
                hpcat     = None # output_dir + f'/galmap_{cap}_{key_i}_{nside}.hp.fits'
                hpmask    = None # output_dir + f'/mask_{cap}_{key_i}_{nside}.hp.fits'
                fracgood  = None # output_dir + f'/frac_{cap}_{key_i}_{nside}.hp.fits'
                fitname   = None # output_dir + f'/ngal_features_{cap}_{key_i}_{nside}.fits'    
            else:
                hpcat = output_dir + f'galmap_{cap}_{key_i}_{nside}.hp.fits'
                hpmask = output_dir + f'mask_{cap}_{key_i}_{nside}.hp.fits'
                fracgood = output_dir + f'frac_{cap}_{key_i}_{nside}.hp.fits'
                fitname = output_dir + f'ngal_features_{cap}_{key_i}_{nside}.fits'    
                
            fitkfold = output_dir + f'ngal_features_{cap}_{key_i}_{nside}.5r.npy'

            # cut data
            self.data.cutz(zcuts[key_i])
            self.data.tohp(nside)
            if hpcat is not None:self.data.writehp(hpcat)    
            
            # cut randoms
            zlim_ran = [2.2, 3.5] if key_i=='zhigh' else [0.8, 2.2] # randoms z cuts
            self.random.cutz(zlim_ran)
            self.random.tohp(nside)

            # --- append the galaxy and random density
            # remove NaN pixels
            dataframe_i = self.dataframe.copy()
            dataframe_i['ngal'] = self.data.hpmap
            dataframe_i['nran'] = self.random.hpmap    
            dataframe_i['nran'][self.random.hpmap == 0] = np.nan

            dataframe_i.replace([np.inf, -np.inf], 
                                value=np.nan, 
                                inplace=True) # replace inf
            
            
            dataframe_i.dropna(inplace=True)
            self.logger.info('df shape : {}'.format(dataframe_i.shape))
            self.logger.info('columns  : {}'.format(columns))

            # --- write 
            hd5_2_fits(dataframe_i, 
                       columns, 
                       fitname, 
                       hpmask, 
                       fracgood, 
                       fitkfold,
                       res=nside, 
                       k=5,
                       logger=self.logger) 
            

class EbossCatalogOld:
    
    logger = logging.getLogger('EbossCatalog')
    
    def __init__(self, filename, kind='galaxy', **kwargs):
        self.kind  = kind
        self.data  = Table.read(filename)
        
        self.select(**kwargs)
    
    def select(self, compmin=0.5, zmin=0.8, zmax=2.2):
        ''' `Full` to `Clustering` Catalog
        '''
        self.logger.info(f'compmin : {compmin}')
        self.logger.info(f'zmin:{zmin}, zmax:{zmax}')
        self.compmin = compmin
        #-- apply cuts on galaxy or randoms
        if self.kind == 'galaxy':            
            
            # galaxy            
            wd = (self.data['Z'] >= zmin) & (self.data['Z'] <= zmax)
            if 'IMATCH' in self.data.columns:
                wd &= (self.data['IMATCH']==1) | (self.data['IMATCH']==2)
            if 'COMP_BOSS' in self.data.columns:
                wd &= self.data['COMP_BOSS'] > compmin
            if 'sector_SSR' in self.data.columns:
                wd &= self.data['sector_SSR'] > compmin
                
            self.logger.info(f'{wd.sum()} galaxies pass the cuts')
            self.logger.info(f'% of galaxies after cut {np.mean(wd):0.2f}')
            self.data = self.data[wd]
            
        elif self.kind == 'random':
            
            # random
            wr  = (self.data['Z'] >= zmin) & (self.data['Z'] <= zmax)
            if 'COMP_BOSS' in self.data.columns:
                wr &= self.data['COMP_BOSS'] > compmin
            if 'sector_SSR' in self.data.columns:
                wr &= self.data['sector_SSR'] > compmin
                
            self.logger.info(f'{wr.sum()} randoms pass the cuts')
            self.logger.info(f'% of randoms after cut {np.mean(wr):0.2f}')        
            self.data = self.data[wr]
            
    
    def cutz(self, zlim):        
        #datat = self.data.copy()        
        zmin, zmax = zlim
        self.logger.info(f'Grab a slice with {zlim}')        
        myz   = (self.data['Z']>= zmin) & (self.data['Z']<= zmax)
        self.logger.info(f'# of data that pass this cut {myz.sum()}')
        self.cdata = self.data[myz]
        
    def prepare_weight(self, raw=True):
        self.logger.info(f'raw: {raw}')
        
        if not hasattr(self, 'cdata'):
            self.logger.info('cdata not found')
            self.cdata = self.data
            
        if raw:            
            if self.kind == 'galaxy':
                self.weight = self.cdata['WEIGHT_CP']*self.cdata['WEIGHT_FKP']*self.cdata['WEIGHT_NOZ']
            elif self.kind == 'random':
                self.weight = self.cdata['COMP_BOSS']*self.cdata['WEIGHT_FKP']
            else:
                raise ValueError(f'{self.kind} not defined')
        else:
            self.weight = self.cdata['WEIGHT_CP']*self.cdata['WEIGHT_FKP']*self.cdata['WEIGHT_NOZ']
            self.weight *= self.cdata['WEIGHT_SYSTOT']
    
    def reassign(self, source, seed=None):
        return reassignment(self.data, source, seed=seed)
        
    def tohp(self, nside, raw=True):
        self.logger.info(f'Projecting to HEALPIX as {self.kind} with {nside}')
        
        if not hasattr(self, 'cdata'):
            self.logger.info('cdata not found')
            self.cdata = self.data
            
        self.prepare_weight(raw=raw) # update the weights
        
        self.hpmap = hpixsum(nside, self.cdata['RA'], self.cdata['DEC'], value=self.weight)

    def swap(self, zcuts, slices, colname='WEIGHT_SYSTOT', clip=False):
        self.orgcol = self.data[colname].copy()
        for slice_i in slices:
            assert slice_i in zcuts.keys(), '%s not available'%slice_i
            #

            my_zcut   = zcuts[slice_i][0]
            my_mask   = (self.data['Z'] >= my_zcut[0])\
                      & (self.data['Z'] <= my_zcut[1])
            
            mapper    = zcuts[slice_i][1]
            self.wmap_data = mapper(self.data['RA'][my_mask], self.data['DEC'][my_mask])
            
            self.logger.info(f'slice: {slice_i}, wsysmin: {self.wmap_data.min():.2f}, wsysmax: {self.wmap_data.max():.2f}')
            self.data[colname][my_mask] = self.wmap_data            
            self.logger.info('number of objs w zcut {} : {}'.format(my_zcut, my_mask.sum()))
        
        
    def writehp(self, filename, overwrite=True):
        if os.path.isfile(filename):
            print('%s already exists'%filename, end=' ')
            if not overwrite:
                raise RuntimeWarning('please change the filename!')
            else:
                print('going to rewrite....')
        hp.write_map(filename, self.hpmap, overwrite=True, fits_IDL=False)    
        
        
    def to_fits(self, filename):
        if os.path.isfile(filename):
            raise RuntimeError('%s exists'%filename)
            
        w = np.ones(self.data['RA'].size, '?')
        if 'IMATCH' in self.data.columns:
            w &= ((self.data['IMATCH']==1) | (self.data['IMATCH']==2))
            
        if 'COMP_BOSS' in self.data.columns:
            w &= (self.data['COMP_BOSS'] > 0.5)
            
        if 'sector_SSR' in self.data.columns:
            w &= (self.data['sector_SSR'] > 0.5)
            
        self.logger.info(f'total w : {np.mean(w)}')
        #ft.write(filename, self.data)     
        self.data = self.data[w]
        
        names = ['RA', 'DEC', 'Z', 'WEIGHT_FKP', 'WEIGHT_SYSTOT', 'WEIGHT_CP']
        names += ['WEIGHT_NOZ', 'NZ', 'QSO_ID']
        
        columns = []
        for name in names:
            if name in self.data.columns:
                columns.append(name)
        
        self.data.keep_columns(columns)
        self.data.write(filename)
    
    def make_plots(self, 
                   zcuts, 
                   filename="wsystot_test.pdf", 
                   zlim=[0.8, 3.6],
                   slices=['low', 'high', 'zhigh']):
        
        import matplotlib.backends.backend_pdf
        pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
        self.plot_nzratio(zlim)
        pdf.savefig(1, bbox_inches='tight')
        self.plot_wsys(zcuts, slices=slices)
        pdf.savefig(2, bbox_inches='tight')
        pdf.close()
        
    def plot_wsys(self, zcuts, slices=['low', 'high', 'zhigh']):
        
        import matplotlib.pyplot as plt
        
        ncols=len(slices)
        fig, ax = plt.subplots(ncols=ncols, figsize=(6*ncols, 4), 
                               sharey=True)
        fig.subplots_adjust(wspace=0.05)
        #ax= ax.flatten() # only one row, does not need this!
        if ncols==1:
            ax = [ax]

        kw = dict(vmax=1.5, vmin=0.5, cmap=plt.cm.seismic, marker='H', rasterized=True)
        
        for i,cut in enumerate(slices):
            
            zlim = zcuts[cut][0]
            mask = (self.data['Z']<= zlim[1]) & (self.data['Z']>= zlim[0])
            mapi = ax[i].scatter(shiftra(self.data['RA'][mask]), self.data['DEC'][mask], 10,
                        c=self.data['WEIGHT_SYSTOT'][mask], **kw)
            
            ax[i].set(title='{0}<z<{1}'.format(*zlim), xlabel='RA [deg]')
            if i==0:ax[i].set(ylabel='DEC [deg]')

        cax = plt.axes([0.92, 0.2, 0.01, 0.6])
        fig.colorbar(mapi, cax=cax, label=r'$w_{sys}$', 
                     shrink=0.7, ticks=[0.5, 1.0, 1.5], extend='both')
        
    def plot_nzratio(self, zlim=[0.8, 3.6]):
        
        import matplotlib.pyplot as plt
        
        kw = dict(bins=np.linspace(*zlim))
        
        w_cpfkpnoz= self.data['WEIGHT_CP']*self.data['WEIGHT_FKP']*self.data['WEIGHT_NOZ']
        y0, x  = np.histogram(self.data['Z'], weights=w_cpfkpnoz, **kw)
        y,  x  = np.histogram(self.data['Z'], weights=self.orgcol*w_cpfkpnoz, **kw)
        y1, x1 = np.histogram(self.data['Z'], weights=self.data['WEIGHT_SYSTOT']*w_cpfkpnoz, **kw)

        fig, ax = plt.subplots(figsize=(6,4))

        ax.step(x[:-1], y1/y,  color='r', where='pre', label='New/Old')
        ax.step(x[:-1], y1/y0, color='k', ls='--', where='pre', label='New/NoWei.')
        ax.axhline(1, color='k', ls=':')
        ax.legend()
        ax.set(ylabel=r'$N_{i}/N_{j}$', xlabel='z')


        

#
#  old codes
#

# def histedges_equalN(x, nbin=10, kind='size', weight=None):
#     '''
#         https://stackoverflow.com/questions/39418380/
#         histogram-with-equal-number-of-points-in-each-bin
#         (c) farenorth
#     '''
#     if kind == 'size':
#         npt = len(x)
#         xp  = np.interp(np.linspace(0, npt, nbin + 1),
#                      np.arange(npt),
#                      np.sort(x))
#     elif kind == 'area':
#         raise RuntimeError('FIX this routine for a repetitave x')
#         npt1  = len(x)-1
#         sumw = np.sum(weight) / nbin
#         i    = 0
#         wst  = 0.0
#         xp   = [x.min()]  # lowest bin is the minimum
#         #
#         #
#         datat        = np.zeros(x.size, dtype=np.dtype([('x', 'f8'), ('w', 'f8'), ('rid', 'i8')]))
#         datat['x']   = x
#         datat['w']   = weight
#         datat['rid'] = np.random.choice(np.arange(x.size), size=x.size, replace=False)
#         datas  = np.sort(datat, order=['x', 'rid'])
#         xs, ws = datas['x'], datas['w'] #zip(*sorted(zip(x, weight)))
#         for wsi in ws:
#             wst += wsi
#             i   += 1
#             if (wst > sumw) or (i == npt1):
#                 xp.append(xs[i])
#                 wst = 0.0
#         xp = np.array(xp)
#     return xp


# def clerr_jack(delta, mask, weight, njack=20, lmax=512):
#     '''

#     '''
#     npix = delta.size
#     hpix = np.argwhere(mask).flatten()
#     dummy = np.ones(mask.sum())
#     hpixl, wl, deltal,_ = split_jackknife(hpix, weight[mask],
#                                           delta[mask], dummy, njack=njack)
#     print('# of jackknifes %d, input : %d'%(len(hpixl), njack))
#     cljks = {}
#     # get the cl of the jackknife mask
#     wlt = wl.copy()
#     hpixt   = hpixl.copy()
#     wlt.pop(0)
#     hpixt.pop(0)
#     wlc = np.concatenate(wlt)
#     hpixc  = np.concatenate(hpixt)
#     maski  = np.zeros(npix, '?')
#     maski[hpixc] = True
#     map_i  = hp.ma(maski.astype('f8'))
#     map_i.mask = np.logical_not(maski)
#     clmaskj = hp.anafast(map_i.filled(), lmax=lmax)
#     sfj = ((2*np.arange(clmaskj.size)+1)*clmaskj).sum()/(4.*np.pi)

#     for i in range(njack):
#         hpixt   = hpixl.copy()
#         wlt     = wl.copy()
#         deltalt = deltal.copy()
#         #
#         hpixt.pop(i)
#         wlt.pop(i)
#         deltalt.pop(i)
#         #
#         hpixc  = np.concatenate(hpixt)
#         wlc    = np.concatenate(wlt)
#         deltac = np.concatenate(deltalt)
#         #
#         maski  = np.zeros(npix, '?')
#         deltai = np.zeros(npix)
#         wlci   = np.zeros(npix)
#         #
#         maski[hpixc]   = True
#         deltai[hpixc]  = deltac
#         wlci[hpixc]    = wlc
#         #
#         map_i       = hp.ma(deltai * wlci)
#         map_i.mask  = np.logical_not(maski)
#         cljks[i]    = hp.anafast(map_i.filled(), lmax=lmax)/sfj
#     #
#     hpixt   = hpixl.copy()
#     wlt     = wl.copy()
#     deltalt = deltal.copy()
#     #
#     hpixc  = np.concatenate(hpixt)
#     wlc    = np.concatenate(wlt)
#     deltac = np.concatenate(deltalt)
#     #
#     maski  = np.zeros(npix, '?')
#     deltai = np.zeros(npix)
#     wlci   = np.zeros(npix)
#     #
#     maski[hpixc]   = True
#     deltai[hpixc]  = deltac
#     wlci[hpixc]    = wlc
#     #
#     map_i      = hp.ma(maski.astype('f8'))
#     map_i.mask = np.logical_not(maski)
#     clmask = hp.anafast(map_i.filled(), lmax=lmax)
#     sf = ((2*np.arange(clmask.size)+1)*clmask).sum()/(4.*np.pi)

#     map_i       = hp.ma(deltai * wlci)
#     map_i.mask  = np.logical_not(maski)
#     cljks[-1]   = hp.anafast(map_i.filled(), lmax=lmax)/sf   # entire footprint
#     #
#     clvar = np.zeros(len(cljks[-1]))
#     for i in range(njack):
#         clvar += (cljks[-1] - cljks[i])*(cljks[-1] - cljks[i])
#     clvar *= (njack-1)/njack
#     return dict(clerr=np.sqrt(clvar), cljks=cljks, clmaskj=clmaskj, clmask=clmask, sf=sf, sfj=sfj)


# def split_jackknife_new(hpix, weight, njack=20):
#     '''
#         split_jackknife(hpix, weight, label, features, njack=20)
#         split healpix-format data into k equi-area regions
#         hpix: healpix index shape = (N,)
#         weight: weight associated to each hpix
#         label: label associated to each hpix
#         features: features associate to each pixel shape=(N,M)
#     '''
#     f = weight.sum() // njack
#     hpix_L = []
#     hpix_l = []
#     frac_L = []
#     frac    = 0
#     w_L = []
#     w_l = []
#     #
#     #
#     for i in range(hpix.size):
#         frac += weight[i]
#         hpix_l.append(hpix[i])
#         w_l.append(weight[i])
#         #
#         #
#         if frac >= f:
#             hpix_L.append(hpix_l)
#             frac_L.append(frac)
#             w_L.append(w_l)
#             frac    = 0
#             w_l     = []
#             hpix_l = []
#         elif (i == hpix.size-1) and (frac > 0.9*f):
#             hpix_L.append(hpix_l)
#             frac_L.append(frac)
#             w_L.append(w_l)
#     return hpix_L, w_L #, frac_L











# def read_split_write(path2file, path2output, k, random=True):
#     '''
#     read path2file, splits the data either randomly or ra-dec
#     then writes the data onto path2output
#     '''
#     DATA  = ft.read(path2file)
#     if random:
#         datakfolds = split2Kfolds(DATA, k=k)
#     else:
#         datakfolds = split2KfoldsSpatially(DATA, k=k)
#     np.save(path2output, datakfolds)




# def write(address, fname, data, fmt='txt'):
#     if not os.path.exists(address):
#         os.makedirs(address)
#     if address[-1] != '/':
#         address += '/'
#     if fmt == 'txt':
#         ouname = address+fname+'.dat'
#         np.savetxt(ouname, data)
#     elif fmt == 'npy':
#         ouname = address+fname
#         np.save(ouname, data)

# --- cosmology


# class camb_pk(object):

#     #
#     def __init__(self, h=0.675, omc=.268, omb=0.048, omk=0.0, num_massive_neutrinos=1,
#            mnu=0.06, nnu=3.046, YHe=None, meffsterile=0, standard_neutrino_neff=3.046,
#            TCMB=2.7255, tau=None, ns=0.95, As=2e-9):
#         self.kwargs = dict(H0=h*100, ombh2=omb*h**2, omch2=omc*h**2, omk=omk,
#                           num_massive_neutrinos=num_massive_neutrinos,
#                            mnu=mnu, nnu=nnu, YHe=YHe, meffsterile=meffsterile,
#                           standard_neutrino_neff=standard_neutrino_neff,
#                            TCMB=TCMB, tau=tau)
#         self.pars = camb.CAMBparams()
#         self.pars.set_cosmology(**self.kwargs)
#         self.pars.InitPower.set_params(ns=ns, As=As)

#     def get_pk(self, z, kmax=.4, npoints=200):
#         h = self.kwargs['H0']/100
#         self.pars.set_matter_power(redshifts=[z], kmax=kmax)
#         self.pars.NonLinear = model.NonLinear_none
#         results = camb.get_results(self.pars)
#         s8 = np.array(results.get_sigma8())
#         print("s8 :", s8)
#         # for nonlinear uncomment this, see http://camb.readthedocs.io/en/latest/CAMBdemo.html
#         #pars.NonLinear = model.NonLinear_both
#         #results = camb.get_results(pars)
#         #results.calc_power_spectra(pars)
#         #
#         kh_nonlin,_, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=kmax, npoints=npoints)
#         return kh_nonlin, np.ravel(pk_nonlin)

#     def get_plk(self, z, kmax=.4, npoints=200, poles=[0,2,4], bias=1.0):
#         k, pk = self.get_pk(z, kmax=kmax, npoints=npoints)
#         h = self.kwargs['H0']/100
#         omega0 = self.kwargs['ombh2'] / h**2
#         beta = (1.0 / bias) * (growthrate(z, omega0))
#         pks = []
#         for pole in poles:
#             rsd_factor = rsd(pole, beta=beta)
#             pks.append(rsd_factor * bias**2 * pk)
#         return k, np.column_stack(pks)


# def rsd(l, ngauss=50, beta=.6):
#     x, wx = scs.roots_legendre(ngauss)
#     px    = scs.legendre(l)(x)
#     rsd_int = 0.0
#     for i in range(ngauss):
#         rsd_int += wx[i] * px[i] * ((1.0 + beta * x[i]*x[i])**2)
#     rsd_int *= (l + 0.5)
#     return rsd_int


# def comvol(bins_edge, fsky=1, omega_c=.3075, hubble_param=.696):
#     """
#         calculate the comoving volume for redshift bins
#     """
#     universe = cosmology(omega_c, 1.-omega_c, h=hubble_param)
#     vols = []
#     for z in bins_edge:
#         vol_i = universe.CMVOL(z) # get the comoving vol. @ redshift z
#         vols.append(vol_i)
#     # find the volume in each shell and multiply by footprint area
#     vols  = np.array(vols) * fsky
#     vols  = np.diff(vols) * universe.h**3            # volume in unit (Mpc/h)^3
#     return vols

# def nzhist(z, fsky, cosmology, bins=None, binw=0.01, weight=None):
#     if bins is None:
#         bins = np.arange(0.99*z.min(), 1.01*z.max(), binw)
#     Nz, zedge = np.histogram(z, bins=bins, weights=weight)
#     #zcenter = 0.5*(zedge[:-1]+zedge[1:])
#     vol_hmpc3 = comvol(zedge, fsky=fsky, omega_c=cosmology['Om0'], hubble_param=cosmology['H0']/100.)
#     return zedge, Nz/vol_hmpc3




# #
# """
#     a modified version of  ImagingLSS
#     https://github.com/desihub/imaginglss/blob/master/imaginglss/analysis/tycho_veto.py

#     veto objects based on a star catalogue.
#     The tycho vetos are based on the email discussion at:
#     Date: June 18, 2015 at 3:44:09 PM PDT
#     To: decam-data@desi.lbl.gov
#     Subject: decam-data Digest, Vol 12, Issue 29
#     These objects takes a decals object and calculates the
#     center and rejection radius for the catalogue in degrees.
#     Note : The convention for veto flags is True for 'reject',
#     False for 'preserve'.

#     apply_tycho takes the galaxy catalog and appends a Tychoveto column
#     the code works fine for ELG and LRGs. For other galaxy type, you need to adjust it!
# """

# def BOSS_DR9(tycho):
#     bmag = tycho['bmag']
#     # BOSS DR9-11
#     b = bmag.clip(6, 11.5)
#     R = (0.0802 * b ** 2 - 1.86 * b + 11.625) / 60. #
#     return R

# def DECAM_LRG(tycho):
#     vtmag = tycho['vtmag']
#     R = 10 ** (3.5 - 0.15 * vtmag) / 3600.
#     return R

# DECAM_ELG = DECAM_LRG

# def DECAM_QSO(tycho):
#     vtmag = tycho['vtmag']
#     # David Schlegel recommends not applying a bright star mask
#     return vtmag - vtmag

# def DECAM_BGS(tycho):
#     vtmag = tycho['vtmag']
#     R = 10 ** (2.2 - 0.15 * vtmag) / 3600.
#     return R

# def radec2pos(ra, dec):
#     """ converting ra dec to position on a unit sphere.
#         ra, dec are in degrees.
#     """
#     pos = np.empty(len(ra), dtype=('f8', 3))
#     ra = ra * (np.pi / 180)
#     dec = dec * (np.pi / 180)
#     pos[:, 2] = np.sin(dec)
#     pos[:, 0] = np.cos(dec) * np.sin(ra)
#     pos[:, 1] = np.cos(dec) * np.cos(ra)
#     return pos

# def tycho(filename):
#     """
#     read the Tycho-2 catalog and prepare it for the mag-radius relation
#     """
#     dataf = ft.FITS(filename, lower=True)
#     data = dataf[1].read()
#     tycho = np.empty(len(data),
#         dtype=[
#             ('ra', 'f8'),
#             ('dec', 'f8'),
#             ('vtmag', 'f8'),
#             ('vmag', 'f8'),
#             ('bmag', 'f8'),
#             ('btmag', 'f8'),
#             ('varflag', 'i8'),
#             ])
#     tycho['ra'] = data['ra']
#     tycho['dec'] = data['dec']
#     tycho['vtmag'] = data['mag_vt']
#     tycho['btmag'] = data['mag_bt']
#     vt = tycho['vtmag']
#     bt = tycho['btmag']
#     b = vt - 0.09 * (bt - vt)
#     v = b - 0.85 * (bt - vt)
#     tycho['vmag']=v
#     tycho['bmag']=b
#     dataf.close()
#     return tycho


# def txts_read(filename):
#     obj = np.loadtxt(filename)
#     typeobj = np.dtype([
#               ('RA','f4'), ('DEC','f4'), ('COMPETENESS','f4'),
#               ('rflux','f4'), ('rnoise','f4'), ('gflux','f4'), ('gnoise','f4'),
#               ('zflux','f4'), ('znoise','f4'), ('W1flux','f4'), ('W1noise','f4'),
#               ('W2flux','f4'), ('W2noise','f4')
#               ])
#     nobj = obj[:,0].size
#     data = np.zeros(nobj, dtype=typeobj)
#     data['RA'][:] = obj[:,0]
#     data['DEC'][:] = obj[:,1]
#     data['COMPETENESS'][:] = obj[:,2]
#     data['rflux'][:] = obj[:,3]
#     data['rnoise'][:] = obj[:,4]
#     data['gflux'][:] = obj[:,5]
#     data['gnoise'][:] = obj[:,6]
#     data['zflux'][:] = obj[:,7]
#     data['znoise'][:] = obj[:,8]
#     data['W1flux'][:] = obj[:,9]
#     data['W1noise'][:] = obj[:,10]
#     data['W2flux'][:] = obj[:,11]
#     data['W2noise'][:] = obj[:,12]
#     #datas = np.sort(data, order=['RA'])
#     return data

# def veto(coord, center, R):
#     """
#         Returns a veto mask for coord. any coordinate within R of center
#         is vet.
#         Parameters
#         ----------
#         coord : (RA, DEC)
#         center : (RA, DEC)
#         R     : degrees
#         Returns
#         -------
#         Vetomask : True for veto, False for keep.
#     """
#     #from sklearn.neighbors import KDTree
#     pos_stars = radec2pos(center[0], center[1])
#     R = 2 * np.sin(np.radians(R) * 0.5)
#     pos_obj = radec2pos(coord[0], coord[1])
#     tree = KDTree(pos_obj)
#     vetoflag = ~np.zeros(len(pos_obj), dtype='?')
#     arg = tree.query_radius(pos_stars, r=R)
#     arg = np.concatenate(arg)
#     vetoflag[arg] = False
#     return vetoflag



# def apply_tycho(objgal, galtype='LRG',dirt='/global/cscratch1/sd/mehdi/tycho2.fits'):
#     # reading tycho star catalogs
#     tychostar = tycho(dirt)
#     #
#     # mag-radius relation
#     #
#     if galtype == 'LRG' or galtype == 'ELG':    # so far the mag-radius relation is the same for LRG and ELG
#         radii = DECAM_LRG(tychostar)
#     else:
#         sys.exit("Check the apply_tycho function for your galaxy type")
#     #
#     #
#     # coordinates of Tycho-2 stars
#     center = (tychostar['ra'], tychostar['dec'])
#     #
#     #
#     # coordinates of objects (galaxies)
#     coord = (objgal['ra'], objgal['dec'])
#     #
#     #
#     # a 0.0 / 1.0 array (1.0: means the object is contaminated by a Tycho-2 star, so 0.0s are good)
#     tychomask = (~veto(coord, center, radii)).astype('f4')
#     objgal = rfn.append_fields(objgal, ['tychoveto'], data=[tychomask], dtypes=tychomask.dtype, usemask=False)
#     return objgal

# def getcf(d):
#     # cut input maps based on PCC
#     from scipy.stats import pearsonr
#     # lbl = ['ebv', 'nstar'] + [''.join((s, b)) for s in ['depth', 'seeing', 'airmass', 'skymag', 'exptime'] for b in 'rgz']
#     cflist = []
#     indices = []
#     for i in range(d['train']['fold0']['features'].shape[1]):
#         for j in range(5):
#             fold = ''.join(['fold', str(j)])
#             cf = pearsonr(d['train'][fold]['label'], d['train'][fold]['features'][:,i])[0]
#             if np.abs(cf) >= 0.02:
#                 #print('{:s} : sys_i: {} : cf : {:.4f}'.format(fold, lbl[i], cf))
#                 indices.append(i)
#                 cflist.append(cf)
#     if len(indices) > 0:
#         indices = np.unique(np.array(indices))
#         return indices
#     else:
#         print('no significant features')
#         return None
#     cf = []
#     indices = []
#     for i in range(features.shape[1]):
#         cf.append(pearsonr(label, features[:,i]))
#         if np.abs(cf) > 0.0
# def change_coord(m, coord):
#     """ Change coordinates of a HEALPIX map
#     (c) dPol
#     https://stackoverflow.com/questions/44443498/
#     how-to-convert-and-save-healpy-map-to-different-coordinate-system

#     Parameters
#     ----------
#     m : map or array of maps
#       map(s) to be rotated
#     coord : sequence of two character
#       First character is the coordinate system of m, second character
#       is the coordinate system of the output map. As in HEALPIX, allowed
#       coordinate systems are 'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)

#     Example
#     -------
#     The following rotate m from galactic to equatorial coordinates.
#     Notice that m can contain both temperature and polarization.
#     >>>> change_coord(m, ['G', 'C']
#     """
#     # Basic HEALPix parameters
#     npix = m.shape[-1]
#     nside = hp.npix2nside(npix)
#     ang = hp.pix2ang(nside, np.arange(npix))

#     # Select the coordinate transformation
#     rot = hp.Rotator(coord=reversed(coord))

#     # Convert the coordinates
#     new_ang = rot(*ang)
#     new_pix = hp.ang2pix(nside, *new_ang)

#     return m[..., new_pix]
