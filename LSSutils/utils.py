"""
   tools for handing pixels
   a bunch of useful functions & classes for calculating
   cosmological quantities

   (c) Mehdi Rezaie medirz90@icloud.com
   Last update: Jul 5, 2020

"""
import os
import sys
import numpy as np
import healpy as hp

from sklearn.cluster import KMeans
from scipy.stats import (binned_statistic, spearmanr, pearsonr)





import numpy.lib.recfunctions as rfn

import fitsio as ft

import pandas as pd

import scipy.special as scs
from scipy import integrate
from scipy.constants import c as clight


from sklearn.model_selection import KFold
from sklearn.neighbors import KDTree



__all__ = ['overdensity', 'hpixsum', 'radec2hpix', 'hpix2radec',
           'radec2r', 'r2radec', 'make_jackknifes']



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

    def build_masks(self, njack, seed=42, kmeans_kw={'n_init':1}):
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
                       color=plt.cm.jet(i/njack)) 
        ax.set(xlabel='RA [deg]', ylabel='DEC [deg]')        
        return ax

    
    def __getitem__(self, item):
        """ function returns i'th Jackknife mask
        """
        return self.masks[item]



def corrmatrix(matrix, estimator='pearsonr'):
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
    >>> corr = corrmatrix(x, estimator='pearsonr')
    >>> assert np.allclose(corr, [[1., 0.9], [0.9, 1.]], rtol=1.0e-2)
    >>>
    >>> # example 2
    >>> df = pd.read_hdf('SDSS_WISE_HI_imageprop_nside512.h5', key='templates')
    >>> df.dropna(inplace=True)
    >>> corr = corrmatrix(df.values)
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


def dr8density(df, n2r=False, perpix=True):
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
    ''' Converts SDSS fluxes to magnitudes,
    correcting for extinction optionally (EBV)

    credit: eBOSS pipeline (Ashley Ross, Julian Bautista et al.)
    '''
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

def hpix2regions(hpix, 
                 nside=256, 
                 min_dec_mzls=32.375,
                 min_dec_decals=-30.0):
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

def mask2regions(mask,
                 min_dec_mzls=32.375,
                 min_dec_decals=-30.0):
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

## UP TO HERE

def histogram(el, cel, bins=None, return_wt=False):
    '''
        bin the C_ell measurements
        
        
        args:
            el
            cel
            bins
            
        returns:
            el mid
            cel mid
            wt weights
        
    '''
    if bins is None:
        bins = np.logspace(0, np.log10(el.max()+1), 10)
    kw  = dict(bins=bins, statistic='sum')
    bins_mid  = 0.5*(bins[1:]+bins[:-1])
    a2lp1 = 2*el + 1
    clwt = binned_statistic(el, a2lp1*cel, **kw)[0] # want the first value only
    wt = binned_statistic(el, a2lp1, **kw)[0]
    #print(clwt, wt)
    if return_wt:
        return bins_mid, clwt/wt, wt
    else:
        return bins_mid, clwt/wt

def modecounting_err(el, cel, bins=None, fsky=1.0):
    '''
        get the mode counting error estimate
    '''
    bins_mid, cl_wt, wt = histogram(el, cel, bins=bins, return_wt=True)

    return bins_mid, (cl_wt/wt)/(np.sqrt(0.5*fsky*wt))

def histogram_jac(cljks, bins=None):
    '''
        Bin jackknife C_ell measurements and get the error estimate
    '''
    njacks = len(cljks) - 1 # -1 for the global measurement

    el = np.arange(cljks[0].size)
    cbljks = []
    for i in range(njacks):
        elb, clb = histogram(el, cljks[i], bins=bins)
        cbljks.append(clb)

    elb, clm = histogram(el, cljks[-1], bins=bins)
    clvar = np.zeros(clm.size)
    for i in range(njacks):
        clvar += (clm - cbljks[i])*(clm - cbljks[i])
    clvar *= (njacks-1)/njacks
    return elb, np.sqrt(clvar)

def hpixmean(nside, ra, dec, value, statistic='mean'):
    '''
        project a quantity (value) onto RA-DEC, and then healpix
        with a given nside
        default is 'mean', but can work with 'min', 'max', etc
    '''
    hpix = radec2hpix(nside, ra, dec)
    nmax = 12*nside*nside
    result = binned_statistic(hpix, value, statistic=statistic,
                                 bins=nmax, range=(0, nmax))[0]
    return result

def hpixsum(nside, ra, dec, weights=None):
    '''
        make a HEALPix map of point sources ra and dec
        default ordering is RING

        credit: Yu Feng, Ellie Kitanidis
        
        
        args:
            nside: int
            ra: array (N), right ascention in degree
            dec: array (N), declination in degree
            
        returns:
            w: array (12*NSIDE*NSIDE)
            
    '''
    pix  = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
    npix = hp.nside2npix(nside)
    w    = np.bincount(pix, weights=weights, minlength=npix)
    return w

def overdensity(galmap, 
                weight, 
                mask,
                selection_fn=None, 
                is_sys=False, 
                nnbar=False):
    """
        constructs a density contrast
        
        args
            galmap: galaxy counts map in HEALPix
            weight: random counts map in HEALPix
            mask: boolean, footprint mask in HEALPix
            selection_fn: selection function in HEALpix
            is_sys: boolean, whether the input 'galmap' is a systematic template
            nnbar: boolean, whether subtract one to make density contrast
    
    """
    assert (weight[mask]>1.0e-8).sum() > 0, "'weight' must be > 0"
    
    delta = np.zeros_like(galmap, dtype=galmap.dtype)*np.nan
    if selection_fn is not None:
        assert (selection_fn[mask]>1.0e-8).sum() > 0, "'selection_mask' must be > 0"
        galmap = galmap / selection_fn

    if is_sys:
        sf = (galmap[mask]*weight[mask]).sum() / weight[mask].sum()
        delta[mask] = galmap[mask] / sf
    else:
        sf = galmap[mask].sum()/weight[mask].sum()
        delta[mask] = galmap[mask]/(weight[mask]*sf)
        
    if not nnbar:
        delta[mask] -= 1.0

    return delta
    

def make_symaps(ran, path_lenz, path_gaia, nside=256):
    from LSSutils.extrn.GalacticForegrounds.hpmaps import NStarSDSS, logHI
    #ran = ft.read('/Volumes/TimeMachine/data/eboss/sysmaps/eBOSSrandoms.ran.fits', lower=True)
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
        hpmaps[name] = hpixmean(nside, ran['ra'], ran['dec'], maps[name])
    
    lenz = logHI(nside=nside, name=path_lenz)
    nstar = NStarSDSS(nside_out=nside, name=path_gaia)
    hpmaps['loghi'] = lenz.loghi
    hpmaps['star_density'] = nstar.nstar
    hpmaps['depth_g_minus_ebv'] = flux_to_mag(hpmaps['depth_g'], 'g', ebv=hpmaps['ebv'])
    hpmaps['w1_med'] = np.ones(12*nside*nside)
    hpmaps['w1_covmed'] = np.ones(12*nside*nside)
    return pd.DataFrame(hpmaps)



def split_mask(mask_in, mask_ngc, mask_sgc, nside=256):
    mask = hp.read_map(mask_in, verbose=False).astype('bool')
    mngc, msgc = split2caps(mask, nside=nside)

    hp.write_map(mask_ngc, mngc, fits_IDL=False, dtype='float64')
    hp.write_map(mask_sgc, msgc, fits_IDL=False, dtype='float64')
    print('done')

def split2caps(mask, coord='C', nside=256):
    if coord != 'C':raise RuntimeError('check coord')
    r = hp.Rotator(coord=['C', 'G'])
    theta, phi = hp.pix2ang(256, np.arange(12*256*256))
    theta_g, phi_g = r(theta, phi) # C to G
    ngc = theta_g < np.pi/2
    sgc = ~ngc
    mngc = mask & ngc
    msgc = mask & sgc
    return mngc, msgc

def histedges_equalN(x, nbin=10, kind='size', weight=None):
    '''
        https://stackoverflow.com/questions/39418380/
        histogram-with-equal-number-of-points-in-each-bin
        (c) farenorth
    '''
    if kind == 'size':
        npt = len(x)
        xp  = np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))
    elif kind == 'area':
        sys.exit('FIX this routine for a repetitave x')
        npt1  = len(x)-1
        sumw = np.sum(weight) / nbin
        i    = 0
        wst  = 0.0
        xp   = [x.min()]  # lowest bin is the minimum
        #
        #
        datat        = np.zeros(x.size, dtype=np.dtype([('x', 'f8'), ('w', 'f8'), ('rid', 'i8')]))
        datat['x']   = x
        datat['w']   = weight
        datat['rid'] = np.random.choice(np.arange(x.size), size=x.size, replace=False)
        datas  = np.sort(datat, order=['x', 'rid'])
        xs, ws = datas['x'], datas['w'] #zip(*sorted(zip(x, weight)))
        for wsi in ws:
            wst += wsi
            i   += 1
            if (wst > sumw) or (i == npt1):
                xp.append(xs[i])
                wst = 0.0
        xp = np.array(xp)
    return xp









def clerr_jack(delta, mask, weight, njack=20, lmax=512):
    '''

    '''
    npix = delta.size
    hpix = np.argwhere(mask).flatten()
    dummy = np.ones(mask.sum())
    hpixl, wl, deltal,_ = split_jackknife(hpix, weight[mask],
                                          delta[mask], dummy, njack=njack)
    print('# of jackknifes %d, input : %d'%(len(hpixl), njack))
    cljks = {}
    # get the cl of the jackknife mask
    wlt = wl.copy()
    hpixt   = hpixl.copy()
    wlt.pop(0)
    hpixt.pop(0)
    wlc = np.concatenate(wlt)
    hpixc  = np.concatenate(hpixt)
    maski  = np.zeros(npix, '?')
    maski[hpixc] = True
    map_i  = hp.ma(maski.astype('f8'))
    map_i.mask = np.logical_not(maski)
    clmaskj = hp.anafast(map_i.filled(), lmax=lmax)
    sfj = ((2*np.arange(clmaskj.size)+1)*clmaskj).sum()/(4.*np.pi)

    for i in range(njack):
        hpixt   = hpixl.copy()
        wlt     = wl.copy()
        deltalt = deltal.copy()
        #
        hpixt.pop(i)
        wlt.pop(i)
        deltalt.pop(i)
        #
        hpixc  = np.concatenate(hpixt)
        wlc    = np.concatenate(wlt)
        deltac = np.concatenate(deltalt)
        #
        maski  = np.zeros(npix, '?')
        deltai = np.zeros(npix)
        wlci   = np.zeros(npix)
        #
        maski[hpixc]   = True
        deltai[hpixc]  = deltac
        wlci[hpixc]    = wlc
        #
        map_i       = hp.ma(deltai * wlci)
        map_i.mask  = np.logical_not(maski)
        cljks[i]    = hp.anafast(map_i.filled(), lmax=lmax)/sfj
    #
    hpixt   = hpixl.copy()
    wlt     = wl.copy()
    deltalt = deltal.copy()
    #
    hpixc  = np.concatenate(hpixt)
    wlc    = np.concatenate(wlt)
    deltac = np.concatenate(deltalt)
    #
    maski  = np.zeros(npix, '?')
    deltai = np.zeros(npix)
    wlci   = np.zeros(npix)
    #
    maski[hpixc]   = True
    deltai[hpixc]  = deltac
    wlci[hpixc]    = wlc
    #
    map_i      = hp.ma(maski.astype('f8'))
    map_i.mask = np.logical_not(maski)
    clmask = hp.anafast(map_i.filled(), lmax=lmax)
    sf = ((2*np.arange(clmask.size)+1)*clmask).sum()/(4.*np.pi)

    map_i       = hp.ma(deltai * wlci)
    map_i.mask  = np.logical_not(maski)
    cljks[-1]   = hp.anafast(map_i.filled(), lmax=lmax)/sf   # entire footprint
    #
    clvar = np.zeros(len(cljks[-1]))
    for i in range(njack):
        clvar += (cljks[-1] - cljks[i])*(cljks[-1] - cljks[i])
    clvar *= (njack-1)/njack
    return dict(clerr=np.sqrt(clvar), cljks=cljks, clmaskj=clmaskj, clmask=clmask, sf=sf, sfj=sfj)


def split_jackknife_new(hpix, weight, njack=20):
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
    w_L = []
    w_l = []
    #
    #
    for i in range(hpix.size):
        frac += weight[i]
        hpix_l.append(hpix[i])
        w_l.append(weight[i])
        #
        #
        if frac >= f:
            hpix_L.append(hpix_l)
            frac_L.append(frac)
            w_L.append(w_l)
            frac    = 0
            w_l     = []
            hpix_l = []
        elif (i == hpix.size-1) and (frac > 0.9*f):
            hpix_L.append(hpix_l)
            frac_L.append(frac)
            w_L.append(w_l)
    return hpix_L, w_L #, frac_L




def split_jackknife(hpix, weight, label, features, njack=20):
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


def split2KfoldsSpatially(data, k=5, shuffle=True, random_seed=123):
    '''
        split data into k contiguous regions
        for training, validation and testing
    '''
    P, W, L, F = split_jackknife(data['hpix'],data['fracgood'],
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




def split2Kfolds(data, k=5, shuffle=True, random_seed=123):
    '''
        split data into k randomly chosen regions
        for training, validation and testing
    '''
    np.random.seed(random_seed)
    kfold = KFold(k, shuffle=shuffle, random_state=random_seed)
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

def read_split_write(path2file, path2output, k, random=True):
    '''
    read path2file, splits the data either randomly or ra-dec
    then writes the data onto path2output
    '''
    DATA  = ft.read(path2file)
    if random:
        datakfolds = split2Kfolds(DATA, k=k)
    else:
        datakfolds = split2KfoldsSpatially(DATA, k=k)
    np.save(path2output, datakfolds)




def write(address, fname, data, fmt='txt'):
    if not os.path.exists(address):
        os.makedirs(address)
    if address[-1] != '/':
        address += '/'
    if fmt == 'txt':
        ouname = address+fname+'.dat'
        np.savetxt(ouname, data)
    elif fmt == 'npy':
        ouname = address+fname
        np.save(ouname, data)


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


class camb_pk(object):

    #
    def __init__(self, h=0.675, omc=.268, omb=0.048, omk=0.0, num_massive_neutrinos=1,
           mnu=0.06, nnu=3.046, YHe=None, meffsterile=0, standard_neutrino_neff=3.046,
           TCMB=2.7255, tau=None, ns=0.95, As=2e-9):
        self.kwargs = dict(H0=h*100, ombh2=omb*h**2, omch2=omc*h**2, omk=omk,
                          num_massive_neutrinos=num_massive_neutrinos,
                           mnu=mnu, nnu=nnu, YHe=YHe, meffsterile=meffsterile,
                          standard_neutrino_neff=standard_neutrino_neff,
                           TCMB=TCMB, tau=tau)
        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(**self.kwargs)
        self.pars.InitPower.set_params(ns=ns, As=As)

    def get_pk(self, z, kmax=.4, npoints=200):
        h = self.kwargs['H0']/100
        self.pars.set_matter_power(redshifts=[z], kmax=kmax)
        self.pars.NonLinear = model.NonLinear_none
        results = camb.get_results(self.pars)
        s8 = np.array(results.get_sigma8())
        print("s8 :", s8)
        # for nonlinear uncomment this, see http://camb.readthedocs.io/en/latest/CAMBdemo.html
        #pars.NonLinear = model.NonLinear_both
        #results = camb.get_results(pars)
        #results.calc_power_spectra(pars)
        #
        kh_nonlin,_, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=kmax, npoints=npoints)
        return kh_nonlin, np.ravel(pk_nonlin)

    def get_plk(self, z, kmax=.4, npoints=200, poles=[0,2,4], bias=1.0):
        k, pk = self.get_pk(z, kmax=kmax, npoints=npoints)
        h = self.kwargs['H0']/100
        omega0 = self.kwargs['ombh2'] / h**2
        beta = (1.0 / bias) * (growthrate(z, omega0))
        pks = []
        for pole in poles:
            rsd_factor = rsd(pole, beta=beta)
            pks.append(rsd_factor * bias**2 * pk)
        return k, np.column_stack(pks)


def rsd(l, ngauss=50, beta=.6):
    x, wx = scs.roots_legendre(ngauss)
    px    = scs.legendre(l)(x)
    rsd_int = 0.0
    for i in range(ngauss):
        rsd_int += wx[i] * px[i] * ((1.0 + beta * x[i]*x[i])**2)
    rsd_int *= (l + 0.5)
    return rsd_int

class cosmology(object):
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

def comvol(bins_edge, fsky=1, omega_c=.3075, hubble_param=.696):
    """
        calculate the comoving volume for redshift bins
    """
    universe = cosmology(omega_c, 1.-omega_c, h=hubble_param)
    vols = []
    for z in bins_edge:
        vol_i = universe.CMVOL(z) # get the comoving vol. @ redshift z
        vols.append(vol_i)
    # find the volume in each shell and multiply by footprint area
    vols  = np.array(vols) * fsky
    vols  = np.diff(vols) * universe.h**3            # volume in unit (Mpc/h)^3
    return vols

def nzhist(z, fsky, cosmology, bins=None, binw=0.01, weight=None):
    if bins is None:
        bins = np.arange(0.99*z.min(), 1.01*z.max(), binw)
    Nz, zedge = np.histogram(z, bins=bins, weights=weight)
    #zcenter = 0.5*(zedge[:-1]+zedge[1:])
    vol_hmpc3 = comvol(zedge, fsky=fsky, omega_c=cosmology['Om0'], hubble_param=cosmology['H0']/100.)
    return zedge, Nz/vol_hmpc3




#
"""
    a modified version of  ImagingLSS
    https://github.com/desihub/imaginglss/blob/master/imaginglss/analysis/tycho_veto.py

    veto objects based on a star catalogue.
    The tycho vetos are based on the email discussion at:
    Date: June 18, 2015 at 3:44:09 PM PDT
    To: decam-data@desi.lbl.gov
    Subject: decam-data Digest, Vol 12, Issue 29
    These objects takes a decals object and calculates the
    center and rejection radius for the catalogue in degrees.
    Note : The convention for veto flags is True for 'reject',
    False for 'preserve'.

    apply_tycho takes the galaxy catalog and appends a Tychoveto column
    the code works fine for ELG and LRGs. For other galaxy type, you need to adjust it!
"""

def BOSS_DR9(tycho):
    bmag = tycho['bmag']
    # BOSS DR9-11
    b = bmag.clip(6, 11.5)
    R = (0.0802 * b ** 2 - 1.86 * b + 11.625) / 60. #
    return R

def DECAM_LRG(tycho):
    vtmag = tycho['vtmag']
    R = 10 ** (3.5 - 0.15 * vtmag) / 3600.
    return R

DECAM_ELG = DECAM_LRG

def DECAM_QSO(tycho):
    vtmag = tycho['vtmag']
    # David Schlegel recommends not applying a bright star mask
    return vtmag - vtmag

def DECAM_BGS(tycho):
    vtmag = tycho['vtmag']
    R = 10 ** (2.2 - 0.15 * vtmag) / 3600.
    return R

def radec2pos(ra, dec):
    """ converting ra dec to position on a unit sphere.
        ra, dec are in degrees.
    """
    pos = np.empty(len(ra), dtype=('f8', 3))
    ra = ra * (np.pi / 180)
    dec = dec * (np.pi / 180)
    pos[:, 2] = np.sin(dec)
    pos[:, 0] = np.cos(dec) * np.sin(ra)
    pos[:, 1] = np.cos(dec) * np.cos(ra)
    return pos

def tycho(filename):
    """
    read the Tycho-2 catalog and prepare it for the mag-radius relation
    """
    dataf = ft.FITS(filename, lower=True)
    data = dataf[1].read()
    tycho = np.empty(len(data),
        dtype=[
            ('ra', 'f8'),
            ('dec', 'f8'),
            ('vtmag', 'f8'),
            ('vmag', 'f8'),
            ('bmag', 'f8'),
            ('btmag', 'f8'),
            ('varflag', 'i8'),
            ])
    tycho['ra'] = data['ra']
    tycho['dec'] = data['dec']
    tycho['vtmag'] = data['mag_vt']
    tycho['btmag'] = data['mag_bt']
    vt = tycho['vtmag']
    bt = tycho['btmag']
    b = vt - 0.09 * (bt - vt)
    v = b - 0.85 * (bt - vt)
    tycho['vmag']=v
    tycho['bmag']=b
    dataf.close()
    return tycho


def txts_read(filename):
    obj = np.loadtxt(filename)
    typeobj = np.dtype([
              ('RA','f4'), ('DEC','f4'), ('COMPETENESS','f4'),
              ('rflux','f4'), ('rnoise','f4'), ('gflux','f4'), ('gnoise','f4'),
              ('zflux','f4'), ('znoise','f4'), ('W1flux','f4'), ('W1noise','f4'),
              ('W2flux','f4'), ('W2noise','f4')
              ])
    nobj = obj[:,0].size
    data = np.zeros(nobj, dtype=typeobj)
    data['RA'][:] = obj[:,0]
    data['DEC'][:] = obj[:,1]
    data['COMPETENESS'][:] = obj[:,2]
    data['rflux'][:] = obj[:,3]
    data['rnoise'][:] = obj[:,4]
    data['gflux'][:] = obj[:,5]
    data['gnoise'][:] = obj[:,6]
    data['zflux'][:] = obj[:,7]
    data['znoise'][:] = obj[:,8]
    data['W1flux'][:] = obj[:,9]
    data['W1noise'][:] = obj[:,10]
    data['W2flux'][:] = obj[:,11]
    data['W2noise'][:] = obj[:,12]
    #datas = np.sort(data, order=['RA'])
    return data

def veto(coord, center, R):
    """
        Returns a veto mask for coord. any coordinate within R of center
        is vet.
        Parameters
        ----------
        coord : (RA, DEC)
        center : (RA, DEC)
        R     : degrees
        Returns
        -------
        Vetomask : True for veto, False for keep.
    """
    #from sklearn.neighbors import KDTree
    pos_stars = radec2pos(center[0], center[1])
    R = 2 * np.sin(np.radians(R) * 0.5)
    pos_obj = radec2pos(coord[0], coord[1])
    tree = KDTree(pos_obj)
    vetoflag = ~np.zeros(len(pos_obj), dtype='?')
    arg = tree.query_radius(pos_stars, r=R)
    arg = np.concatenate(arg)
    vetoflag[arg] = False
    return vetoflag



def apply_tycho(objgal, galtype='LRG',dirt='/global/cscratch1/sd/mehdi/tycho2.fits'):
    # reading tycho star catalogs
    tychostar = tycho(dirt)
    #
    # mag-radius relation
    #
    if galtype == 'LRG' or galtype == 'ELG':    # so far the mag-radius relation is the same for LRG and ELG
        radii = DECAM_LRG(tychostar)
    else:
        sys.exit("Check the apply_tycho function for your galaxy type")
    #
    #
    # coordinates of Tycho-2 stars
    center = (tychostar['ra'], tychostar['dec'])
    #
    #
    # coordinates of objects (galaxies)
    coord = (objgal['ra'], objgal['dec'])
    #
    #
    # a 0.0 / 1.0 array (1.0: means the object is contaminated by a Tycho-2 star, so 0.0s are good)
    tychomask = (~veto(coord, center, radii)).astype('f4')
    objgal = rfn.append_fields(objgal, ['tychoveto'], data=[tychomask], dtypes=tychomask.dtype, usemask=False)
    return objgal

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
