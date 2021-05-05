'''
    Galactic foregrounds
    1. SFD 1998 E(B-V)
    2. Gaia DR2 Stellar density
    3. HI column density Lenz et. al.
'''
import healpy as hp
import numpy  as np
import fitsio as ft
import warnings

#templ_dir = '/Volumes/TimeMachine/data/templates/'
templ_dir = '/home/mehdi/data/templates/'

class NStarSDSS:
    """
        Uses the SDSS stellar density map (used in Bautista et al. 2018)
        
        (c) Ashley Ross
    """    
    def __init__(self, name=f'{templ_dir}allstars17.519.9Healpixall256.dat', nside_out=256):
        self.unit = '# stars'        
        self.nstar = np.loadtxt(name)
        self.map = hp.reorder(self.nstar, n2r=True)
        self.nside = hp.get_nside(self.map)
        
        if nside_out != self.nside:
            self.map = hp.ud_grade(self.map, nside_out=nside_out, power=-2)
            warnings.warn('upgrading/downgrading SDSS star density')

class SFD98(object):
    """ 
        Read E(B-V) from SFD 1998 
    """
    def __init__(self, nside_out=256):
        self.nside_out    = nside_out
        self.ordering = 'ring'
        self.unit     = 'EBV [SFD 98]'
        self.name     = f'{templ_dir}healSFD_256_fullsky.fits'
        self.map      = hp.read_map(self.name, verbose=False)

        if nside_out!=256:
            self.map = hp.ud_grade(self.map, nside_out=nside_out)
            warnings.warn('upgrading/downgrading EBV')

class Gaia(object):
    """
        Read the Gaia DR2 star density catalog (c) Anand Raichoor
    """
    def __init__(self, path=f'{templ_dir}Gaia.dr2.bGT10.12g17.hp256.fits', nside_out=256):

        self.ordering = 'ring'
        self.unit     = 'Gaia DR2'
        self.map     = ft.read(path, lower=True, columns=['hpstardens'])['hpstardens'] # only column
        
        self.nside_in = hp.get_nside(self.map)
        # the map is in per sq. deg., this transforms it to # of stars
        self.map     = self.map.astype('f8') * hp.nside2pixarea(self.nside_in, degrees=True)

        if nside_out!=self.nside_in:
            self.map = hp.ud_grade(self.map, nside_out=nside_out, power=-2)
            warnings.warn('upgrading/downgrading Gaia star density')  


def G_to_C(mapi, res_in=1024, res_out=256):
    """
     Rotates from Galactic to Celestial coordinates
    """
    thph    = hp.pix2ang(res_out, np.arange(12*res_out*res_out))
    r       = hp.Rotator(coord=['C', 'G'])
    thphg   = r(thph[0], thph[1])
    hpix    = hp.ang2pix(res_in, thphg[0], thphg[1])
    return mapi[hpix]



class logHI(object):
    ''' Reads Lenz et. al. HI column density '''
    def __init__(self, nside_out=256, path=f'{templ_dir}NHI_HPX.fits'):

        self.nside_out    = nside_out
        nside_in      = 1024
        self.ordering = 'ring'
        self.unit     = 'Lenz et. al. HI'

        if nside_out!= nside_in:warnings.warn('upgrading/downgrading HI column density')
        nhi           = ft.FITS(path, lower=True)
        nhi           = nhi[1].read()['nhi'] # only nhi column

        nhi_c         = G_to_C(nhi, res_in=nside_in, res_out=nside_out)
        nhi_neg_hpix  = np.argwhere(nhi_c <= 0.0).flatten()
        neighbors     = hp.get_all_neighbours(nside_out, nhi_neg_hpix)

        nhi_c[nhi_neg_hpix] = np.mean(nhi_c[neighbors], axis=0)

        self.map    = np.log10(nhi_c)
