#import sys
#sys.path.append('/Users/rezaie/github/LSSutils')
import warnings
import os
import pandas as pd
import fitsio as ft
import numpy as np
import healpy as hp

from LSSutils.utils import radec2hpix


def extract_keys_dr8(mapi):
    band = mapi.split('/')[-1].split('_')[4]
    sysn = mapi.split('/')[-1].split('_')[7]
    oper = mapi.split('/')[-1].split('_')[-1].split('.')[0]
    return '_'.join((sysn, band, oper))

def IvarToDepth(ivar):
    """ function to change IVAR to DEPTH """
    depth = nanomaggiesToMag(5./np.sqrt(ivar))
    return depth

def nanomaggiesToMag(nm):
    ''' nano maggies to magnitude '''
    return -2.5 * (np.log10(nm) - 9.)

def maskmap(filename, nside=256):    
    data   = ft.read(filename, lower=True)
    if 'ivar' in filename:
        print('change ivar to depth ...')
        signal = IvarToDepth(data['signal'])
#     elif 'fwhm' in filename:
#         print('change fwhm to arcsec ...')
#         signal = data['signal']*0.262
    else:
        signal = data['signal']

    output = np.empty(12*nside*nside)
    output.fill(np.nan)
    output[data['pixel']] = signal
    return output

class Readfits(object):
    #
    def __init__(self, paths, extract_keys=extract_keys_dr8, res_out=256):
        files = paths
        print('total number of files : %d'%len(files))
        print('file-0 : %s %s'%(files[0], extract_keys(files[0])))
        self.files        = files
        self.extract_keys = extract_keys
        self.nside        = res_out
        
    def run(self, add_foreground=False):
        #
        self._run()
        if add_foreground:
            self._add_foreground()
        #
        # replace inf with nan
        self.metadata.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.ready2write  = True
        
    def save(self, path2output, name='metadata'):
        if os.path.isfile(path2output):
            warnings.warn('%s exists'%path2output)            
        self.metadata.to_hdf(path2output, name, mode='w', format='fixed')
        
    def _run(self):
        metadata = {}
        for file_i in self.files:    
            name_i  = self.extract_keys(file_i)    
            print('working on ... %s'%name_i)
            if 'ivar' in name_i:name_i = name_i.replace('ivar', 'depth')
            if name_i in metadata.keys():
                raise RuntimeError('%s already in metadata'%name_i)
            metadata[name_i] = maskmap(file_i, self.nside)            
            
        self.metadata = pd.DataFrame(metadata)
        
    def _add_foreground(self):
        from LSSutils.extrn.GalacticForegrounds import hpmaps
        # 
        Gaia    = hpmaps.gaia_dr2(nside=self.nside)
        self.metadata['nstar'] = Gaia.gaia
        
        EBV     = hpmaps.sfd98(nside=self.nside)
        self.metadata['ebv']   = EBV.ebv
        
        logNHI  = hpmaps.logHI(nside=self.nside)
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
            hp.mollview(self.metadata[name], hold=True, title=name, rot=-89)
            
        plt.savefig(path2fig, bbox_inches='tight')
        
        
        
def hd5_2_fits(myfit, cols, fitname=None, hpmask=None, hpfrac=None, fitnamekfold=None, res=256, k=5):
    from LSSutils.utils import split2Kfolds
    for output_i in [fitname, hpmask, hpfrac, fitnamekfold]:
        if output_i is not None:
            if os.path.isfile(output_i):raise RuntimeError('%s exists'%output_i)
    #
    hpind    = myfit.index.values
    label    = (myfit.ngal / (myfit.nran * (myfit.ngal.sum()/myfit.nran.sum()))).values
    fracgood = (myfit.nran / myfit.nran.mean()).values
    features = myfit[cols].values

    outdata = np.zeros(features.shape[0], 
                       dtype=[('label', 'f8'),
                              ('hpind', 'i8'), 
                              ('features',('f8', features.shape[1])),
                              ('fracgood','f8')])
    outdata['label']    = label
    outdata['hpind']    = hpind
    outdata['features'] = features
    outdata['fracgood'] = fracgood    

    if fitname is not None:
        ft.write(fitname, outdata, clobber=True)
        print('wrote %s'%fitname)

    if hpmask is not None:
        mask = np.zeros(12*res*res, '?')
        mask[hpind] = True
        hp.write_map(hpmask, mask, overwrite=True, fits_IDL=False)
        print('wrote %s'%hpmask)

    if hpfrac is not None:
        frac = np.zeros(12*res*res)
        frac[hpind] = fracgood
        hp.write_map(hpfrac, frac, overwrite=True, fits_IDL=False)
        print('wrote %s'%hpfrac)  
    
    if fitnamekfold is not None:
        outdata_kfold = split2Kfolds(outdata, k=k)
        np.save(fitnamekfold, outdata_kfold)
        print('wrote %s'%fitnamekfold)  
    
#   
class EBOSSCAT(object):
    '''
        Class to facilitate reading eBOSS cats
    '''
    def __init__(self, gals, weights=['weight_noz', 'weight_cp']):
        #
        self.weightnames = weights

        print('len of gal cats %d'%len(gals))
        gal = []
        for gali in gals:
            gald = ft.read(gali, lower=True)
            gal.append(gald)
            
        #    
        #
        gal  = np.concatenate(gal)
        
        # 
        self.gal  = gal
        self.cols = gal.dtype.names
        for colname in ['ra', 'dec', 'z', 'nz']+weights:
            if colname not in self.cols:raise RuntimeError('%s not in columns'%colname)
        self.num  = gal['ra'].size
        self.ra   = gal['ra']
        self.dec  = gal['dec']
        self.z    = gal['z']
        self.nz   = gal['nz']
        #
        #
        print('num of gal obj %d'%self.num)
        value     = np.ones(self.num)
        for weight_i in weights:
            if weight_i in self.cols:
                value *= gal[weight_i]
            else:
                print('col %s not in columns'%weight_i)
        #
        self.w = value
        
    def apply_zcut(self, zcuts=[None, None]):
        #
        # if no limits were provided
        zmin = self.z.min()
        zmax = self.z.max()
        if (zcuts[0] is None):
            zcuts[0] = zmin-1.e-7
        if (zcuts[1] is None):
            zcuts[1] = zmax+1.e-7
        print('going to apply z-cuts : {}'.format(zcuts))
        #
        #
        zmask    = (self.z > zcuts[0]) & (self.z < zcuts[1])
        self.z   = self.z[zmask]
        self.ra  = self.ra[zmask]
        self.dec = self.dec[zmask]
        self.w   = self.w[zmask]
        self.nz  = self.nz[zmask]
        self.gal = self.gal[zmask]
        self.num = self.z.size 
        print('num of gal obj after cut %d'%self.num)
    
    def swap_keys(self, key, array):
        if key not in self.cols:raise RuntimeWarning('$s not in columns'%key)
        self.gal[key] = array

    def project2hp(self, nside=512):
        from LSSutils.utils import hpixsum
        print('projecting into a healpix map with nside of %d'%nside)
        self.galm = hpixsum(nside, self.ra, self.dec, value=self.w).astype('f8')
        
    def writehp(self, filename, overwrite=True):
        if os.path.isfile(filename):
            print('%s already exists'%filename, end=' ')
            if not overwrite:
                print('please change the filename!')
                return
            else:
                print('going to rewrite....')
        hp.write_map(filename, self.galm, overwrite=True, fits_IDL=False)
            
    def plot_hist(self, titles=['galaxy map', 'Ngal distribution']):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(ncols=2, figsize=(8,3))
        plt.sca(ax[0])
        hp.mollview(self.galm, title=titles[0], hold=True)
        ax[1].hist(self.galm[self.galm!=0.0], histtype='step')
        ax[1].text(0.7, 0.8, r'%.1f $\pm$ %.1f'%(np.mean(self.galm[self.galm!=0.0]),\
                                          np.std(self.galm[self.galm!=0.0], ddof=1)),
                  transform=ax[1].transAxes)
        ax[1].set_yscale('log')
        ax[1].set_title(titles[1])
        plt.show()


class swap_weights(object):    
    def __init__(self, catalog):
        print('going to read %s'%catalog)
        self.data = ft.read(catalog)
        
    def run(self, weights, zcuts, colname='WEIGHT_SYSTOT'):
        self.orgcol = self.data[colname].copy()
        for keyi in zcuts.keys():
            assert keyi in weights.keys(), '%s not available'%keyi
            #
            my_wmap   = hp.read_map(weights[keyi], verbose=False)
            nside     = hp.get_nside(my_wmap)

            my_zcut   = zcuts[keyi]
            my_mask   = (self.data['Z'] > my_zcut[0])\
                      & (self.data['Z'] < my_zcut[1])
            #
            data_hpix = radec2hpix(nside, 
                                   self.data['RA'][my_mask],
                                   self.data['DEC'][my_mask])
            # swap
            # get the neighbors mean for non-probed pixels
            self.wmap_data = my_wmap[data_hpix]  
            nanweights     = np.argwhere(np.isnan(self.wmap_data)).flatten()
            nanhpix        = data_hpix[nanweights]
            neighbors      = hp.get_all_neighbours(nside, nanhpix)            
            self.wmap_data[nanweights] = np.nanmean(my_wmap[neighbors], axis=0)

            #self.wmap_data = self.wmap_data.clip(0.5, 2.0)
            assert np.all(self.wmap_data > 0.0),'the weights are zeros!'
            self.data[colname][my_mask] = 1./self.wmap_data            
            print('number of objs w zcut {} : {}'.format(my_zcut, my_mask.sum()))
            
    def to_fits(self, filename):
        if os.path.isfile(filename):raise RuntimeError('%s exists'%filename)
        ft.write(filename, self.data)             
