#import sys
#sys.path.append('/Users/rezaie/github/LSSutils')
import warnings
import os
import pandas as pd
import fitsio as ft
import numpy as np
import healpy as hp

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
        
        
        
def hd5_2_fits(myfit, cols, fitname, hpmask, hpfrac, fitnamekfold, res=256, k=5):
    from LSSutils.utils import split2Kfolds
    for output_i in [fitname, hpmask, hpfrac, fitnamekfold]:
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

    ft.write(fitname, outdata, clobber=True)
    print('wrote %s'%fitname)

    mask = np.zeros(12*res*res, '?')
    mask[hpind] = True
    hp.write_map(hpmask, mask, overwrite=True, fits_IDL=False)
    print('wrote %s'%hpmask)

    frac = np.zeros(12*res*res)
    frac[hpind] = fracgood
    hp.write_map(hpfrac, frac, overwrite=True, fits_IDL=False)
    print('wrote %s'%hpfrac)  
    
    outdata_kfold = split2Kfolds(outdata, k=k)
    np.save(fitnamekfold, outdata_kfold)
    print('wrote %s'%fitnamekfold)  
    
    