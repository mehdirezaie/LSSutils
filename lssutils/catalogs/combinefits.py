import sys
#sys.path.append('/Users/rezaie/github/lssutils')
import warnings
import os
import pandas as pd
import fitsio as ft
import numpy as np
import healpy as hp

from lssutils.utils import radec2hpix, hpixsum, shiftra, make_overdensity
from astropy.table import Table
import logging

#logging.basicConfig(level=logging.INFO)


__all__ = ['EbossCatalog', 'SysWeight']



def extract_keys_dr8(mapi):
    band = mapi.split('/')[-1].split('_')[4]
    sysn = mapi.split('/')[-1].split('_')[7]
    oper = mapi.split('/')[-1].split('_')[-1].split('.')[0]
    return '_'.join((sysn, band, oper))

def IvarToDepth(ivar):
    """ function to change IVAR to DEPTH """
    depth = nanomaggiesToMag(5./np.sqrt(ivar))
    return depth

def Magtonanomaggies(m):
    return 10.**(-m/2.5+9.)

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
        from lssutils.extrn.galactic import hpmaps
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
        if logger is not None:
            logger.info('wrote %s'%fitname)

    if hpmask is not None:
        mask = np.zeros(12*res*res, '?')
        mask[hpind] = True
        hp.write_map(hpmask, mask, overwrite=True, fits_IDL=False)
        if logger is not None:
            logger.info('wrote %s'%hpmask)

    if hpfrac is not None:
        frac = np.zeros(12*res*res)
        frac[hpind] = fracgood
        hp.write_map(hpfrac, frac, overwrite=True, fits_IDL=False)
        if logger is not None:
            logger.info('wrote %s'%hpfrac)  
    
    if fitnamekfold is not None:
        outdata_kfold = split2Kfolds(outdata, k=k)
        np.save(fitnamekfold, outdata_kfold)
        if logger is not None:
            logger.info('wrote %s'%fitnamekfold)  
        



def make_clustering_catalog(mock):
    # (c) Julien Bautista 
    w = ((mock['IMATCH']==1) | (mock['IMATCH']==2))
    w &= (mock['COMP_BOSS'] > 0.5)
    w &= (mock['sector_SSR'] > 0.5)

    names = ['RA', 'DEC', 'Z', 'WEIGHT_FKP', 'WEIGHT_SYSTOT', 'WEIGHT_CP']
    names += ['WEIGHT_NOZ', 'NZ', 'QSO_ID']

    mock = mock[w]

    fields = []
    for name in names:
        fields.append(mock[name])
    mock_clust = Table(fields, names=names)
    return mock_clust

def reassignment(randoms, data, seed=None):
    '''
    This function re-assigns the attributes from data to randoms
    
    Parameters
    ----------
    randoms : numpy structured array for randoms
    
    data : numpy structured array for data
        
        
    Returns
    -------
    rand_clust : numpy structured array for randoms
        

    (c) Julien Bautista
    
    Updates
    --------
    March 9, 20: Z, NZ, FKP must be assigned from data
    
    Examples    
    --------
    '''

    rand_clust = Table()
    rand_clust['RA'] = randoms['RA']*1
    rand_clust['DEC'] = randoms['DEC']*1
    rand_clust['COMP_BOSS'] = randoms['COMP_BOSS']*1
    rand_clust['sector_SSR'] = randoms['sector_SSR']*1

    if seed is not None:
        np.random.seed(seed)
    
    index = np.arange(len(data))
    indices = np.random.choice(index, size=len(randoms), replace=True)
    
    fields = ['WEIGHT_NOZ', 'WEIGHT_CP', 'WEIGHT_SYSTOT', 'WEIGHT_FKP', 'Z', 'NZ'] 
    for f in fields:
        rand_clust[f] = data[f][indices]

    #-- As in real data:
    rand_clust['WEIGHT_SYSTOT'] *= rand_clust['COMP_BOSS']

    w = (rand_clust['COMP_BOSS'] > 0.5) & (rand_clust['sector_SSR'] > 0.5) 

    return rand_clust[w]

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
class EbossCatalog:
    
    logger = logging.getLogger('EbossCatalog')
    
    columns = ['RA', 'DEC', 'Z', 
               'WEIGHT_FKP', 'WEIGHT_SYSTOT', 'WEIGHT_CP',
               'WEIGHT_NOZ', 'NZ', 'QSO_ID', 'IMATCH',
               'COMP_BOSS', 'sector_SSR']    
    comp_min = 0.5
    
    def __init__(self, filename, kind='galaxy', **clean_kwargs):
        self.kind  = kind
        self.read(filename)
        self.clean(**clean_kwargs)
        
    def read(self, filename):
        if filename.endswith('.fits'):
            self.data  = Table.read(filename)
        else:
            raise NotImplementedError(f'file {filename} not implemented')
    
    def clean(self, zmin=0.8, zmax=2.2):
        ''' `Full` to `Clustering` Catalog
        '''           
        columns = []
        for i, column in enumerate(self.columns):
            if column not in self.data.columns:
                self.logger.warning(f'column {column} not in the {self.kind} file')
            else:
                columns.append(column)
                
        self.columns = columns
        self.data  = self.data[self.columns]        
        
        #-- apply cuts on galaxy or randoms
        good = (self.data['Z'] > zmin) & (self.data['Z'] < zmax)
        self.logger.info(f'{zmin} < z < {zmax}')
        for column in ['COMP_BOSS', 'sector_SSR']:
            if column in self.data.columns:                
                good &= self.data[column] > self.comp_min
                self.logger.info(f'{column} > {self.comp_min}')
                
        if self.kind=='galaxy':
            if 'IMATCH' in self.data.columns:
                good &= (self.data['IMATCH']==1) | (self.data['IMATCH']==2)
                self.logger.info(f'IMATCH = 1 or 2 for {self.kind}')
                                
        self.logger.info(f'{good.sum()} ({100*good.mean():3.1f}%) {self.kind} pass the cuts')
        self.data = self.data[good]
        
    def prepare_weights(self, raw=0):        
        self.logger.info(f'raw: {raw}')        
        
        if raw==1:                        
            if self.kind == 'galaxy':
                self.data['WEIGHT'] = self.data['WEIGHT_FKP']
                self.data['WEIGHT'] *= self.data['WEIGHT_CP']
                self.data['WEIGHT'] *= self.data['WEIGHT_NOZ']
                
            elif self.kind == 'random':                
                self.data['WEIGHT'] = self.data['WEIGHT_FKP']
                self.data['WEIGHT'] *= self.data['COMP_BOSS']
                
            else:
                raise ValueError(f'{self.kind} not defined')
                
        elif raw==2:
            # data and randoms both are weighted by CP x FKP x NOZ x SYSTOT
            self.data['WEIGHT'] = self.data['WEIGHT_FKP']
            self.data['WEIGHT'] *= self.data['WEIGHT_CP']            
            self.data['WEIGHT'] *= self.data['WEIGHT_NOZ']
            self.data['WEIGHT'] *= self.data['WEIGHT_SYSTOT']
        elif raw==0:
            self.data['WEIGHT'] = 1.0
        else:
            raise ValueError(f'{raw} should be 0, 1, or 2!')
            
    def tohp(self, nside, zmin, zmax, raw=0): 
        self.prepare_weights(raw=raw)
        assert 'WEIGHT' in self.data.columns, "run `self.prepare_weights'"
        self.logger.info(f'Projecting {self.kind}  to HEALPix with {nside}')
        good = (self.data['Z'] > zmin) & (self.data['Z'] < zmax)
        self.logger.info((f'{good.sum()} ({100*good.mean():3.1f}%)'
                          f' {self.kind} pass ({zmin:.1f} < z < {zmax:.1f})'))
        
        return hpixsum(nside, 
                       self.data['RA'][good], 
                       self.data['DEC'][good], 
                       weights=self.data['WEIGHT'][good])
    
    def __getitem__(self, index):
        return self.data[index]

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
               
        if label=='nnbar':
            return self._prep_nnbar(nside, zmin, zmax, frac_min, nran_exp)
        elif label=='ngal':
            return self._prep_ngal(nside, zmin, zmax, frac_min, nran_exp)
        elif label=='ngalw':
            return self._prep_ngalw(nside, zmin, zmax, frac_min, nran_exp)
        else:
            raise ValueError(f'{label} must be nnbar, ngal, or ngalw')
    
    def _prep_nnbar(self, nside, zmin, zmax, frac_min, nran_exp):
        
        ngal = self.data.tohp(nside, zmin, zmax, raw=1)        
        nran = self.randoms.tohp(nside, zmin, zmax, raw=1)
        if nran_exp is None:
            nran_exp = np.mean(nran[nran>0])
            self.logger.info(f'using {nran_exp} as nran_exp')            
            
        frac = nran / nran_exp
        
        mask_random = (frac >  frac_min)        
        mask = mask_random & self.mask                
        self.logger.info(f'{mask.sum()} pixels ({mask.mean()*100:.1f}%) have imaging')
        
        nnbar = overdensity(ngal, nran, mask, nnbar=True) 
        
        return self._to_numpy(nnbar[mask], self.features[mask, :],
                             frac[mask], np.argwhere(mask).flatten())        
        
    def _prep_ngalw(self, nside, zmin, zmax, frac_min, nran_exp):
        
        ngal = self.data.tohp(nside, zmin, zmax, raw=1)        
        nran = self.randoms.tohp(nside, zmin, zmax, raw=1)
        if nran_exp is None:
            nran_exp = np.mean(nran[nran>0])
            self.logger.info(f'using {nran_exp} as nran_exp')            
            
        frac = nran / nran_exp        
        mask_random = (frac >  frac_min)        
        mask = mask_random & self.mask        
        self.logger.info(f'{mask.sum()} pixels ({mask.mean()*100:.1f}%) have imaging')
        
        return self._to_numpy(ngal[mask], self.features[mask, :],
                             frac[mask], np.argwhere(mask).flatten())

    def _prep_ngal(self, nside, zmin, zmax, frac_min, nran_exp):
        
        ngal = self.data.tohp(nside, zmin, zmax, raw=0)
        ngalw = self.data.tohp(nside, zmin, zmax, raw=1)        
        
        wratio = np.zeros_like(ngal)
        good = ngal > 0.0
        wratio[good] = ngalw[good]/ngal[good]        
        
        nran = self.randoms.tohp(nside, zmin, zmax, raw=1)
        if nran_exp is None:
            nran_exp = np.mean(nran[nran>0])
            self.logger.info(f'using {nran_exp} as nran_exp')            
            
        frac = nran / nran_exp
        
        mask_random = (frac >  frac_min)        
        mask = mask_random & self.mask        
        self.logger.info(f'{mask.sum()} pixels ({mask.mean()*100:.1f}%) have imaging')
        
        wratio[mask & (~good)] = 1.0 # have randoms but no data
        fracw = np.zeros_like(frac)
        fracw[mask] = frac[mask] / wratio[mask]
        
        return self._to_numpy(ngal[mask], self.features[mask, :],
                             fracw[mask], np.argwhere(mask).flatten())    
    
    def _to_numpy(self, t, features, frac, hpind):
        
        dtype = [('features', ('f8', features.shape[1])), 
                 ('label', 'f8'),
                 ('fracgood', 'f8'),
                 ('hpind', 'i8')]    
        
        dataset = np.zeros(t.size, dtype=dtype)
        dataset['label'] = t
        dataset['fracgood'] = frac
        dataset['features'] = features
        dataset['hpind'] = hpind
        
        return dataset    

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
                NNaNs  = np.isnan(wsys).sum()
                self.logger.info(f'# NaNs (after)  : {NNaNs}')

            
            
        assert np.all(wsys > 0.0),f'{(wsys <= 0.0).sum()} weights <= 0.0!' 
        
        return 1./wsys # Systematic weight = 1 / Selection mask