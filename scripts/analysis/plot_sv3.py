""" Plot SV3 Results """

# LRGs
import sys
sys.path.append('/home/mehdi/github/LSSutils')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import healpy as hp

import numpy as np
from time import time
import fitsio as ft

from lssutils.lab import (make_overdensity, AnaFast, 
                          histogram_cell, hpixsum, get_meandensity)
from lssutils.stats.pcc import pcc
from lssutils.dataviz import setup_color
import pandas as pd

root_dir = '/home/mehdi/data/dr9v0.57.0/'

class SV3Data:
    
    def __init__(self, target, region):
        
        self.nside = 256
        
        p = f'{root_dir}sv3_v1/'
        self.dcat = ft.read(f'{p}sv3target_{target}_{region}.fits', 
                            columns=['RA', 'DEC'])
        self.rcat = ft.read(f'{p}{region}_randoms-1-0x2.fits', 
                            columns=['RA', 'DEC'])
        
        self.wrf = ft.read(f'{p}sv3target_{target}_{region}.fits_EdWsys/wsys_v0.fits')['wsys']
        print(f'mean(wrf): {self.wrf.mean():.2f}, {self.wrf.min():.1f} < wrf < {self.wrf.max():.1f}')
        
        self.wnn = ft.read(f'{p}sv3target_{target}_{region}.fits_MrWsys/wsys_v0.fits')['wsys']        
        print(f'mean(wnn): {self.wnn.mean():.2f}, {self.wnn.min():.1f} < wnn < {self.wnn.max():.1f}')

        
        self.af = AnaFast()
        
        tmpl = pd.read_hdf(f'/home/mehdi/data/templates/dr9/pixweight_dark_dr9m_nside{self.nside}.h5')
        #self.cols = ['nstar', 'ebv', 'loghi']\
        #              +[f'{s}_{b}' for s in ['ccdskymag_mean', 'fwhm_mean', 'fwhm_min', 'fwhm_max', 'depth_total', 
        #                                'mjd_mean', 'mjd_min', 'mjd_max', 'airmass_mean', 'exptime_total']\
        #                      for b in ['g', 'r', 'z']]
        self.cols = ['stardens', 'ebv', 'loghi',
                     'psfdepth_g', 'psfdepth_r', 'psfdepth_z',
                     'galdepth_g', 'galdepth_r', 'galdepth_z', 
                     'psfsize_g', 'psfsize_r', 'psfsize_z', 
                     'psfdepth_w1', 'psfdepth_w2']

        self.tmpl = tmpl[self.cols].values        
        
        
        
    def make_delta(self):

        nran = hpixsum(self.nside, self.rcat['RA'], self.rcat['DEC'])*1.0
        self.mask = (nran > 0)
        print(f'mask: {self.mask.sum()} pixels')
        
        is_good = np.isfinite(self.tmpl).sum(axis=1) == 14
        self.mask &= is_good
        
        print(f'mask: {self.mask.sum()} pixels (with imaging)') 
        self.frac = nran / nran[self.mask].mean()
        
        self.ngal_now = hpixsum(self.nside, self.dcat['RA'], self.dcat['DEC'])*1.0
        self.ngal_rf  = hpixsum(self.nside, self.dcat['RA'], self.dcat['DEC'], weights=self.wrf)
        self.ngal_wnn = hpixsum(self.nside, self.dcat['RA'], self.dcat['DEC'], weights=self.wnn)
        
        self.delta_now = make_overdensity(self.ngal_now, self.frac, self.mask)
        self.delta_rf  = make_overdensity(self.ngal_rf,   self.frac, self.mask)
        self.delta_wnn = make_overdensity(self.ngal_wnn, self.frac, self.mask)   
        
    def make_cl(self):
        self.cl_now = self.af(self.delta_now, self.frac, self.mask)
        self.cl_rf = self.af(self.delta_rf, self.frac, self.mask)
        self.cl_nn = self.af(self.delta_wnn, self.frac, self.mask)        
        
    def make_nbar(self):
        self.nbar_now = get_meandensity(self.ngal_now, self.frac, self.mask, self.tmpl)
        self.nbar_rf  = get_meandensity(self.ngal_rf,  self.frac, self.mask, self.tmpl)
        self.nbar_nn  = get_meandensity(self.ngal_wnn, self.frac, self.mask, self.tmpl)
        
    def make_pcc(self):
        self.pcc_now = pcc(self.tmpl[self.mask], self.delta_now[self.mask], return_err=True)
        self.pcc_rf  = pcc(self.tmpl[self.mask], self.delta_rf[self.mask])
        self.pcc_nn  = pcc(self.tmpl[self.mask], self.delta_wnn[self.mask])


setup_color()



region = sys.argv[1] # NDECALS
target = sys.argv[2] # QSO

assert region in ['NDECALS', 'SDECALS', 'NBMZLS', 'DES', 'SDECALS_noDES']
assert target in ['QSO', 'LRG', 'ELG', 'BGS_ANY']


print(f'target: {target}, region: {region}')

target_region = f'{target}-{region}'


t0 = time()
sv = SV3Data(target, region)
t1 = time()
print(f'Finished reading in {t1-t0:.1f} sec')

sv.make_delta()
t2 = time()
print(f'Finished deltas in {t2-t1:.1f} sec')


sv.make_cl()
t3 = time()
print(f'Finished Cell in {t3-t2:.1f} sec')


sv.make_nbar()
t4 = time()
print(f'Finished nbar in {t4-t3:.1f} sec')


sv.make_pcc()
t5 = time()
print(f'Finished pcc in {t5-t4:.1f} sec')



pp = PdfPages(''.join([target_region, '.pdf']))



# C_ell
methods = ['No weight', 'RF weight', 'NN weight']
cls = [sv.cl_now, sv.cl_rf, sv.cl_nn]

fg, ax = plt.subplots(figsize=(8, 6))
for n_i, cl_i in zip(methods, cls ):

    lb, clb = histogram_cell(cl_i['cl'], bins=np.logspace(0, np.log10(770), 10))
    
    l_, = ax.plot(cl_i['cl'], lw=1, zorder=-1, alpha=0.2)
    ax.plot(lb, clb, marker='.', mfc='w', ls='None', color=l_.get_color(), label=n_i)
    
ax.legend(title=target_region, frameon=False)
ax.set(xscale='log', yscale='log', ylim=(2.0e-8, 8.0e-3), 
       xlabel=r'$\ell$', ylabel=r'C$_{\ell}$')

#fg.savefig('cl_lrg_bmzls.png', dpi=300, bbox_inches='tight')
pp.savefig(bbox_inches='tight')


# Nbar
fig, ax = plt.subplots(ncols=3, nrows=5, figsize=(22, 25), sharey=True)
fig.subplots_adjust(hspace=0.35, wspace=0.1)
ax = ax.flatten()

nbars = [sv.nbar_now, sv.nbar_rf, sv.nbar_nn]
for name_i, nbar_i in zip(methods, nbars):    
    for j, nbar_ij in enumerate(nbar_i):
        ax[j].plot(nbar_ij['bin_avg'], nbar_ij['nnbar'], marker='.', mfc='w', label=name_i)        
        if name_i == 'No weight':
            ax[j].fill_between(nbar_ij['bin_avg'], 1-nbar_ij['nnbar_err'], 1+nbar_ij['nnbar_err'],
                              color='grey', alpha=0.2, zorder=-1)                          
ax[2].legend(title=target_region, frameon=False)
for j, colj in enumerate(sv.cols):
    ax[j].set_xlabel(colj)
    if j%3==0:
        ax[j].set_ylabel('Mean Density')        
pp.savefig(bbox_inches='tight')

# PCC
fg, ax = plt.subplots(figsize=(12, 4))
x_columns = np.arange(len(sv.cols))
ax.set_xticks(x_columns)
ax.set_xticklabels(sv.cols, rotation=90)

pcc_min, pcc_max = np.percentile(sv.pcc_now[1], [2.5, 97.5], axis=0)

ax.bar(x_columns-0.25, sv.pcc_now[0], width=0.25, label='No weight')
ax.bar(x_columns,      sv.pcc_rf[0],  width=0.25, label='RF')
ax.bar(x_columns+0.25, sv.pcc_nn[0], width=0.25, label='NN')
ax.fill_between(x_columns, pcc_min, pcc_max, color='grey', alpha=0.2, zorder=10)
ax.legend(title=target_region, frameon=False)
ax.grid(ls=':')
ax.set(ylabel='PCC')
pp.savefig(bbox_inches='tight')

pp.close()
