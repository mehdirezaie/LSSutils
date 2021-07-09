"""
    Read selection mask from NN
    Compute Mean density given DR9 templates
    Plot Selection masks
"""
import numpy as np
import fitsio as ft
import healpy as hp

import sys
sys.path.insert(0, '/home/mehdi/github/sysnetdev')
sys.path.insert(0, '/home/mehdi/github/LSSutils')

from lssutils.utils import hpix2radec
from lssutils.stats.nnbar import MeanDensity
from lssutils.stats.cl import get_cl as get_angularpower
from lssutils.dataviz import mycolor

import matplotlib.pyplot as plt

def makehp(hpix, value, nside):
    res_ = np.zeros(12*nside*nside)
    res_[hpix] = value
    return res_

# input parameters
sample = sys.argv[1] # e.g. 'elg'
cap = sys.argv[2]    # e.g. 'N'

nside = 256

get_nbar = False
make_nbarplot = False
make_radecplot = False
get_cl=True


maps = ['STARDENS', 'EBV', 
      'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 
      'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 
      'PSFDEPTH_W1', 'PSFDEPTH_W2',
      'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

axes = dict(zip(maps, np.arange(len(maps))))



data = ft.read(f'results/dr9m_{sample}_{cap}.fits')
npred_ = ft.read(f'results/regression/{sample}/{cap}/nn_all_{nside}/nn-weights.fits')

# to healpix, normalize, clip
npred = np.zeros(12*nside*nside)
npred[npred_['hpix']] = np.mean(npred_['weight'], axis=1)

hpix = data['hpix'].copy()
frac = data['fracgood'].copy()
ngal = data['label'].copy()
mask = frac > 0

sf = (ngal/npred[hpix]).sum()/ngal.sum()
wnn = npred[hpix]*sf
wnn = wnn.clip(0.5, 2.0)

if get_cl:
    
    ngal = makehp(hpix, ngal, nside)
    frac = makehp(hpix, frac, nside)
    mask = frac > 0
    wnn  = makehp(hpix, wnn, nside)
    
    features = []
    for i in range(data['features'].shape[1]):
        features.append(makehp(hpix, data['features'][:, i], nside))
    features = np.array(features).T
    
    cl_before = get_angularpower(ngal, frac, mask, selection_fn=None, systematics=features)
    cl_after = get_angularpower(ngal, frac, mask, selection_fn=wnn, systematics=features)
    
    np.savez(f'results/cl_{sample}_{cap}GC.npz', **{'before':cl_before, 'after':cl_after})
    
if get_nbar:
    
    # mean density
    nbar = {}

    for sysname in maps:

        ix = axes[sysname]
        sysm = data['features'][:, ix]

        nb_true = MeanDensity(ngal, frac, mask, sysm)
        nb_pred = MeanDensity(wnn*frac/sf, frac, mask, sysm)
        nb_corr = MeanDensity(ngal, frac, mask, sysm, selection=wnn)

        nb_true.run()
        nb_pred.run()
        nb_corr.run()

        nbar_ = {}
        for nmi, nbi in zip(['Observed', 'Observed (corrected)', 'Model'],
                            [nb_true, nb_corr, nb_pred]):

            nbar_[nmi] = (nbi.output['bin_avg'], nbi.output['nnbar']-1, nbi.output['nnbar_err'])

        nbar[sysname] = nbar_

        print('.', end='')
    np.savez(f'results/nbar_{sample}_{cap}GC.npz', **nbar)

else:
    nbar = np.load(f'results/nbar_{sample}_{cap}GC.npz', allow_pickle=True) 
    
    
# plot
if make_nbarplot:
    
    fig, ax = plt.subplots(ncols=3, nrows=5, 
                           figsize=(18, 20), sharey='row')
    fig.subplots_adjust(wspace=0)
    ax = ax.flatten()

    for i, sysname in enumerate(maps):

        for nmi in ['Observed', 'Observed (corrected)', 'Model']:

            ax[i].errorbar(*nbar[sysname][nmi], 
                           capsize=2, marker='o', 
                           label=nmi, mfc='w')

        ax[i].set_xlabel(sysname)
        ax[i].axhline(0.0, ls=':', lw=1, alpha=0.5)    
        ax[i].tick_params(direction='in', which='both', axis='both', right=True)
        if i == 2:ax[i].legend(title=f'{sample.upper()} {cap}GC')
        if i % 3 == 0:ax[i].set_ylabel(r'$\delta$')

    fig.delaxes(ax[14])
    fig.delaxes(ax[13])


    fig.savefig(f'results/nbar_{sample}_{cap}GC.png', dpi=300, bbox_inches='tight')


if make_radecplot:
    # plot ra dec selection mask
    ra, dec = hpix2radec(nside, hpix)
    pixarea = hp.nside2pixarea(nside, degrees=True)
    fig, ax = plt.subplots(ncols=3, figsize=(18, 4), sharey=True)

    fig.subplots_adjust(wspace=0.0)

    vmin, vmax = np.percentile(ngal/(frac*pixarea), [5, 95])
    kw = dict(cmap=mycolor(), vmin=vmin, vmax=vmax, rasterized=True)


    sindec = np.sin(np.deg2rad(dec))

    cax = plt.axes([0.3, -0.1, 0.4, 0.04])


    ax[0].scatter(ra, sindec, 2, c=ngal/(frac*pixarea), **kw)


    ax[1].scatter(ra, sindec, 2, c=wnn/(pixarea*sf), **kw)

    map1 = ax[2].scatter(ra, sindec, 2, c=ngal/(frac*pixarea*wnn), **kw)


    fig.colorbar(map1, cax=cax, label=fr'{sample} {cap}GC [deg$^{-2}$]', 
                 extend='both', orientation='horizontal')

    ax[1].set_xlabel('RA [deg]')
    ax[0].set_ylabel('sin(DEC)')
    fig.savefig(f'./results/radec_{sample}_{cap}GC.png', dpi=300, bbox_inches='tight')