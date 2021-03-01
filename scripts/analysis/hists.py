from scipy.stats import binned_statistic
from glob import glob
import numpy as np
import fitsio as ft
import matplotlib.pyplot as plt

import healpy as hp

dr5data = ft.read('/Volumes/TimeMachine/data/mocks/dr7mock-features.fits')


# # binfile
def binfiles(files, features, hpix, fracgood):
     Xs = []
     Ys = []
     for i,file in enumerate(files):
         #if i % 10 == 0:print('%d/100 is done'%i)
         xs = []
         ys = []
         data = hp.read_map(file, verbose=False)
         label = data[hpix]/fracgood
         meanl = np.mean(label)
         for j in range(features.shape[1]):
             y, x, _ = binned_statistic(features[:,j], label)
             xs.append(x)
             ys.append(y/meanl)
         Xs.append(xs)
         Ys.append(ys)
     return Xs, Ys



files  = glob('/Volumes/TimeMachine/data/mocks/3dbox/*/*.hp.256.fits')
files2 = glob('/Volumes/TimeMachine/data/mocks/3dbox/*/cablin/*.cablin.hp.256.fits')



feathpixfrac = dr5data['features'], dr5data['hpix'], dr5data['fracgood']
stats  = binfiles(files, *feathpixfrac)   # uncont.
stats2 = binfiles(files2, *feathpixfrac) # contam.

f,a = plt.subplots(ncols=3,
                    nrows=6, 
                    figsize=(9,15),
                    sharey=True)
plt.subplots_adjust(hspace=0.3)
bands  = ['r', 'g', 'z']
labels = ['EBV', r'nstar $[deg^{-2}]$']
unit = dict(depth=' [mag]', seeing=' [arcsec]', airmass='', skymag=' [mag]', exptime=' [s]', mjd=' ')
for l in ['depth', 'seeing', 'skymag', 'exptime', 'mjd']:
    labels += [l+'-'+s+unit[l] for s in bands]

a = a.flatten()
f.delaxes(a[-1])
for i in range(100):
    for j in range(17):
         a[j].plot(stats[0][i][j][:-1], stats[1][i][j], alpha=0.2, color='grey')
        
for i in range(100):
    for j in range(17):
         a[j].plot(stats2[0][i][j][:-1], stats2[1][i][j], alpha=0.2, color='crimson')
         if i ==1:
             a[j].axhline(1.0, color='k', ls='--')
             a[j].set_xlabel(labels[j])
             a[j].set_ylim(0., 2)
             if j% 3 == 0:
                 a[j].set_ylabel(r'$Ngal/\overline{Ngal}$')





a[0].text(0.1, 0.9, 'Uncontaminated', color='grey', transform=a[0].transAxes)
a[0].text(0.1, 0.8, 'Contaminated', color='crimson', transform=a[0].transAxes)
plt.show()
