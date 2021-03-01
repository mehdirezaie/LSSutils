
'''
    Aug 1: add the theta and theta_err from curve_fit
           draw randomly within 1 sig. C.L. of parameters
           to contaminate the mocks
           export in=/Volumes/TimeMachine/data/mocks/dr5mock-features.fits
           export bfit=/Volumes/TimeMachine/data/dr5.0/eboss/dr5fit.npy
           export inhp=/Volumes/TimeMachine/data/mocks/
           export ouhp=/Volumes/TimeMachine/data/mocks/
           export fsysf=/Volumes/TimeMachine/data/mocks/Fsys.png
           export nnbarf=/Volumes/TimeMachine/data/mocks/nnbar_mock.pdf
           python contaminate.py --input $in --bfitparams $bfit --inhp $inhp --ouhp $ouhp --fsysfig $fsysf  --nnbarfig $nnbarf
    Jan 16:
 python contaminate.py --features /Volumes/TimeMachine/data/DR5/dr5-features.fits --bfitparams /Volumes/TimeMachine/data/DR5/eboss/regression/multivar-depthz/regression_log.npy --mocksdir /Volumes/TimeMachine/data/mocks/3dbox/ --tag c4n
'''
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#plt.rc('font', family='serif')
from   glob import glob
import fitsio as ft
import numpy as np
import healpy as hp
import sys
from   time import time
import os        



class mock(object):
     def __init__(self, featsfile, paramsfile, func='lin'):
         # read inputs
         feats       = ft.read(featsfile)
         params      = np.load(paramsfile).item()
         # attrs
         self.hpix   = feats['hpix']
         self.feats  = feats['features']
         self.ax     = params['ax']
         self.xstats = params['xstats']
         self.bfp    = params['params'][func]
         #
         # prepare
         self.n   = self.feats.shape[0]
         x        = (self.feats - self.xstats[0])/self.xstats[1] # select axis
         x_scaled = x[:, self.ax]
         if func == 'lin':
             x_vector = np.column_stack([np.ones(self.n), x_scaled])
         elif func == 'quad':
             x_vector = np.column_stack([np.ones(self.n), x_scaled, x_scaled*x_scaled])
         else:
             exit(f"func:{func} is not defined")
         #
         # 
         self.x_vector = x_vector
     
     def simulate(self, kind='truth', seed=12345):
         if kind not in ['fixed', 'random', 'truth']:
             exit(f"kind : {kind} is not defined")
         np.random.seed(seed) # set the seed
         
         if kind == 'truth':
             thetas = self.bfp[0]
         elif kind == 'fixed':
             thetas = np.random.multivariate_normal(*self.bfp)
         elif kind == 'random':
             thetas = np.random.multivariate_normal(*self.bfp, size=self.n)
         else:
             exit(f"kind : {kind} is not defined")
              
         tx       = (thetas * self.x_vector)
         self.txs = np.sum(tx, axis=1)
     
     def project(self, hpin, tag):
         hpmin = hp.read_map(hpin, verbose=False)
         fpath = '/'.join((hpin.split('/')[:-1] + [tag]))
         fname = '_'.join((tag, hpin.split('/')[-1]))
         if not os.path.exists(fpath):
            print(f'{fpath} does not exist')
            os.makedirs(fpath)

         hpout = np.zeros_like(hpmin) 
         hpout[self.hpix] = self.txs * hpmin[self.hpix]  
         fou = '/'.join((fpath, fname))
         #if os.path.isfile(fou):
         #    print(f'{fou} already exists')
         hp.write_map(fou, hpout, fits_IDL=False, overwrite=True, dtype=np.float64)


from argparse import ArgumentParser
ap = ArgumentParser(description='Multivariate linear/quadratic regression')
ap.add_argument('--features',       default='/Volumes/TimeMachine/data/mocks/dr5mock-features.fits')
ap.add_argument('--bfitparams',     default='/Volumes/TimeMachine/data/dr5.0/eboss/dr5fit.npy')
ap.add_argument('--tag',            default='c4')
ap.add_argument('--model',          default='lin')
ap.add_argument('--kind',           default='truth')
ap.add_argument('--hpmaps',         nargs='*')
ns = ap.parse_args()

# log
dicts = ns.__dict__
for a in dicts.keys():
    print(f'{a} : {dicts[a]}')

#if os.path.exists(ns.mocksdir):
#   log.write(f'path : {ns.mocksdir} exists\n')
#else:
#   log.write(f'path : {ns.mocksdir} does not exist\n')
#   os.makedirs(ns.mocksdir)

seed_i  = 12345
hpfiles = ns.hpmaps
mymock  = mock(ns.features, ns.bfitparams, func=ns.model)
mymock.simulate(kind=ns.kind, seed=seed_i)
for hpfile in hpfiles:
    mymock.project(hpfile, ns.tag)



#        print('time [sec] for mocks {} : {}'.format(seed, time()-t1))
#        theta, phi = hp.pix2ang(256, dr5data['hpix'])
#        ra, dec = np.degrees(phi), np.degrees(np.pi/2 - theta)
#        fig,a = plt.subplots(ncols=3, figsize=(12, 3), sharey=True)
#        plt.subplots_adjust(wspace=0.05)
#        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
#        from mpl_toolkits.axes_grid1.colorbar import colorbar
#        f0 = a[0].scatter(ra, dec, s=1, c=ngal_uncon[dr5data['hpix']], cmap=plt.cm.inferno, vmin=0, vmax=ngal_con.max())
#        f1 = a[1].scatter(ra, dec, s=1, c=model, cmap=plt.cm.inferno)
#        f2 = a[2].scatter(ra, dec, s=1, c=ngal_con[dr5data['hpix']], cmap=plt.cm.inferno, vmin=0, vmax=ngal_con.max())
#        for i in range(3):a[i].set_xlabel('RA')
#        a[0].set_ylabel('DEC')
#        flist = [f0, f1, f2]
#        label = ['uncont.', 'Fsys', 'cont.']
#        for i,a_i in enumerate(a):            
#            a_i.text(0.02, 0.93, label[i], color='k', transform=a_i.transAxes)
#            ax2_divider = make_axes_locatable(a_i)
#            cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
#            cb2 = colorbar(flist[i], cax=cax2, orientation="horizontal")
#            cax2.xaxis.set_ticks_position("top")
#        plt.savefig(ns.mocksdir+dirl+'/'+tag+'/Fsys.png', bbox_inches='tight', dpi=300)
#        print('plotting '+ns.mocksdir+dirl+'/'+tag+'/Fsys.png')
#
#
# plot the nnbar for mocks
#
# from scipy.stats import binned_statistic
# from glob import glob

# # binfile
# def binfiles(files, features, hpix, fracgood):
#     Xs = []
#     Ys = []
#     for i,file in enumerate(files):
#         #if i % 10 == 0:print('%d/100 is done'%i)
#         xs = []
#         ys = []
#         data = hp.read_map(file, verbose=False)
#         label = data[hpix]/fracgood
#         meanl = np.mean(label)
#         for j in range(features.shape[1]):
#             y, x, _ = binned_statistic(features[:,j], label)
#             xs.append(x)
#             ys.append(y/meanl)
#         Xs.append(xs)
#         Ys.append(ys)
#     return Xs, Ys



# files  = glob(ns.mocksdir + 'seed*/3dbox*hp256.fits')
# files2 = glob(ns.mocksdir + 'seed*/'+tag+'/cont3dbox*hp256.fits')

# feathpixfrac = dr5data['features'], dr5data['hpix'], dr5data['fracgood']
# stats  = binfiles(files, *feathpixfrac)   # uncont.
# stats2 = binfiles(files2, *feathpixfrac) # contam.

# f,a = plt.subplots(ncols=3,
#                    nrows=6, 
#                    figsize=(9,15),
#                    sharey=True)
# plt.subplots_adjust(hspace=0.3)
# bands  = ['r', 'g', 'z']
# labels = ['EBV', r'nstar $[deg^{-2}]$']
# unit = dict(depth=' [mag]', seeing=' [arcsec]', airmass='', skymag=' [mag]', exptime=' [s]')
# for l in ['depth', 'seeing', 'airmass', 'skymag', 'exptime']:
#     labels += [l+'-'+s+unit[l] for s in bands]

# a = a.flatten()
# f.delaxes(a[-1])
# for i in range(100):
#     for j in range(17):
#         a[j].plot(stats[0][i][j][:-1], stats[1][i][j], alpha=0.2, color='grey')
        
# for i in range(100):
#     for j in range(17):
#         a[j].plot(stats2[0][i][j][:-1], stats2[1][i][j], alpha=0.2, color='crimson')
#         if i ==1:
#             a[j].axhline(1.0, color='k', ls='--')
#             a[j].set_xlabel(labels[j])
#             a[j].set_ylim(0.4, 1.6)
#             if j% 3 == 0:
#                 a[j].set_ylabel(r'$Ngal/\overline{Ngal}$')
# a[0].text(0.1, 0.9, 'Uncontaminated', color='grey', transform=a[0].transAxes)
# a[0].text(0.1, 0.8, 'Contaminated', color='crimson', transform=a[0].transAxes)
# plt.savefig(ns.mocksdir + 'nnbar_'+tag+'.pdf', bbox_inches='tight', dpi=300)
# print('plotting '+ns.mocksdir +'nnbar_'+tag+'.pdf')
