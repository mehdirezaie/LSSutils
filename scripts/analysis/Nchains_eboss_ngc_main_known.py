#!/usr/bin/env python
# coding: utf-8

# # The number of chains and angular C_ell

# In[1]:


import sys
sys.path.insert(0, '/home/mehdi/github/LSSutils')


# In[4]:


import matplotlib.pyplot as plt

import fitsio as ft
import numpy as np
import healpy as hp

from lssutils.lab import get_cl, EbossCat, histogram_cell


# In[5]:


def normalize(nnmodel, nobsw, good):
    """ normalize systematic weights such that N_{qso,tot} stays the same """
    norm = (nobsw[good]/nnmodel[good]).sum()/nobsw[good].sum()
    return norm*nnmodel

def make_hp(hpix, values, nside=512):
    """ make HEALPix """
    mapi = np.zeros(12*nside*nside)
    mapi[hpix] = values
    return mapi

# read data and randoms
zmin = 0.8
zmax = 2.2
nside = 512

cats_dir = '/home/mehdi/data/eboss/data/v7_2/'

data = EbossCat(f'{cats_dir}eBOSS_QSO_full_NGC_v7_2.dat.fits', kind='data', zmin=0.8, zmax=2.2)
randoms = EbossCat(f'{cats_dir}eBOSS_QSO_full_NGC_v7_2.ran.fits', kind='randoms', zmin=0.8, zmax=2.2)

# to HEALPix
nobs_w = data.to_hp(nside, zmin, zmax, raw=1)
nran_w = randoms.to_hp(nside, zmin, zmax, raw=1)
nran_mean = 5000.*hp.nside2pixarea(nside, degrees=True)


nobs_def = data.to_hp(nside, zmin, zmax, raw=2)
nran_def = randoms.to_hp(nside, zmin, zmax, raw=2)

print(f'nran_mean: {nran_mean}')

# new NN result
wnn = ft.read(f'{cats_dir}1.0/NGC/512/main/nn_pnnl_known/nn-weights.fits')
hpix = wnn['hpix']
mask = np.zeros(12*nside*nside, '?')
mask[hpix] = True

# old NN result
wnn_old = ft.read(f'{cats_dir}0.3/results/NGC_all_512/regression/nn_known/nn-weights512.fits')
wnn_old_hp = make_hp(wnn_old['hpind'], wnn_old['weight'].mean(axis=1))


# In[7]:


assert np.array_equal(np.sort(wnn_old['hpind']), np.sort(wnn['hpix']))


# In[8]:


# compute C_ells

kw = dict(njack=0, nran_bar=nran_mean)

cl_old = get_cl(nobs_w, nran_w, mask, selection_fn=wnn_old_hp, **kw) # old NN
cl_def = get_cl(nobs_def, nran_def, mask, **kw)                      # default systot

cl_avg = {}
cl_avg[-1] = get_cl(nobs_w, nran_w, mask, selection_fn=None, **kw)   # wo correction

# average wNN over chains -> get C_ell
for idx in range(1, wnn['weight'].shape[1]+1):

    wsys = wnn['weight'][:, :idx].mean(axis=1)
    wsys_hp = make_hp(hpix, wsys)

    wsys_normed = normalize(wsys_hp, nobs_w, mask)

    cl_avg[idx] = get_cl(nobs_w, nran_w, mask, selection_fn=wsys_normed, **kw)
    print('.', end='')


# get C_ell from chains -> average
cl_ind = {}
for idx in range(0, wnn['weight'].shape[1]):

    wsys = wnn['weight'][:, idx]
    wsys_hp = make_hp(hpix, wsys)

    wsys_normed = normalize(wsys_hp, nobs_w, mask)

    cl_ind[idx] = get_cl(nobs_w, nran_w, mask, selection_fn=wsys_normed, **kw)
    print('.', end='')


# In[13]:


np.savez(f'{cats_dir}1.0/measurements/cl/cl_ngc_main_known_512_nchains.npz',
        **{'cl_avg':cl_avg, 'cl_indv':cl_ind, 'cl_def':cl_def, 'cl_old':cl_old})
