'''
    code that calculates the auto/cross correlation functions
    using healpix based estimator
    
    created 4/3/2018 - Mehdi Rezaie
'''
import os
import sys
sys.path.append('/global/homes/m/mehdi/github/DESILSS')
import xi
import numpy  as np
import healpy as hp
import fitsio as ft
from time import time
from argparse import ArgumentParser


# command arguments
ap = ArgumentParser(description='XI routine: healpix based xi estimator')
ap.add_argument('--galmap')
ap.add_argument('--ranmap')
ap.add_argument('--mask')
ap.add_argument('--njack',  default=20,  type=int)
ap.add_argument('--nside',  default=256, type=int)
ap.add_argument('--oudir',  default='./')
ap.add_argument('--sysmap', default='none') # for cross-correlation
ap.add_argument('--ouname', default='xi-eboss-dr5')
ap.add_argument('--selection', default='none')
ns = ap.parse_args()


log = 'running the healpix based paircount XI \n'
log += 'njack : {}   nside : {}\n'.format(ns.njack, ns.nside) 
# check if output directory is there
if not os.path.exists(ns.oudir):
    log  += 'creating the directory {} ...\n'.format(ns.oudir)
    os.makedirs(ns.oudir)

    
# check if selection function is given
if not ns.selection in ['none', 'None', 'NONE']:
    log += 'selection function : {}\n'.format(ns.selection)
    select_fun_i = hp.read_map(ns.selection)
    select_fun   = xi.check_nside(select_fun_i, ns.nside) # check nside
else:
    log += 'uniform selection function is used!!!\n'
    select_fun = np.ones(12*ns.nside**2)                  # uniform selection mask

    
# if a systematic is given, 
# it computes the cross correlation
if not ns.sysmap in ['none', 'None', 'NONE']:
    log += 'computing the cross-correlation against {}\n'.format(ns.sysmap)
    sysm_i = hp.read_map(ns.sysmap)
    sysm   = xi.check_nside(sysm_i, ns.nside)
else:
    log += 'computing the auto correlation\n'
    sysm   = None


# read galaxy, random and mask maps
galm_i = hp.read_map(ns.galmap)
galm   = xi.check_nside(galm_i, ns.nside)
ranm_i = hp.read_map(ns.ranmap)
ranm   = xi.check_nside(ranm_i, ns.nside)
mask_i = hp.read_map(ns.mask)
mask   = xi.check_nside(mask_i, ns.nside).astype('bool') # should be boolean

log += 'galaxy hp map : {}\n'.format(ns.galmap)
log += 'random hp map : {}\n'.format(ns.ranmap)
log += 'mask   hp map : {}\n'.format(ns.mask)
# run and save
path = ns.oudir + ns.ouname + '_nside_' + str(ns.nside) + '_njack_' + str(ns.njack)
log += 'output under {}'.format(path)
print(log)
xi.run_XI(path, galm, ranm, select_fun, mask, sysm=sysm, njack=ns.njack)
