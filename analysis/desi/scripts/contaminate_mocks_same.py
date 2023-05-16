"""
    Sample from the posterior distribution of linear models
    and contaminate lognormal mocks

"""
import sys
import os
import numpy as np
import fitsio as ft
import healpy as hp
from glob import glob


def modelp(x, theta):
    """ Linear model Poisson """
    u = x.dot(theta[1:]) + theta[0]    
    is_high = u > 20
    ret = u*1
    ret[~is_high] = np.log(1.+np.exp(u[~is_high]))
    return ret


class Chains:
    def __init__(self, filename, plot=False):    
        chains_ = np.load(filename, allow_pickle=True)
        self.chains = chains_['chain']
        self.stats = {'x':chains_['x']}#, 'y':chains_['y']}
        print(self.chains.shape)
        self.ndim = self.chains.shape[-1]
        
    def get_sample(self, skip_rows=200):
        return self.chains[skip_rows:, :, :].reshape(-1, self.ndim)


#--- read mcmc chains
np.random.seed(85)
nside = 256
model = 'linp'
tag_d = '0.57.0'
regions = ['ndecalsc', 'sdecalsc']
axes = [0, 4, 7] # maps for known1
root_dir = '/fs/ess/PHS0336/data/rongpu/imaging_sys'
target = 'lrg'
maps = 'known1'
fnl = 'zero'

input_path = '/fs/ess/PHS0336/data/lognormal/v3/hpmaps/'
path_mocks = os.path.join(input_path, f'lrghp-{fnl}-103-f1z1.fits')

# read imaging
hpix = {}
features = {}
params = {}
stats = {}
for region in regions:
    input_path=f'{root_dir}/tables/{tag_d}/n{target}_features_{region}_{nside}.fits'
    chain_path=f'{root_dir}/regression/{tag_d}/{model}_{target}_sdecalsc_{nside}_{maps}/mcmc_sdecalsc_{maps}.npz'
    df = ft.read(input_path)    
    ch1 = Chains(chain_path)

    params[region] = ch1.get_sample(skip_rows=1000)
    stats[region] = ch1.stats
    features[region] = (df['features'][:, axes] - ch1.stats['x'][0]) / ch1.stats['x'][1]
    hpix[region] = df['hpix']

for i,mock_i in enumerate([path_mocks, ]):

    window_i = np.zeros(12*nside*nside)
    count_i = np.zeros_like(window_i)
    for region in regions:
        wind_ = modelp(features[region], params[region][i, :])
        window_i[hpix[region]] += wind_
        count_i[hpix[region]] += 1.0
        
    is_good = count_i > 0.0
    window_i[is_good] = window_i[is_good] / count_i[is_good]
    window_mean       = window_i[is_good].mean()
    window_i[is_good] = window_i[is_good]/window_mean

    nmock_i = hp.read_map(mock_i, verbose=False)
    cnmock_i = nmock_i * window_i
    cnmock_i[~is_good] = hp.UNSEEN

    cmock_i = mock_i.replace(fnl, 'csame'+fnl)
    hp.write_map(cmock_i, cnmock_i, dtype=np.float64, fits_IDL=False, overwrite=True)
    print(f"done writing {cmock_i}")
