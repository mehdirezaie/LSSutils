"""
    Sample from the posterior distribution of linear models

"""
import sys
import os
import numpy as np
import fitsio as ft
import healpy as hp

def model(x, theta):
    """ Linear model """
    return x.dot(theta[1:]) + theta[0]

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
        #if plot:
        #    fg, ax = plt.subplots(nrows=12, figsize=(8, 12*1), sharex=True)#, sharey=True)
        #    ax = ax.flatten()
        #    #ax[0].set_ylim(-.5, .5)
        #    for i, ix in enumerate(range(12)): #[0, 1, 2, 3, 5]):
        #        for j in range(400):
        #            ax[i].plot(self.chains[:, j, ix])
        #        ax[i].axhline(0.0, ls=':')    
        #    fg.show()
        
    def get_sample(self, skip_rows=200):
        return self.chains[skip_rows:, :, :].reshape(-1, self.ndim)


#--- read mcmc chains
from argparse import ArgumentParser
ap = ArgumentParser(description='Linear Regression')
ap.add_argument('-o', '--output_path', required=True)
ap.add_argument('-m', '--maps', required=True)
ap.add_argument('-ax', '--axes', nargs='*', type=int) 
ns = ap.parse_args()



np.random.seed(85)
nside = 256
model = 'linp'
tag_d = '0.57.0'
regions = ['bmzls', 'ndecals', 'sdecals']
axes = ns.axes #[0, 2, 3, 4, 5, 6, 7, 8]
root_dir = '/fs/ess/PHS0336/data/rongpu/imaging_sys'
target = 'lrg'
maps = ns.maps #'all'
output_path = ns.output_path #f'{root_dir}/regression/{tag_d}/{model}_{target}_{maps}.hp{nside}.fits'



hpix = {}
features = {}
params = {}
stats = {}

for region in regions:
    input_path=f'{root_dir}/tables/{tag_d}/n{target}_features_{region}_{nside}.fits'
    chain_path=f'{root_dir}/regression/{tag_d}/{model}_{target}_{region}_{nside}_{maps}/mcmc_{region}_{maps}.npz'
    df = ft.read(input_path)    
    ch1 = Chains(chain_path)

    params[region] = ch1.get_sample(skip_rows=1000).mean(axis=0)
    stats[region] = ch1.stats
    features[region] = (df['features'][:, axes] - ch1.stats['x'][0]) / ch1.stats['x'][1]
    hpix[region] = df['hpix']


window_i = np.zeros(12*nside*nside)
count_i = np.zeros_like(window_i)
for region in regions:
    wind_ = modelp(features[region], params[region])
    window_i[hpix[region]] += wind_
    count_i[hpix[region]] += 1.0
    
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
     os.makedirs(output_dir)

is_good = count_i > 0.0
window_i[is_good] = window_i[is_good] / count_i[is_good]
window_i[~is_good] = hp.UNSEEN
hp.write_map(output_path, window_i, dtype=np.float64, fits_IDL=False, overwrite=True)
print(f"done writing {output_path}")
