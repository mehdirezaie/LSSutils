"""
    Sample from the posterior distribution of linear models

"""
import os
import numpy as np
import fitsio as ft
import healpy as hp

def model(x, theta):
    """ Linear model """
    return x.dot(theta[1:]) + theta[0]


class Chains:
    def __init__(self, filename, plot=False):    
        chains_ = np.load(filename, allow_pickle=True)
        self.chains = chains_['chain']
        self.stats = {'x':chains_['x'], 'y':chains_['y']}
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
np.random.seed(85)
nside = 1024     #
npoints = 240000 # 600x400
nwindows = 1000  # 
regions = ['bmzls', 'ndecals', 'sdecals']
axes = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]

hpix = {}
features = {}
params = {}
stats = {}

for region in regions:
    df = ft.read(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/v3/nelg_features_{region}_{nside}.fits')    
    ch1 = Chains(f'/fs/ess/PHS0336/data/tanveer/dr9/v3/elg_linear/mcmc_{region}_{nside}.npz')
    params[region] = ch1.get_sample(skip_rows=400)
    stats[region] = ch1.stats
    features[region] = (df['features'][:, axes] - ch1.stats['x'][0]) / ch1.stats['x'][1]
    hpix[region] = df['hpix']
    
ix = np.random.choice(np.arange(npoints), size=nwindows, replace=False)
for j, i in enumerate(ix):
    window_i = np.zeros(12*nside*nside)
    for region in regions:
        n_mu, n_std = stats[region]['y']
        wind_ = n_std*model(features[region], params[region][i, :]) + n_mu
        window_i[hpix[region]] += wind_
        #if j == 0:
        #    hp.mollview(window_i, rot=-90, cmap=plt.cm.jet, max=10)
        #    plt.show()
        
    print('.', end='')
    if (j+1)%100==0:
        print()
    output_path = f'/fs/ess/PHS0336/data/tanveer/dr9/v3/elg_linear/windows/linwindow_{j}.hp{nside}.fits'
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
         os.makedirs(output_dir)
    hp.write_map(output_path, window_i, dtype=np.float64)

print("done!!!")
