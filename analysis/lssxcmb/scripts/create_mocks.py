"""
    Create Gaussian Mocks

"""
import numpy as np
import healpy as hp


def gen_mock(cls_th, cls_shot_noise, seed=42):

    #parameters
    NSIDE = 1024
    LMAX = 2*NSIDE-1

    kw = dict(nside=NSIDE, lmax=LMAX, pol=False, verbose=False)

    #generate overdensity signal mock
    np.random.seed(seed)
    delta_g = hp.synfast(cls_th, **kw) 

    #generate noise mock
    np.random.seed(2*seed) #random different seed for noise
    noise_g = hp.synfast(cls_shot_noise, **kw)

    return delta_g+noise_g



debug = False
np.random.seed(85)
nmocks = 1000
seeds = np.random.randint(0, 2**30-1, size=nmocks)
assert np.unique(seeds).size == nmocks

# to match the model power spectrum to the observed C_ell, we multiply 
# Cell by 0.48 and surface density by 4/3
cls_th = np.load('/home/mehdi/data/tanveer/cl_th.npy', allow_pickle=True)
cls_th = cls_th*0.48

# noise parameters
nbar_sqdeg = 2400./0.75 # number density per deg^2 
nbar_sr = (np.pi/180)**(-2) * nbar_sqdeg #conversion factor from sq deg to sr
cls_shot_noise = 1/nbar_sr * np.ones_like(cls_th)




for i, seed in enumerate(seeds):
    
    output = f'/home/mehdi/data/tanveer/mocks/delta_{i}.hp1024.fits' 
    print(seed, output, end=' ')
    
    delta_i = gen_mock(cls_th, cls_shot_noise, seed)
    #print(delta_i.mean())
    hp.write_map(output, delta_i, fits_IDL=False)
    print('done')



if debug:
    cls_obs = hp.anafast(delta_i)
    cls_data = np.load('/home/mehdi/data/rongpu/imaging_sys/clustering/v2/cl_elg_bmzls_256_nn.npy', 
                       allow_pickle=True)

    plt.loglog(cls_obs, 'C0--',
    cls_th+cls_shot_noise, 'C1-')

    plt.plot(cls_data.item()['cl_gg']['cl'], 'C4')

    plt.legend(['Mock', 'Input+Noise', 'DR9 (BMZLS)'])
    plt.ylim(1.0e-8, 1.0e-4)
    plt.ylabel(r'C$_{\ell}$')
    plt.xlabel(r'$\ell$')
    plt.show()

    plt.plot(cls_data.item()['cl_gg']['cl'][:750]/(cls_th+cls_shot_noise)[:750])
    plt.ylim(0., 2.)
    plt.ylabel('DR9 / Input+Noise')
    plt.axhline(1, color='r')
    plt.show()
