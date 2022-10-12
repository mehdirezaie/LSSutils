
import numpy as np


def read_chain(chain_filename, skip=5000, ndim=3, ifnl=0, iscale=[2, ]):
    
    ch_ = np.load(chain_filename)
    
    chain = ch_['chain']
    assert chain.shape[-1] == ndim
    
    map_bf = ch_['best_fit'][ifnl]   
    #map_chain = ch_['chain'].reshape(-1, ndim)[ch_['log_prob'].argmax()][ifnl]
    
    sample = chain[skip:, :, :].flatten().reshape(-1, ndim)
    for icol in iscale:
        sample[:, icol] *= 1.0e7
    
    mean_chain = sample[:, ifnl].mean()
    vmin2, vmin1, vmax1, vmax2 = np.percentile(sample[:, ifnl], [2.5, 16, 84, 97.5], axis=ifnl)
    
    return [map_bf, mean_chain, vmin1, vmax1, vmin2, vmax2], sample


def read_window(region, nside=256):
    """ Return Window, Mask, Noise
    """
    from .utils import make_hp
    import fitsio as ft
    import healpy as hp
    
    # read survey geometry
    if region in ['bmzls', 'ndecals', 'sdecals']:        
        data_path = '/fs/ess/PHS0336/data/'    
        dt = ft.read(f'{data_path}/rongpu/imaging_sys/tables/v3/nlrg_features_{region}_256.fits')
        mask  = make_hp(256, dt['hpix'], 1.0) > 0.5
        if nside != 256:
            mask   = hp.ud_grade(mask, nside)        
    else:
        mask = np.ones(12*nside*nside)
        
        
    weight = mask * 1.0
    print(f'read window')
    
    return weight, mask

def read_clmocks(region, method, plot_cov=False, bins=None, return_cov=False):
    from .utils import histogram_cell
    from glob import glob
    import matplotlib.pyplot as plt

    data_path = '/fs/ess/PHS0336/data/'

    if bins is None:
        bins = np.array([2*(i+1) for i in range(10)] + [2**i for i in range(5, 9)])
    
    # compute covariance
    cl_list = []
    for i in range(101, 1001):    
        cl_i = f'{data_path}lognormal/v0/clustering/clmock_{i}_lrg_{region}_256_noweight.npy'        
        cl_ = np.load(cl_i, allow_pickle=True).item()['cl_gg']['cl']
        lb_, clb_ = histogram_cell(np.arange(cl_.size), cl_, bins=bins)
        cl_list.append(clb_)
        print('.', end='')
        
    nmocks, nbins = np.array(cl_list).shape
    hf = (nmocks - 1.0)/(nmocks - nbins - 2.0)
    cl_cov = np.cov(cl_list, rowvar=False)*hf / nmocks
    inv_cov = np.linalg.inv(cl_cov)
    print(f'Hartlap with #mocks ({nmocks}) and #bins ({nbins}): {hf:.2f}' )
    print('bins:', lb_)
    

    # compute mean power spectrum
    cl_list = []
    for i in range(1, 101):    
        cl_i = f'{data_path}lognormal/v0/clustering/clmock_{i}_lrg_{region}_256_{method}.npy'        
        cl_ = np.load(cl_i, allow_pickle=True).item()['cl_gg']['cl']
        lb_, clb_ = histogram_cell(np.arange(cl_.size), cl_, bins=bins)        
        cl_list.append(clb_)
        print('.', end='')
        
    cl_mean = np.mean(cl_list, axis=0)

    if plot_cov:
        vmin, vmax = np.percentile(cl_cov, [5, 95])
        lim = np.minimum(abs(vmin), abs(vmax))        
        plt.imshow(cl_cov, origin='lower', vmin=-1.*lim, vmax=lim, cmap=plt.cm.bwr)
        plt.show()
        plt.imshow(cl_cov.dot(inv_cov), origin='lower')
        plt.show()
        
    print(cl_mean.shape, cl_cov.shape, len(cl_list))
    ret = (lb_, cl_mean, inv_cov)
    if return_cov:
        ret += (cl_cov,)
        
    return ret  



def read_nnbar(filename):
    
    d_i = np.load(filename, allow_pickle=True)    
    err_mat = []    
    for i, d_ij in enumerate(d_i):
        err_mat.append(d_ij['nnbar']-1.0)
    
    return np.array(err_mat).flatten()

def read_nbmocks(list_nbars):
    
    err_mat = []    
    for nbar_i in list_nbars:
        
        err_i  = read_nnbar(nbar_i)
        err_mat.append(err_i)
        print('.', end='')

    err_mat = np.array(err_mat)
    print(err_mat.shape)
    return err_mat


def read_clx(fn, cl_ss=None, lmin=0):
    from .utils import histogram_cell
    #--- 
    cl = np.load(fn, allow_pickle=True).item()
    cl_x = []
    
    if cl_ss is None:
        cl_ss = []
        read_clss = True
    else:
        read_clss = False

    nsys = len(cl['cl_sg'])
    #lbins = np.arange(1, len(cl['cl_sg'][0]['cl']), 10)
    lbins = np.arange(1, 101, 10)
    #print(nsys)    
    for i in range(nsys):    
        l_, cl_sg_ = histogram_cell(cl['cl_sg'][i]['cl'], bins=lbins)
        if read_clss:
            _, cl_ss_ = histogram_cell(cl['cl_ss'][i]['cl'], bins=lbins)
            cl_ss.append(cl_ss_)
        else:
            cl_ss_ = cl_ss[i] 
            
        cl_x.append(cl_sg_[lmin:]*cl_sg_[lmin:]/cl_ss_[lmin:])    

    #print(l_[:lmax])
    return l_, np.array(cl_x).flatten(), cl_ss


def read_clxmocks(mocks, cl_ss, lmin=0):    
    #--- mocks
    clx = []
    for mock_i in mocks:
        clx_ = read_clx(mock_i, cl_ss=cl_ss, lmin=lmin)[1]   
        clx.append(clx_) 
        print('.', end='')
        
    err_tot = np.array(clx)
    return err_tot


def readnbodykit(filename):
    with open(filename, 'r') as infile:
        lines = infile.readlines()

        shotnoise = None
        values = []
        for line in lines:
            if '#shotnoise' in line:
                shotnoise = float(line.split(':')[-1])
                #print(f'shotnoise {shotnoise}')
            else:
                if line.startswith('#'):
                    continue
                else:
                    strings = line.split(' ')
                    values.append([float(s) for s in strings])
        if values != []:
            values = np.array(values)
        else:
            raise RuntimeError('reading failed')
    return values, shotnoise
