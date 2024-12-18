
import healpy as hp
import numpy as np

from lssutils import CurrentMPIComm
from lssutils.utils import make_overdensity, KMeansJackknifes


__all__ = ['AnaFast', 'get_cl']



def gauleg(ndeg, a=-1.0, b=1.0):
    '''
       Gauss-Legendre (default interval is [-1, 1])
    '''
    x, w = np.polynomial.legendre.leggauss(ndeg)
    # Translate x values from the interval [-1, 1] to [a, b]
    t = 0.5*(x + 1)*(b - a) + a
    w *= 0.5*(b - a)
    return t, w

def xi2cl(x, w, xi, nlmax):
    '''
        calculates Cell from omega
    '''
    cl  = []
    xiw = xi*w
    for i in range(nlmax+1):
        Pl    = np.polynomial.Legendre.basis(i)(x)
        cl.append((xiw * Pl).sum())
        
    cl = 2.*np.pi*np.array(cl)
    return cl

def cl2xi(cell, costheta):
    '''
        calculates omega from Cell at Cos(theta)
    '''
    ell = np.arange(cell.size)
    coef = (2*ell+1) * cell / (4.*np.pi)
    y = np.polynomial.legendre.legval(costheta, c=coef, tensor=False)
    return y




@CurrentMPIComm.enable
def get_cl(ngal, nran, mask, selection_fn=None,
           systematics=None, njack=20, nran_bar=None, lmax=None, 
           cache_jackknifes=True, do_jack_sys=False, comm=None,
           cross_only=False, is_delta=False):
    
    # initialize AnaFast
    af_kw = {'njack':njack, 'lmax':lmax}
    
    if do_jack_sys:
        af_kws = {'njack':njack, 'lmax':lmax}
    else:
        af_kws = {'njack':0, 'lmax':lmax}
        
    af = AnaFast(cache_jackknifes=cache_jackknifes)

    if comm.rank==0:        
        nran_exp = nran_bar if nran_bar is not None else nran[mask].mean()
        weight = nran / nran_exp

        if not is_delta:
            delta = make_overdensity(ngal, nran, mask, selection_fn=selection_fn)
        else:
            delta = ngal*1.0
    else:
        delta = None
        weight = None

    delta = comm.bcast(delta, root=0)
    weight = comm.bcast(weight, root=0)

    if comm.rank==0:
        #--- auto power spectrum
        cl_gg = af(delta, weight, mask, **af_kw)
        cl_gg['shotnoise'] = get_shotnoise(ngal, weight, mask)

    if systematics is not None:
        #if comm.rank==0:
        #    print('C_s,g', end=' ')
        #    print(f'{systematics.shape}')

        # split across processes
        chunk_size = systematics.shape[1]//comm.size
        if systematics.shape[1]%comm.size!=0:
            chunk_size +=1
        start = chunk_size*comm.rank
        end = np.minimum(start+chunk_size, systematics.shape[1])
        #print(start, end)

        cl_ss_list = []
        cl_sg_list = []
        for i in range(start, end):
            systematic_i = make_overdensity(systematics[:, i],
                                        weight, mask, is_sys=True)
            if not cross_only:
                cl_ss_list.append(af(systematic_i, weight, mask, **af_kws))
            cl_sg_list.append(af(delta, weight, mask,
                        map2=systematic_i, weight2=weight, mask2=mask, **af_kws))

        comm.Barrier()

        if not cross_only:
            cl_ss_list = comm.gather(cl_ss_list, root=0)
        cl_sg_list = comm.gather(cl_sg_list, root=0)

        if comm.rank==0:
            if not cross_only:
                cl_ss_list = [cl_j for cl_i in cl_ss_list for cl_j in cl_i if len(cl_i)!=0]
            else:
                cl_ss_list = np.nan
            cl_sg_list = [cl_j for cl_i in cl_sg_list for cl_j in cl_i if len(cl_i)!=0]
            output = {
                'cl_gg':cl_gg,
                'cl_sg':cl_sg_list,
                'cl_ss':cl_ss_list
            }
            return output

    else:
        if comm.rank==0:
            output = {
                'cl_gg':cl_gg
            }
            return output

def get_shotnoise(ngal, weight, mask, estimator='nbar'):
    '''
        ngal is the weighted number of galaxies
    '''
    nside = hp.get_nside(ngal)
    area = hp.nside2pixarea(nside)*weight[mask].sum()
    nbar = ngal[mask].sum()/area
    if estimator=='nbar':
        return 1./nbar
    elif estimator=='signbar':
        return np.std(ngal[mask])/nbar
    else:
        raise ValueError(f'{estimator} either nbar or signbar')
        
class AnaFast:
    '''

    examples
    --------
    # make mock data
    ell_true = np.arange(1024)
    cl_true = 1.e-6*(0.001+ell_true)/(1+ell_true*ell_true)
    map1 = hp.synfast(cl_true, nside=256, new=True)

    mask1 = np.ones_like(map1, '?')
    weight1 = np.ones_like(map1)
    maskp5 = mask1.copy()
    maskp5[mask1.size//2:] = False

    # compute
    af = AnaFast()
    cl_obs_jk = af(map1, weight1, mask1, njack=20)
    cl_obs_half_jk = af(map1, weight1, maskp5, njack=20)


    # visualize
    for cl_i in [cl_obs_jk, cl_obs_half_jk]:

        plt.errorbar(cl_i['l'], cl_i['cl'], yerr=cl_i['cl_error'],
                     marker='.', alpha=0.5, ls='none', capsize=2)

    plt.plot(ell_true, cl_true, 'grey')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    '''
    def __init__(self, cache_jackknifes=False):
        self.cache_jackknifes = cache_jackknifes


    def __call__(self, map1, weight1, mask1,
                 map2=None, weight2=None, mask2=None,
                 lmax=None, njack=0):


        #print(f'lmax: {lmax}')   # if none, lmax = 3N_side-1
        #print(f'njack: {njack}')

        if njack == 0:
            cl_auto = self.run(map1, weight1, mask1,
                                 map2=map2, weight2=weight2, mask2=mask2,
                                 lmax=lmax)

            output = {
                'l':np.arange(cl_auto.size),
                'cl':cl_auto,
                'cl_error':np.nan,
                'njack':njack,
                'lmax':cl_auto.size
            }

        elif njack > 1:
            cl_jackknifes, cl_std = self.run_w_jack(map1, weight1, mask1,
                                                     map2=map2, weight2=weight2, mask2=mask2,
                                                     lmax=lmax, njack=njack)

            output = {
                'l':np.arange(cl_jackknifes[-1].size),
                'cl':cl_jackknifes[-1],
                'cl_error':cl_std,
                'njack':njack,
                'lmax':cl_jackknifes[-1].size,
                'cl_jackknifes':cl_jackknifes
            }
        else:
            raise RuntimeError(f'njack: {njack} must be > 1 or == 0')

        return output

    def run_w_jack(self, map1, weight1, mask1,
                   map2=None, weight2=None, mask2=None,
                   lmax=None, njack=4, visualize=False, subsample=True):

        #print(f'njack: {njack}')

        #--- split the common mask into N Jackknifes
        mask_common = mask1.copy()
        if mask2 is not None:
            mask_common &= mask2

        if not hasattr(self, 'jk_masks'):
            jk = KMeansJackknifes(mask_common, weight1)
            jk.build_masks(njack)
            jk_masks = jk.masks
            
            if self.cache_jackknifes:
                self.jk_masks = jk_masks
        else:
            assert np.allclose(self.jk_masks[-1], mask_common), 'mask has changed'
            jk_masks = self.jk_masks
            

        #--- compute the mean
        cl_jackknifes = {}
        for j, jk_mask in jk_masks.items():
            cl_jackknifes[j] = self.run(map1, weight1, jk_mask,
                                       map2=map2, weight2=weight2, mask2=jk_mask,
                                       lmax=lmax)

        #--- compute the dispersion
        cl_var = np.zeros_like(cl_jackknifes[-1])
        for i in range(njack):
            res = (cl_jackknifes[-1] - cl_jackknifes[i])
            cl_var += res*res
        cl_var *= (njack-1)/njack
        cl_std = np.sqrt(cl_var)

        return cl_jackknifes, cl_std


    def run(self, map1, weight1, mask1,
            map2=None, weight2=None, mask2=None, lmax=None):

        mask_common = mask1.copy()

        if (map2 is not None) & (weight2 is not None) & (mask2 is not None):
            mask_common &= mask2
            hp_map2 = self.prepare_hpmap(map2, weight2, mask_common)

        else:
            hp_map2 = None

        hp_map1 = self.prepare_hpmap(map1, weight1, mask_common)
        normalization = np.mean(mask_common)

#         map_i  = hp.ma(mask_common.astype('f8'))
#         map_i.mask = np.logical_not(mask_common)
#         clmask = hp.anafast(map_i.filled())
#         sf = ((2*np.arange(clmask.size)+1)*clmask).sum()/(4.*np.pi)

        #print(sf, normalization)
        return hp.anafast(hp_map1, map2=hp_map2, lmax=lmax) / normalization


    def prepare_hpmap(self, map1, weight1, mask1):

        weight1_norm = np.mean(weight1[mask1])

        hp_map1 = hp.ma(map1 * weight1) / weight1_norm
        hp_map1.mask = np.logical_not(mask1)
        hp_map1 = hp_map1.filled()

        return hp_map1
