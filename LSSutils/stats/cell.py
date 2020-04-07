import numpy  as np
import healpy as hp
from time import time
from LSSutils.utils import split_jackknife_new


def jkmasks(mask, weight, njack=4):
    '''
    Function that makes Jackknife masks
    
    
    example
    --------
    >>> jkmasks_dic = jkmasks(mask_1, weight_1)
    >>> for k in jkmasks_dic:
    ...     hp.mollview(jkmasks_dic[k], title=k)
    
    '''
    assert mask.size == weight.size
    
    nside = hp.get_nside(mask)
    
    # --- make jackknife samples
    hpix_jk,_ = split_jackknife_new(np.argwhere(mask).flatten(),
                                    weight[mask], 
                                    njack=njack)
    masks_dic = {-1:mask}
    for jackid in range(njack):
        mask_tmp = mask.copy()
        mask_tmp[hpix_jk[jackid]] = False
        masks_dic[jackid] = mask_tmp
        
    return masks_dic


class AnaFast:
    '''
    
    examples
    --------
    # credit: Mehdi Rezaie
    >>> #--- create mock C_ell
    >>> ell_true = np.arange(1024)
    >>> cl_true = 1.e-6*(0.001+ell_true)/(1+ell_true*ell_true)
    >>> map1 = hp.synfast(cl_true, nside=256, new=True)

    >>> #--- create a mock window, e.g., half the sky
    >>> mask_1 = np.ones_like(map1, '?')
    >>> weight_1 = np.ones_like(map1)
    >>> mask_p5 = mask_1.copy()
    >>> mask_p5[mask_1.size//2:] = False

    >>> #--- run AnaFast with Jackknife
    >>> af = AnaFast()
    >>> af(map1, weight_1, mask_p5, lmax=512, njack=20)

    >>> #--- plot
    >>> fig, ax = plt.subplots()
    >>> ax.loglog(af.output['ell'], af.output['Cell'], 
    ...           c='crimson', label='Measured')
    >>> for cli in af.jkcells:
    ...    ax.loglog(af.jkcells[cli], color='grey', zorder=-1, alpha=0.2)

    >>> ax.loglog(cl_true, 'k', label='True')
    >>> ax.legend(fontsize=12)
    >>> ax.grid(True, ls=':', which='both', alpha=0.2)
    >>> ax.set(xlabel=r'$\ell$', ylabel=r'C$_\ell$', ylim=(1.0e-9, 2.0e-6))
    
    '''
    def __init__(self):          
        pass
    
    def __call__(self, map1, weight1, mask1, 
                 map2=None, weight2=None, mask2=None, 
                 lmax=None, njack=0):
        
        print(f'lmax: {lmax}')
        print(f'njack: {njack}')
        
        if njack == 0:
            cl_auto = self.run(map1, weight1, mask1, 
                                 map2=map2, weight2=weight2, mask2=mask2, 
                                 lmax=lmax)
            
            self.output = {'ell':np.arange(cl_auto.size),
                           'Cell':cl_auto,
                           'Cell_error':np.nan,
                           'njack':njack, 
                           'lmax':lmax}
            
        elif njack > 1:            
            self.run_w_jack(map1, weight1, mask1, 
                             map2=map2, weight2=weight2, mask2=mask2, 
                             lmax=lmax, njack=njack)
            
            self.output = {'ell':np.arange(self.jkcells[-1].size),
                           'Cell':self.jkcells[-1],
                          'Cell_error':self.clstd,
                          'njack':njack,
                          'lmax':lmax}
        else:                 
            raise RuntimeError(f'njack: {njack} must be > 1 or == 0')
        
        
    
    def run_w_jack(self, map1, weight1, mask1, 
                   map2=None, weight2=None, mask2=None, 
                   lmax=None, njack=4):
        
        #print(f'njack: {njack}')
        
        #--- split the common mask into N Jackknifes        
        mask_common = mask1.copy()        
        if mask2 is not None:            
            mask_common &= mask2   
        self.jkmasks_dic = jkmasks(mask_common, weight1, njack=njack)
        
        #--- compute the mean
        self.jkcells = {}
        for k in self.jkmasks_dic:
            t0 = time()
            self.jkcells[k] = self.run(map1, weight1, self.jkmasks_dic[k],
                                  map2=map2, weight2=weight2, mask2=self.jkmasks_dic[k],
                                  lmax=lmax)
            print(f'{k}, {time()-t0:.2f} secs')
        
        #--- compute the dispersion
        clvar = np.zeros_like(self.jkcells[-1])
        for i in range(njack):
            res = (self.jkcells[-1] - self.jkcells[i])
            clvar += res*res
        clvar *= (njack-1)/njack        
        self.clstd = np.sqrt(clvar)
        
        
    def run(self, map1, weight1, mask1, 
            map2=None, weight2=None, mask2=None, lmax=None):
        
        mask_common = mask1.copy()

        if (map2 is not None) & (weight2 is not None) & (mask2 is not None):
            
            mask_common &= mask2   
            weight2 /= np.mean(weight2[mask_common])
            
            hp_map2 = hp.ma(map2 * weight2)
            hp_map2.mask = np.logical_not(mask_common)
            hp_map2 = hp_map2.filled()

        else:
            hp_map2 = None
        
        weight1 /= np.mean(weight1[mask_common])
        
        hp_map1 = hp.ma(map1 * weight1)
        hp_map1.mask = np.logical_not(mask_common)
        hp_map1 = hp_map1.filled()        
        normalization = np.mean(mask_common)
        
        return hp.anafast(hp_map1, map2=hp_map2, lmax=lmax)/normalization        