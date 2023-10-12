
import numpy as np

from scipy.interpolate import interp1d
from lssutils.stats.cl import AnaFast, gauleg

from scipy.optimize import curve_fit

def model(l, *p):
    return p[0]*np.log10(l)+p[1]

def smooth_cl(cl_wind):
    el_p = 2
    el = np.arange(cl_wind.size)
    is_small = el < el_p

    lmin = 10
    lmax = 200 #2*nside-1
    x = np.arange(lmin, lmax+1)
    y = cl_wind[lmin:lmax+1]
    res = curve_fit(model, x, np.log10(y), p0=[1, 1])

    cl_window = np.zeros(el.size)
    cl_window[:el_p] = cl_wind[:el_p]
    cl_window[~is_small] = 10**model(el[~is_small], *res[0])
    return cl_window
    
    
    
class WindowSHT:
    
    def __init__(self, weight, mask, ell_ob, ngauss=2**12, smooth_window=False):
        af = AnaFast()
        cl_ = af(mask*1.0, weight, mask)        
        
        xi_zero = (cl_['cl']*(2.*cl_['l']+1.)).sum() / (4.*np.pi)

        self.ell_ob = ell_ob
        self.twopi = 2.*np.pi

        self.x, self.w = gauleg(ngauss)
        self.xi_mask = self.cl2xi(cl_['l'], cl_['cl']) / xi_zero
        self.cl_mask = cl_['cl']
        if smooth_window:
            self.cl_mask = smooth_cl(self.cl_mask*1.0)
        
        self.xi_sht = interp1d(self.x, self.xi_mask)
                
        self.Pl = []
        for ell in self.ell_ob:
            self.Pl.append(np.polynomial.Legendre.basis(ell)(self.x))
    
    def read_rr(self, rr_file, ntot, npix):
        
        area = ntot / npix
        
        raw_data = np.load(rr_file, allow_pickle=True)
        sep = raw_data[0][::-1]           # in radians
        rr_counts = raw_data[1][::-1]*2.0 # paircount uses symmetry

        sep_mid = 0.5*(sep[1:]+sep[:-1])
        dsep = np.diff(sep)        
        window = rr_counts / (dsep*np.sin(sep_mid)) * (2./(npix*npix*area))
        
        self.sep_mid = sep_mid
        self.xi_rr = interp1d(np.cos(sep_mid), window, fill_value=0, bounds_error=False)

        theta_p = 10.0
        is_small = self.x > np.cos(np.deg2rad(theta_p))
        self.xi_mask_smooth = np.zeros_like(self.x)
        self.xi_mask_smooth[is_small] = self.xi_sht(self.x[is_small])
        self.xi_mask_smooth[~is_small] = self.xi_rr(self.x[~is_small])


        
    def convolve(self, el_model, cl_model, with_smooth=False):
        
        xi_th = self.cl2xi(el_model, cl_model)
        if with_smooth:
            xi_thw = xi_th * self.xi_mask_smooth
        else:
            xi_thw = xi_th * self.xi_mask
            
        cl_thw = self.xi2cl(xi_thw)        
        
        return cl_thw
    
    def apply_ic(self, cl_model):
        lmax = len(cl_model)
        return cl_model - cl_model[0]*(self.cl_mask[:lmax]/self.cl_mask[0])

    def xi2cl(self, xi):
        '''
            calculates Cell from omega
        '''
        cl  = []
        xiw = xi*self.w
        for i in range(len(self.Pl)):
            cl.append((xiw * self.Pl[i]).sum())
            
        return self.twopi*np.array(cl)

    def cl2xi(self, ell, cell):
        '''
            calculates omega from Cell at Cos(theta)
        '''
        twol4pi = (2.*ell+1.)/(4.*np.pi)
        return np.polynomial.legendre.legval(self.x, c=twol4pi*cell, tensor=False)
