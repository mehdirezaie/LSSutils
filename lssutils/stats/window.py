
import numpy as np

from lssutils.stats.cl import AnaFast, gauleg



class WindowSHT:
    
    def __init__(self, weight, mask, ell_ob, ngauss=2**12):
        af = AnaFast()
        cl_ = af(mask*1.0, weight, mask)        
        
        xi_zero = (cl_['cl']*(2.*cl_['l']+1.)).sum() / (4.*np.pi)

        self.ell_ob = ell_ob
        self.twopi = 2.*np.pi

        self.x, self.w = gauleg(ngauss)
        self.xi_mask = self.cl2xi(cl_['l'], cl_['cl']) / xi_zero
                
        self.Pl = []
        for ell in self.ell_ob:
            self.Pl.append(np.polynomial.Legendre.basis(ell)(self.x))
            
            
#     def read_rr(self, rr_file, ntot, npix):
        
#         area = ntot / npix
        
#         raw_data = np.load(rr_file, allow_pickle=True)
#         sep = raw_data[0][::-1]           # in radians
#         rr_counts = raw_data[1][::-1]*2.0 # paircount uses symmetry

#         sep_mid = 0.5*(sep[1:]+sep[:-1])
#         dsep = np.diff(sep)        
#         window = rr_counts / (dsep*np.sin(sep_mid)) * (2./(npix*npix*area))   
#         self.window_spl = interp1d(np.cos(sep_mid), window, fill_value="extrapolate")

#         small_scale = self.x < -0.5735764363510458 # i.e., cos(x) < cos(125)
#         self.xi_mask = self.xi_mask_ * 1.0
#         self.xi_mask[small_scale] = self.window_spl(self.x[small_scale])
    

    def convolve(self, el_model, cl_model):
        
        xi_th = self.cl2xi(el_model, cl_model)
        xi_thw = xi_th * self.xi_mask
        cl_thw = self.xi2cl(xi_thw)        
        
        return cl_thw

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