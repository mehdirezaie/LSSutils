
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
        self.cl_mask = cl_['cl']
                
        self.Pl = []
        for ell in self.ell_ob:
            self.Pl.append(np.polynomial.Legendre.basis(ell)(self.x))
                

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