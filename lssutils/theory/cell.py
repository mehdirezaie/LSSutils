
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.integrate import romberg, simps

import nbodykit.cosmology as cosmology
from lssutils.extrn.fftlog import fftlog
from lssutils.stats.window import WindowSHT



def dNdz_model(sample='qso', z_low=0.1, kind=1):
    
    if sample=='qso':
        
        dNdz, z = np.loadtxt(f'/fs/ess/PHS0336/data/dr9v0.57.0/p38fnl/RF_g.txt').T
        dNdz_interp = IUS(z, dNdz)
        dNdz_interp.set_smoothing_factor(2.0)

        z_high = 4.0
        nz_low = lambda z:z**kind*dNdz_interp(z_low)/z_low**kind            
        nz_high = lambda z:np.exp(-z**1.65)*dNdz_interp(z_high)/np.exp(-z_high**1.65)

        z_g = np.linspace(0.0, 10.0, 500)
        dNdz_g = dNdz_interp(z_g)
        dNdz_g[(z_g<z_low)] = nz_low(z_g[z_g<z_low])
        dNdz_g[(z_g>z_high)] = nz_high(z_g[z_g>z_high])

        #plt.plot(z, dNdz, 'b--', lw=4, alpha=0.3)
        #plt.plot(z_g, dNdz_g, 'r-')
        # plt.semilogy()
        # plt.ylim(1.0e-8, 1.e3)
        return z_g, dNdz_g
    
    elif sample == 'lrg':
        
        zmin, zmax, dNdz = np.loadtxt('/fs/ess/PHS0336/data/rongpu/sv3_lrg_dndz_denali.txt', 
                                      usecols=(0, 1, 2), unpack=True)        
        zmid = 0.5*(zmin+zmax)
        dNdz_interp = IUS(zmid, dNdz, ext=1)
        dNdz_interp.set_smoothing_factor(2.0)
        z_g = np.linspace(0.0, 5.0, 500)
        dNdz_g = dNdz_interp(z_g)
        
        return z_g, dNdz_g
        
    
    elif sample == 'mock':
        
        z = np.arange(0.0, 3.0, 0.001)
        i_lim = 26.                          # Limiting i-band magnitude
        z0 = 0.0417*i_lim - 0.744
        Ngal = 46. * 100.31 * (i_lim - 25.)            # Normalisation, galaxies/arcmin^2
        pz = 1./(2.*z0) * (z / z0)**2. * np.exp(-z/z0)  # Redshift distribution, p(z)
        dNdz = Ngal * pz                               # Number density distribution
        
        return z, dNdz
    else:
        raise NotImplemented(f'{sample}')

def bias_model_lrg(z):
    """ arxiv.org/abs/2001.06018
    
    
    """
    cste = 1.0 ## 1.4262343145500318 was used to generate mocks. 1.42 (Rongpu suggestion)
    kw_cosmo = dict(h=0.67556, T0_cmb=2.7255, Omega0_b=0.0482754208891869,
                    Omega0_cdm=0.26377065934278865, N_ur=None, m_ncdm=[0.06],
                    P_k_max=10.0, P_z_max=100.0, sigma8=0.8225, gauge='synchronous',
                    n_s=0.9667, nonlinear=False)
    sigma8 = kw_cosmo.pop('sigma8')            
    cosmo = cosmology.Cosmology(**kw_cosmo).match(sigma8=sigma8) 
    Dlin = cosmo.scale_independent_growth_factor       # D(z), normalized to one at z=0
    
    return cste/Dlin(z)
    
    
    
def bias_model_qso(z):
    """
     Bias of quasars Laurent et al. 2017 (1705.04718).
     TODO: Check the range in which this function is valid
    """
    alpha = 0.278
    beta = 2.393
    return alpha * ((1+z)**2 - 6.565) + beta

def init_sample(kind='qso', plot=False):
    
    if kind=='qso':
        z, dNdz = dNdz_model(kind)
        b = bias_model_qso(z)
        
    elif kind=='lrg':
        z, dNdz = dNdz_model(kind)
        b = bias_model_lrg(z)
        
    elif kind=='mock':
        z, dNdz = dNdz_model(kind)
        b = 1.5*np.ones_like(z)
        
    else:
        raise NotImplementedError(f'{kind} not implemented')
        
    if plot:
        fg, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(z, b, 'k-', label='Bias')
        ax2.plot(z, dNdz, 'C1--', label='Redshift Distribution')
        ax1.set(xlabel='z', ylabel='bias(z)')
        ax2.set_ylabel(r'$n(z)~[{\rm arcmin}^{-2}]$', color='C1')
        ax2.spines['right'].set_color('C1')
        ax2.tick_params(axis='both', colors='C1')
        # ax1.legend()
        # ax2.legend(frameon=False)
        # fg.savefig('sample.png', dpi=300, bbox_inches='tight')        
    return z, b, dNdz


class Spectrum:
    """
    Redshift space power spectrum of biased tracers
    """    
    def __init__(self, h=0.67556, T0_cmb=2.7255, Omega0_b=0.0482754208891869,
                 Omega0_cdm=0.26377065934278865, N_ur=None, m_ncdm=[0.06],
                 P_k_max=10.0, P_z_max=100.0, sigma8=0.8225, gauge='synchronous',
                 n_s=0.9667, nonlinear=False):
        
        # input parameters
        kw_cosmo = locals()
        kw_cosmo.pop('self')
        for kw,val in kw_cosmo.items():
            print(f'{kw:10s}: {val}')
            
        sigma8 = kw_cosmo.pop('sigma8')            
        cosmo = cosmology.Cosmology(**kw_cosmo).match(sigma8=sigma8) 
                
        # --- cosmology calculators
        redshift = 0.0
        delta_crit = 1.686      # critical overdensity for collapse from Press-Schechter
        DH = (cosmo.H0/cosmo.C) # in units of h/Mpc, eq. 4 https://arxiv.org/pdf/astro-ph/9905116.pdf
        Omega0_m = cosmo.Omega0_m
        
        k_ = np.logspace(-10, 10, 10000)
        Plin_ = cosmology.LinearPower(cosmo, redshift, transfer='CLASS') # P(k) in (Mpc/h)^3 
        self.Plin = IUS(k_, Plin_(k_), ext=1)
        Tlin_ = cosmology.transfers.CLASS(cosmo, redshift)  # T(k), normalized to one at k=0
        tk_ = Tlin_(k_)
        is_good = np.isfinite(tk_)
        self.Tlin = IUS(k_[is_good], tk_[is_good], ext=1)
        self.Dlin = cosmo.scale_independent_growth_factor       # D(z), normalized to one at z=0
        self.f = cosmo.scale_independent_growth_rate            # dlnD/dlna
        self.Dc = cosmo.comoving_distance              
        self.h = cosmo.efunc
        self.inv_pi = 2.0 / np.pi
                 
        # alpha_fnl: equation 2 of Mueller et al 2017    
        self.alpha_fnl  = 3.0*delta_crit*Omega0_m*(DH**2)
        
    def __call__(self, ell, fnl=0.0, b=1.0, noise=0.0, **kwargs):

        if not self.kernels_ready:
            print('will create windows')
            self.add_kernels(ell, **kwargs)
            
        elif not np.array_equal(ell, self.ell):
            print('will update windows')            
            self.add_kernels(ell, **kwargs)
                       
        return self.run(fnl, b, noise)
        
    def add_tracer(self, z, b, dNdz, p=1.6):
        print(f'p = {p:.1f}')
        # dz/dr
        dNdr_spl = IUS(self.Dc(z), dNdz*self.h(z), ext=1) 
        dNdr_tot = romberg(dNdr_spl, self.Dc(z.min()), self.Dc(z.max()), divmax=1000)
        #print(dNdr_tot)
        
        # bias
        b_spl = IUS(self.Dc(z), b, ext=1)

        # growth
        d_spl = IUS(self.Dc(z), self.Dlin(z), ext=1)

        # growth rate: f = dlnD/dlna
        f_spl = IUS(self.Dc(z), self.f(z), ext=1)

        # prepare kernels
        self.fr_wk = lambda r:r * b_spl(r) * d_spl(r) * (dNdr_spl(r)/dNdr_tot)   # W_ell: r*b*(D(r)/D(0))*dN/dr         
        self.fr_wrk = lambda r:r * f_spl(r) * d_spl(r) * (dNdr_spl(r)/dNdr_tot)  # Wr_ell:r*f*(D(r)/D(0))*dN/dr
        self.fr_wkfnl1 = lambda r: r * b_spl(r) * (dNdr_spl(r)/dNdr_tot) # Wfnl_ell:r*b*dN/dr   
        self.fr_wkfnl2 = lambda r: r * -p * (dNdr_spl(r)/dNdr_tot) # Wfnl_ell:r*-p*dN/dr   
        self.kernels_ready = False

    def add_kernels(self, ell, logrmin=-10., logrmax=10., num=10000):        
        assert (ell[1:] > ell[:-1]).all(), "ell must be increasing"        
        kw_fft = dict(nu=1.01, N_extrap_low=0, N_extrap_high=0, 
                      c_window_width=0.25, N_pad=0)
        
        self.ell = ell            
        r = np.logspace(logrmin, logrmax, num=num) # k = (ell+0.5) / r
        
        # w_ell(k)
        fr = self.fr_wk(r)
        self.wk = self.fftlog(0, r, fr, **kw_fft)
             
        # wr_ell(k)        
        fr = self.fr_wrk(r)
        self.wrk = self.fftlog(1, r, fr, **kw_fft)

        # wfnl_ell(k)
        fr = self.fr_wkfnl1(r)
        self.wfnlk1 = self.fftlog(0, r, fr, **kw_fft)

        # wfnl_ell(k)
        fr = self.fr_wkfnl2(r)
        self.wfnlk2 = self.fftlog(0, r, fr, **kw_fft)
        
        self.add_integrals()
        self.kernels_ready = True
        
    def fftlog(self, n, r, fr, **kw_fft):
                        
        wk = []        
        fftl = fftlog(r, fr, **kw_fft)        
        if n==0:            
            for ell_ in self.ell:
                kwk_ = fftl.fftlog(ell_, )        
                wk.append(kwk_)
        else:            
            for ell_ in self.ell:
                kwk_ = fftl.fftlog_ddj(ell_, )        
                wk.append(kwk_)            
            
        return wk        
     
    def simps(self, intg, lnk):        
        is_good = np.isfinite(intg)
        
        return simps(intg[is_good], x=lnk[is_good])        
        
    def add_integrals(self):
        
        i_gg = []
        i_rr = []
        i_f1f1 = []
        i_f2f2 = []
        
        i_gr = []
        i_gf1 = []
        i_gf2 = []
        i_rf1 = []
        i_rf2 = []
        i_f1f2 = []
        
        for i in range(len(self.wk)):
            
            k, w_g = self.wk[i]            
            w_r = self.wrk[i][1]
            k2 = k*k
            
            fnl_f = self.alpha_fnl/(k2*self.Tlin(k))
            w_f1 = self.wfnlk1[i][1]*fnl_f
            w_f2 = self.wfnlk2[i][1]*fnl_f
            
            lnk = np.log(k)
            k3pk = k2*k*self.Plin(k)
            
            i_gg.append(self.simps(k3pk*w_g*w_g, lnk))
            i_rr.append(self.simps(k3pk*w_r*w_r, lnk))
            i_f1f1.append(self.simps(k3pk*w_f1*w_f1, lnk))
            i_f2f2.append(self.simps(k3pk*w_f2*w_f2, lnk))
            i_gr.append(self.simps(k3pk*w_g*w_r, lnk))
            i_gf1.append(self.simps(k3pk*w_g*w_f1, lnk))
            i_gf2.append(self.simps(k3pk*w_g*w_f2, lnk))
            i_rf1.append(self.simps(k3pk*w_r*w_f1, lnk))
            i_rf2.append(self.simps(k3pk*w_r*w_f2, lnk))
            i_f1f2.append(self.simps(k3pk*w_f1*w_f2, lnk))
            
        self.i_gg = np.array(i_gg)
        self.i_rr = np.array(i_rr)
        self.i_f1f1 = np.array(i_f1f1)
        self.i_f2f2 = np.array(i_f2f2)
        
        self.i_gr = np.array(i_gr)
        self.i_gf1 = np.array(i_gf1)
        self.i_gf2 = np.array(i_gf2)
        self.i_rf1 = np.array(i_rf1)
        self.i_rf2 = np.array(i_rf2)
        self.i_f1f2 = np.array(i_f1f2)            
            
            
    def run(self, fnl, b, noise):
        #print("fnl, b", fnl, b)
        return self.inv_pi*(b*b*self.i_gg + self.i_rr + fnl*fnl*b*b*self.i_f1f1 + fnl*fnl*self.i_f2f2 \
                       - 2*b*self.i_gr + 2*fnl*b*b*self.i_gf1 + 2*fnl*b*(self.i_gf2-self.i_rf1) \
                       - 2*fnl*self.i_rf2 + 2*fnl*fnl*b*self.i_f1f2) + noise
    
    
class SurveySpectrum(Spectrum, WindowSHT):
    el_model = np.arange(2000)
    
    def __init__(self, *arrays, **kwargs):
        Spectrum.__init__(self, *arrays, **kwargs)
        
    def add_window(self, *arrays, **kwargs):
        WindowSHT.__init__(self, *arrays, **kwargs)
        
    def __call__(self, el, fnl=0.0, b=1.0, noise=0.0):
        
        cl_ = Spectrum.__call__(self, self.el_model, fnl=fnl, b=b, noise=noise)   
        
        clm_ = self.convolve(self.el_model, cl_)
        lmax = max(el)+1
        clm = self.apply_ic(clm_[:lmax])
        
        return clm[el]
    

# class SpectrumOld:
#     """
#     Redshift space power spectrum of biased tracers
#     """    
#     def __init__(self, h=0.67556, T0_cmb=2.7255, Omega0_b=0.0482754208891869,
#                  Omega0_cdm=0.26377065934278865, N_ur=None, m_ncdm=[0.06],
#                  P_k_max=10.0, P_z_max=100.0, sigma8=0.8225, gauge='synchronous',
#                  n_s=0.9667, nonlinear=False):
        
#         # input parameters
#         kw_cosmo = locals()
#         kw_cosmo.pop('self')
#         for kw,val in kw_cosmo.items():
#             print(f'{kw:10s}: {val}')
            
#         sigma8 = kw_cosmo.pop('sigma8')            
#         cosmo = cosmology.Cosmology(**kw_cosmo).match(sigma8=sigma8) 
                
#         # --- cosmology calculators
#         redshift = 0.0
#         delta_crit = 1.686      # critical overdensity for collapse from Press-Schechter
#         DH = (cosmo.H0/cosmo.C) # in units of h/Mpc, eq. 4 https://arxiv.org/pdf/astro-ph/9905116.pdf
#         Omega0_m = cosmo.Omega0_m
        
#         k_ = np.logspace(-10, 10, 10000)
#         Plin_ = cosmology.LinearPower(cosmo, redshift, transfer='CLASS') # P(k) in (Mpc/h)^3 
#         self.Plin = IUS(k_, Plin_(k_), ext=1)
#         Tlin_ = cosmology.transfers.CLASS(cosmo, redshift)  # T(k), normalized to one at k=0
#         is_good = np.isfinite(Tlin_(k_))
#         self.Tlin = IUS(k_[is_good], Tlin_(k_)[is_good], ext=1)
#         self.Dlin = cosmo.scale_independent_growth_factor       # D(z), normalized to one at z=0
#         self.f = cosmo.scale_independent_growth_rate            # dlnD/dlna
#         self.Dc = cosmo.comoving_distance              
#         self.h = cosmo.efunc
                 
#         # alpha_fnl: equation 2 of Mueller et al 2017    
#         self.alpha_fnl  = 3.0*delta_crit*Omega0_m*(DH**2)
        
#     def __call__(self, ell, fnl=0.0, **kwargs):

#         if not self.kernels_ready:
#             print('will create windows')
#             self.make_kernels(ell, **kwargs)
            
#         elif not np.array_equal(ell, self.ell):
#             print('will update windows')            
#             self.make_kernels(ell, **kwargs)
                       
#         return self.__integrate_dk(fnl)
        
#     def add_tracer(self, z, b, dNdz, p=1.6):
#         print(f'p = {p:.1f}')
#         # dz/dr
#         dNdr_spl = IUS(self.Dc(z), dNdz*self.h(z), ext=1) 
#         dNdr_tot = romberg(dNdr_spl, self.Dc(z.min()), self.Dc(z.max()), divmax=1000)
#         #print(dNdr_tot)
        
#         # bias
#         b_spl = IUS(self.Dc(z), b, ext=1)

#         # growth
#         d_spl = IUS(self.Dc(z), self.Dlin(z), ext=1)

#         # growth rate: f = dlnD/dlna
#         f_spl = IUS(self.Dc(z), self.f(z), ext=1)

#         # prepare kernels
#         self.fr_wk = lambda r:r * b_spl(r) * d_spl(r) * (dNdr_spl(r)/dNdr_tot)   # W_ell: r*b*(D(r)/D(0))*dN/dr         
#         self.fr_wrk = lambda r:r * f_spl(r) * d_spl(r) * (dNdr_spl(r)/dNdr_tot)  # Wr_ell:r*f*(D(r)/D(0))*dN/dr
#         self.fr_wkfnl = lambda r: r * (b_spl(r)-p) * (dNdr_spl(r)/dNdr_tot) # Wfnl_ell:r*(b-p)*dN/dr   
#         self.kernels_ready = False

#     def make_kernels(self, ell, logrmin=-10., logrmax=10., num=10000):
#         assert (ell[1:] > ell[:-1]).all(), "ell must be increasing"
#         kw_fft = dict(nu=1.01, N_extrap_low=0, N_extrap_high=0, 
#                       c_window_width=0.25, N_pad=0)
        
#         r = np.logspace(logrmin, logrmax, num=num) # k = (ell+0.5) / r
        
#         # w_ell(k)
#         fr = self.fr_wk(r)
#         fftl1 = fftlog(r, fr, **kw_fft)
#         self.wk = []
#         for ell_ in ell:
#             kwk_ = fftl1.fftlog(ell_)        
#             self.wk.append(kwk_)
             
#         # wr_ell(k)        
#         fr = self.fr_wrk(r)
#         fftl2 = fftlog(r, fr, **kw_fft)
#         self.wrk = []
#         for ell_ in ell:
#             kwk_ = fftl2.fftlog_ddj(ell_)
#             self.wrk.append(kwk_)                

#         # wfnl_ell(k)
#         fr = self.fr_wkfnl(r)
#         fftl3 = fftlog(r, fr, **kw_fft)
#         self.wfnlk = []
#         for ell_ in ell:
#             kwk_ = fftl3.fftlog(ell_)
#             self.wfnlk.append(kwk_)
            
#         self.ell = ell    
#         self.kernels_ready = True
        
#     def __integrate_dk(self, fnl):
        
#         res = []
#         for i in range(len(self.wk)):
            
#             k, wk_gg = self.wk[i]
#             wk_rsd = self.wrk[i][1]
#             wk_fnl = self.wfnlk[i][1]

#             #print(np.percentile(np.log10(k), [0, 100]))
#             wk_t = wk_gg - wk_rsd + fnl*self.alpha_fnl*wk_fnl/(k*k*self.Tlin(k))
                        
#             lnk = np.log(k)
#             intg = k*k*k*self.Plin(k)*wk_t*wk_t
            
#             is_finite = np.isfinite(intg)
#             lnk = lnk[is_finite]
#             intg = intg[is_finite]
            
#             cl_ = simps(intg, x=lnk)
            
#             res.append(cl_)       

#         cls = (2./np.pi)*np.array(res)
#         return cls
    
# class SurveySpectrumOld(SpectrumOld, WindowSHT):
#     el_model = np.arange(2000)
    
#     def __init__(self, *arrays, **kwargs):
#         SpectrumOld.__init__(self, *arrays, **kwargs)
        
#     def add_window(self, *arrays, **kwargs):
#         WindowSHT.__init__(self, *arrays, **kwargs)
        
#     def __call__(self, el, fnl=0.0, noise=0.0):
        
#         cl_ = SpectrumOld.__call__(self, self.el_model, fnl=fnl)   
        
#         clm_ = self.convolve(self.el_model, cl_)+noise
#         lmax = max(el)+1
#         clm = self.apply_ic(clm_[:lmax])
        
#         return clm[el]

    
# import healpy as hp
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
# from scipy.integrate import romberg, simps, quad

# import pyccl as ccl
# # from tqdm import tqdm

# from lssutils.utils import histogram_cell, mask2regions
# from lssutils.stats.cl import AnaFast, cl2xi, xi2cl, gauleg
# from lssutils.dataviz import setup_color


# class NoZ:
    
#     def __init__(self, 
#                  filename,
#                  k=3,
#                  zmin=0.0,
#                  zmax=5.0, 
#                  fill_value='extrapolate'):
        
#         self.nz_, self.z_ = np.loadtxt(filename).T        
#         nz_spl = interp1d(self.z_, self.nz_, kind=k, 
#                           fill_value=fill_value, bounds_error=False) # extrapolate
#         #nz_spl.set_smoothing_factor(smfactor)
                
#         self.nz_norm = romberg(nz_spl, zmin, zmax, divmax=1000)        
#         self.nz_fn = lambda z:nz_spl(z)/self.nz_norm        
        
        
        
# def read_weight_mask():
    
#     #mask_ = hp.read_map('/Volumes/TimeMachine/data/DR7/mask.cut.hp.256.fits', dtype=np.float32, verbose=False) > 0
#     #mask = mask2regions(mask_)[0] # NGC
    
#     # read mask
#     mask = hp.read_map('/Volumes/TimeMachine/data/DR9fnl/north_mask_hp256.hp.fits', dtype=np.float64) > 0.0
    
#     maskf = np.ones_like(mask)
#     weight = maskf * 1.0        # all sky is the same
    
#     return weight, mask
    

# def read_mocks(case, return_cov=False):
#     # data
#     cl_mocks = np.load('cl_mocks_2k_qso.npz', allow_pickle=True)
#     #print(cl_mocks.files)
#     cl_full = cl_mocks[case] # select full sky mocks

#     # bin measurements
#     #bins = #np.arange(2, 501, 6)
#     bins = np.array([2+i*2 for i in range(10)] + [2**i for i in range(5, 10)])

#     cl_fullb = []
#     for i in range(cl_full.shape[0]):

#         x, clb_ =  histogram_cell(cl_full[i, :], bins=bins)

#         cl_fullb.append(clb_)
#         #print('.', end='')
#     cl_fullb = np.array(cl_fullb)

#     y = cl_fullb.mean(axis=0)

#     nmocks, nbins = cl_fullb.shape
#     hf = (nmocks - 1.0)/(nmocks - nbins - 2.0)

#     cov = np.cov(cl_fullb, rowvar=False)*hf
#     invcov = np.linalg.inv(cov)
#     x = x.astype('int')
#     print(f'bins: {x}, nmocks: {nmocks}, nbins: {nbins}')
#     ret = (x, y, invcov)
#     if return_cov:
#         ret += (cov, )
        
#     return ret

# def read_datacl(kind='after'):
#     cl_data = np.load('/Volumes/TimeMachine/data/DR9fnl/cl_qso_north_rf.npz', allow_pickle=True)
#     shotnoise = 1.017379567874895e-06
    
#     cl_obs = cl_data[f'cl_{kind}']-shotnoise
#     el_obs = np.arange(cl_obs.size)
#     bins = np.array([2+i*2 for i in range(10)] + [2**i for i in range(5, 10)])
#     elb, clb = histogram_cell(cl_obs, bins=bins)
#     return elb, clb


# def run_ccl(cosmo, dndz, bias, ell, has_rsd=False):
#     """ Run CCL angular power spectrum
    
#     """
#     clu1 = ccl.NumberCountsTracer(cosmo, has_rsd=has_rsd, dndz=dndz, bias=bias, )
#     return ccl.angular_cl(cosmo, clu1, clu1, ell) #Clustering    




# def plot_sample(z, dNdz, b):
#     plt.plot(z, dNdz)
#     plt.xlabel('z')
#     plt.ylabel('dN/dz')
#     plt.twinx()
#     plt.plot(z, b, 'C1--')
#     plt.ylabel('b')    

# def init_sample(kind='mock', verb=False, **kw):
#     """ Define a mock N(z), credit: DESC CCL

#     """
#     if kind == 'mock':             
#         z = np.arange(0.0, 3.0, 0.001)
#         i_lim = 26.                          # Limiting i-band magnitude
#         z0 = 0.0417*i_lim - 0.744
        
#         Ngal = 46. * 100.31 * (i_lim - 25.)            # Normalisation, galaxies/arcmin^2
#         pz = 1./(2.*z0) * (z / z0)**2. * np.exp(-z/z0)  # Redshift distribution, p(z)
#         dNdz = Ngal * pz                               # Number density distribution
#         b = 1.5*np.ones(z.size)                        # Galaxy bias (constant with scale and z)
        
#     elif kind == 'eboss':
        
#         from scipy.ndimage import gaussian_filter1d
#         z, dNdz_ = np.loadtxt('./nbar_eBOSS_QSO_NGC_v7_2.dat', usecols=(0, 3), unpack=True)
#         dNdz_[z < 0.05] = 0.0
#         b = 1.5*np.ones(dNdz_.size)
#         dNdz = gaussian_filter1d(dNdz_, 10.0)
        
#     elif kind == 'qsomock':
        
#         nz = NoZ('/Volumes/TimeMachine/data/DR9fnl/FOR_MEDHI/RF_g.txt', **kw)
#         z = np.arange(0.0, 5.0, 0.02)
#         dNdz = nz.nz_fn(z)
#         b = 2.5*np.ones_like(z)
        
#     else:
#         raise RuntimeError(f"{kind} not supported.")
        

#     if verb:plot_sample(z, dNdz, b)
    
#     return z, b, dNdz


# class WindowRR:
#     def __init__(self, rr_file, ntot, npix):
        
#         area = ntot / npix
        
#         raw_data = np.load(rr_file, allow_pickle=True)
#         sep = raw_data[0][::-1]           # in radians
#         rr_counts = raw_data[1][::-1]*2.0 # paircount uses symmetry

#         sep_mid = 0.5*(sep[1:]+sep[:-1])
#         dsep = np.diff(sep)        
#         window = rr_counts / (dsep*np.sin(sep_mid)) * (2./(npix*npix*area))
        
#         self.window_spl = interp1d(np.cos(sep_mid), window, fill_value="extrapolate")

        
# class WindowSHT:
    
#     def __init__(self, weight, mask, ell_ob, ngauss=2**12):
#         af = AnaFast()
#         cl_ = af(mask*1.0, weight, mask)        
        
#         xi_zero = (cl_['cl']*(2.*cl_['l']+1.)).sum() / (4.*np.pi)

#         self.ell_ob = ell_ob
#         self.twopi = 2.*np.pi

#         self.x, self.w = gauleg(ngauss)
#         self.xi_mask = self.cl2xi(cl_['l'], cl_['cl']) / xi_zero
                
#         self.Pl = []
#         for ell in self.ell_ob:
#             self.Pl.append(np.polynomial.Legendre.basis(ell)(self.x))
      
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
    

#     def convolve(self, el_model, cl_model):
        
#         xi_th = self.cl2xi(el_model, cl_model)
#         xi_thw = xi_th * self.xi_mask
#         cl_thw = self.xi2cl(xi_thw)        
        
#         return cl_thw

#     def xi2cl(self, xi):
#         '''
#             calculates Cell from omega
#         '''
#         cl  = []
#         xiw = xi*self.w
#         for i in range(len(self.Pl)):
#             cl.append((xiw * self.Pl[i]).sum())
            
#         return self.twopi*np.array(cl)

#     def cl2xi(self, ell, cell):
#         '''
#             calculates omega from Cell at Cos(theta)
#         '''
#         twol4pi = (2.*ell+1.)/(4.*np.pi)
#         return np.polynomial.legendre.legval(self.x, c=twol4pi*cell, tensor=False)
    
    
# class Posterior:
#     """ Log Posterior for PNGModel
#     """
#     def __init__(self, model):
#         self.model = model
        
#     def logprior(self, theta):
#         ''' The natural logarithm of the prior probability. '''
#         lp = 0.
#         # unpack the model parameters from the tuple
#         fnl = theta
#         # uniform prior on fNL
#         fmin = -1000. # lower range of prior
#         fmax = 1000.  # upper range of prior
#         # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
#         lp = 0. if fmin < fnl < fmax else -np.inf
#         ## Gaussian prior on ?
#         #mmu = 3.     # mean of the Gaussian prior
#         #msigma = 10. # standard deviation of the Gaussian prior
#         #lp -= 0.5*((m - mmu)/msigma)**2

#         return lp

#     def loglike(self, theta, y, invcov, x):
#         '''The natural logarithm of the likelihood.'''
#         # unpack the model parameters
#         fnl = theta
#         # evaluate the model
#         md = self.model(x, fnl=fnl)
#         # return the log likelihood
#         return -0.5 * (y-md).dot(invcov.dot(y-md))

#     def logpost(self, theta, y, invcov, x):
#         '''The natural logarithm of the posterior.'''
#         return self.logprior(theta) + self.loglike(theta, y, invcov, x)
    

# class PNGModel:
#     """
    
#     **See Fang et al (2019); arXiv:1911.11947**
    
#     """
        
#     def __init__(self, cosmo, zref=0.0, has_rsd=False, has_fnl=False):
#         """
        
        
#         E.g.:
#             _h = 0.6777
#             _Ob0 = 0.048206
#             _Ocdm0 = 0.307115 - _Ob0
#             _ns = 0.9611
#             _Tcmb = 2.7255
#             _sigma8 = 0.8225
            
#             cosmo = ccl.Cosmology(Omega_c=_Ocdm0, Omega_b=_Ob0, h=_h, 
#                                   n_s=_ns, sigma8=_sigma8,
#                                   T_CMB=_Tcmb, transfer_function='boltzmann_camb', 
#                                   matter_power_spectrum='linear')        
        
#             th = Model(cosmo)
#             z = np.linspace(0., 3.)
#             k = np.logspace(-3, 0)
#             fig, ax = plt.subplots(ncols=3, figsize=(12, 3))
            
#             ax[0].loglog(k, th.pk(k))
#             ax[1].plot(z, th.z2r(z))
#             ax[2].plot(z, th.d(z))
            
#         """
#         msg = ("NOTE: This code uses a cosmology calculator that returns k"
#               " and P(k) in units of 1/Mpc and Mpc^3.\n Therefore, "
#               "the coefficient alpha in the model uses H_0 = 100h, not H_0=100.")
#         print(msg)

#         self.has_rsd = has_rsd
#         self.has_fnl = has_fnl
#         a = 1./(1.+zref)        
#         self.Omega_M = cosmo.cosmo.params.Omega_c+cosmo.cosmo.params.Omega_b
        
#         # note that k is in unit of Mpc^-1
#         self.H0c = (cosmo.cosmo.params.h/ccl.physical_constants.CLIGHT_HMPC)
#         self.delta_c = 1.686 # spherical collapse overdensity
#         self.coef_fnl = 3*self.delta_c*self.Omega_M*self.H0c**2        
                    
#         self.pk = lambda k:ccl.linear_matter_power(cosmo, k, a)
#         self.z2r = lambda z:ccl.comoving_radial_distance(cosmo, 1./(1.+z)) # (1+z) D_A(z)
#         self.d = lambda z:ccl.growth_factor(cosmo, 1./(1.+z)) # normalized to 1 at z=0
#         self.f = lambda z:ccl.growth_rate(cosmo, 1./(1.+z))
#         self.h = lambda z:ccl.h_over_h0(cosmo, 1./(1.+z))
        
#         self.kernels_ready = False

#     def __call__(self, ell, fnl=0.0, **kwargs):

#         if not self.kernels_ready:
#             print('will create windows')
#             self.__make_kernels(ell, **kwargs)
#         elif not np.array_equal(ell, self.ell):
#             print('will update windows')            
#             self.__make_kernels(ell, **kwargs)
                       
#         return self.__integrate_dk(fnl)

#     def add_tracer(self, z, b, nz, p=1.0):
#         self.p = p
#         intrp_kw = dict(fill_value=0.0, bounds_error=False)

#         # dz/dr
#         nh_spl = interp1d(self.z2r(z), nz*self.h(z), **intrp_kw) 
#         nh_tot = romberg(nh_spl, self.z2r(z.min()), self.z2r(z.max()), divmax=1000)

#         # bias
#         b_spl = interp1d(self.z2r(z), b, **intrp_kw)

#         # growth
#         d_spl = interp1d(self.z2r(z), self.d(z), **intrp_kw)

#         # growth rate: f = dlnD/dlna
#         f_spl = interp1d(self.z2r(z), self.f(z), **intrp_kw)

#         self.fr_wk = lambda r:r * b_spl(r) * d_spl(r) * (nh_spl(r)/nh_tot)   # W_ell: r*b*(D(r)/D(0))*dN/dr         
#         self.fr_wrk = lambda r:r * f_spl(r) * d_spl(r) * (nh_spl(r)/nh_tot)  # Wr_ell:r*f*(D(r)/D(0))*dN/dr
#         self.fr_wkfnl = lambda r: r * (b_spl(r)-self.p) * (nh_spl(r)/nh_tot) # Wfnl_ell:r*(b-p)*dN/dr            
        

    
#     def __make_kernels(self, ell, rmin=-10., rmax=10., num=1000):
#         assert (ell[1:] > ell[:-1]).all(), "ell must be increasing"
#         kw_fft = dict(nu=1.01, N_extrap_low=1500, N_extrap_high=1500, c_window_width=0.25, N_pad=5000)
#         #assert (ell.min() > 1), "RSD requires ell > 2"
        
#         r = np.logspace(rmin, rmax, num=num) # k = (ell+0.5) / r
        
#         # w_ell(k)
#         fr = self.fr_wk(r)
#         fftl1 = fftlog(r, fr, **kw_fft)
#         self.wk = []
#         for ell_ in ell:
#             kwk_ = fftl1.fftlog(ell_)        
#             self.wk.append(kwk_)
             
#         # wr_ell(k)        
#         fr = self.fr_wrk(r)
#         fftl2 = fftlog(r, fr, **kw_fft)
#         self.wrk = []
#         for ell_ in ell:
#             kwk_ = fftl2.fftlog_ddj(ell_)
#             self.wrk.append(kwk_)                

#         # wfnl_ell(k)
#         fr = self.fr_wkfnl(r)
#         fftl3 = fftlog(r, fr, **kw_fft)
#         self.wfnlk = []
#         for ell_ in ell:
#             kwk_ = fftl3.fftlog(ell_)
#             self.wfnlk.append(kwk_)
            
#         self.ell = ell    
#         self.kernels_ready = True
        
#     def __integrate_dk(self, fnl):
        
#         res = []
#         #for i in tqdm(range(len(self.wk))):
#         for i in range(len(self.wk)):
#             k_, wk_ = self.wk[i]
#             wk_t = wk_*1.0
            
#             if self.has_rsd:
#                 _, wrk_ = self.wrk[i]
#                 wk_t -= wrk_

#             if self.has_fnl:
#                 _, wrk_ = self.wfnlk[i]
#                 wk_t += self.coef_fnl*fnl*wrk_/(k_*k_) # fnl ~ 1/k^2
            
#             lnk = np.log(k_)
#             intg = k_*k_*k_*self.pk(k_)*wk_t*wk_t            
#             cl_ = simps(intg, x=lnk)
#             res.append(cl_)       

#         cls = (2./np.pi)*np.array(res)
#         if self.ell[0] == 0:
#             cls[0] = 0.0
#         #cls[1] = 0.0
#         return cls
#                 
#     def savetxt(self, filename):
#         el_ = np.arange(self.cl.size)
#         np.savetxt(filename, np.column_stack([el_, self.cl]), fmt='%d %.15f')
