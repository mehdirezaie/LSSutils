import matplotlib.pyplot as plt
import numpy as np

# [
#     'star_density', 'ebv', 'loghi',
#     'sky_g', 'sky_r', 'sky_i', 'sky_z',
#     'depth_g_minus_ebv','depth_r_minus_ebv', 
#     'depth_i_minus_ebv', 'depth_z_minus_ebv', 
#     'psf_g', 'psf_r', 'psf_i', 'psf_z',
#      'run', 'airmass'
# ]

def setup_color():
    from cycler import cycler
    colors = ['#377eb8', '#ff7f00', '#4daf4a',
              '#f781bf', '#a65628', '#984ea3',
              '#999999', '#e41a1c']
    styles = 2*['-', '--', '-.', ':']
    plt.rc('axes', prop_cycle=(cycler('color', colors)+cycler('linestyle', styles)))

def plot_nz(fig_path):
    '''
    !head -n 3 /home/mehdi/data/eboss/data/v7_2/nbar_eBOSS_QSO_*GC_v7_2.dat
    ==> /home/mehdi/data/eboss/data/v7_2/nbar_eBOSS_QSO_NGC_v7_2.dat <==
    # effective area (deg^2), effective volume (Mpc/h)^3: 2860.4406814189183 99740127.64438488
    # zcen,zlow,zhigh,nbar,wfkp,shell_vol,total weighted gals
    0.005 0.0 0.01 0.00013462213635665303 0.5976890894834057 7771.038808820277 1.046153846153846

    ==> /home/mehdi/data/eboss/data/v7_2/nbar_eBOSS_QSO_SGC_v7_2.dat <==
    # effective area (deg^2), effective volume (Mpc/h)^3: 1838.8730140161808 53674436.18509478
    # zcen,zlow,zhigh,nbar,wfkp,shell_vol,total weighted gals
    0.005 0.0 0.01 0.00020558148614182404 0.4931191556659562 4995.717495292908 1.027027027027027
    In [3]:

    2860.4406814189183 + 1838.8730140161808
    Out[3]:
    4699.313695435099

    '''
    #plt.rc('font', family='serif', size=15)

    path_nbar = '/home/mehdi/data/eboss/data/v7_2/'
    nbar_ngc = np.loadtxt(f'{path_nbar}nbar_eBOSS_QSO_NGC_v7_2.dat')
    nbar_sgc = np.loadtxt(f'{path_nbar}nbar_eBOSS_QSO_SGC_v7_2.dat')


    fig, ax = plt.subplots(figsize=(6, 4))

    kw = dict(where='mid')
    ax.step(nbar_ngc[:,0], nbar_ngc[:, 3]/1.0e-5, label='NGC',  **kw) # ls='--', color='royalblue',
    ax.step(nbar_sgc[:,0], nbar_sgc[:, 3]/1.0e-5, label='SGC',  **kw) # ls='-', color='darkorange',

    samples = ['Main', 'High-z']
    lines  = [0.8, 2.2, 3.5]

    for i, line in enumerate(lines):
        ax.axvline(line, zorder=0, ls=':', color='grey')

        if i<2:
            #-- annotation
            if i==0:
                ypos = 0.5
            else:
                ypos = 1.8

            ax.text(0.45*(lines[i]+lines[i+1]), ypos, samples[i], color='grey')

            hwidth=0.1
            width=0.001
            kw = dict(shape='full', width=width, 
                      head_width=hwidth, color='grey', alpha=0.25)  

            # left to right
            ax.arrow(lines[i], ypos-0.1, lines[i+1]-lines[i]-1.5*hwidth, 0.0, **kw)        
            # right to left
            ax.arrow(lines[i+1], ypos-0.1, -lines[i+1]+lines[i]+1.5*hwidth, 0.0, **kw) 

    ax.tick_params(direction='in', which='both', axis='both', pad=6, right=True, top=True)
    ax.set(xlabel='z', ylabel=r'$10^{5}n(z)~[h/{\rm Mpc}]^{3}$', 
           ylim=(0.0, 2.5), xlim=(0.3, 3.8))
    ax.legend(loc='center right')
    fig.savefig(fig_path, bbox_inches='tight')
    return fig, ax

def mollweide(fig_path):
    """ Plot a mollweide project of Nqso """
    import lssutils.dataviz as dv
    from lssutils.utils import EbossCat, nside2pixarea
    import healpy as hp


    ''' READ THE FULL CATALOGS
    '''
    nside = 128
    area_1pix = nside2pixarea(nside, degrees=True)
    nran_bar = area_1pix*5000. # 5000 per sq deg
    path_cats = '/home/mehdi/data/eboss/data/v7_2/'
    dNGC = EbossCat(f'{path_cats}eBOSS_QSO_full_NGC_v7_2.dat.fits', zmin=0.8, zmax=3.5)
    rNGC = EbossCat(f'{path_cats}eBOSS_QSO_full_NGC_v7_2.ran.fits', kind='randoms', zmin=0.8, zmax=3.5)

    dSGC = EbossCat(f'{path_cats}eBOSS_QSO_full_SGC_v7_2.dat.fits', zmin=0.8, zmax=3.5)
    rSGC = EbossCat(f'{path_cats}eBOSS_QSO_full_SGC_v7_2.ran.fits', kind='randoms', zmin=0.8, zmax=3.5)



    ''' AFTER CORRECTION
    '''
    # NGC
    ngal_ngc = dNGC.to_hp(nside, 0.8, 3.5, raw=2)
    nran_ngc = rNGC.to_hp(nside, 0.8, 3.5, raw=2)

    # SGC
    ngal_sgc = dSGC.to_hp(nside, 0.8, 3.5, raw=2)
    nran_sgc = rSGC.to_hp(nside, 0.8, 3.5, raw=2)

    frac_ngc = nran_ngc / nran_bar
    frac_sgc = nran_sgc / nran_bar

    ngal_tot_f = ngal_ngc + ngal_sgc
    frac_tot_f = frac_ngc + frac_sgc


    ngal_dens = ngal_tot_f / (frac_tot_f * area_1pix)

    vmin, vmax = np.percentile(ngal_dens[np.isfinite(ngal_dens)], [5, 95])
    dv.mollview(ngal_dens, vmin, vmax,
                r'$n_{{\rm QSO}}~[{\rm deg}^{-2}]$', cmap=plt.cm.YlOrRd_r, galaxy=True, colorbar=True)
    plt.savefig(fig_path, bbox_inches='tight', dpi=300, rasterized=True)

    
def plot_deltaNqso(fig_path):
    import sys
    sys.path.insert(0, '/home/mehdi/github/LSSutils')
    from lssutils.utils import nside2pixarea, EbossCat
    from lssutils.dataviz import mollview, mycolor, shiftedColorMap

    ''' READ THE FULL CATALOGS
    '''
    nside = 128
    pix_area = nside2pixarea(nside, degrees=True)
    nran_bar = pix_area*5000. # 5000 per sq deg
    
    cmap = shiftedColorMap(plt.cm.jet, midpoint=0.5)


    path_cats = '/home/mehdi/data/eboss/data/v7_2/3.0/catalogs/'
    print(path_cats)

    # nn
    dNGC = EbossCat(f'{path_cats}eBOSS_QSO_full_NGC_known_mainhighz_512_v7_2.dat.fits', zmin=0.8, zmax=3.5)
    rNGC = EbossCat(f'{path_cats}eBOSS_QSO_full_NGC_known_mainhighz_512_v7_2.ran.fits', kind='randoms', zmin=0.8, zmax=3.5)
    dSGC = EbossCat(f'{path_cats}eBOSS_QSO_full_SGC_known_mainhighz_512_v7_2.dat.fits', zmin=0.8, zmax=3.5)
    rSGC = EbossCat(f'{path_cats}eBOSS_QSO_full_SGC_known_mainhighz_512_v7_2.ran.fits', kind='randoms', zmin=0.8, zmax=3.5)

    ngal_ngc = dNGC.to_hp(nside, 0.8, 3.5, raw=2)
    nran_ngc = rNGC.to_hp(nside, 0.8, 3.5, raw=2)
    ngal_sgc = dSGC.to_hp(nside, 0.8, 3.5, raw=2)
    nran_sgc = rSGC.to_hp(nside, 0.8, 3.5, raw=2)
    frac_ngc = nran_ngc / nran_bar
    frac_sgc = nran_sgc / nran_bar
    ngal_tot = ngal_ngc + ngal_sgc
    frac_tot = frac_ngc + frac_sgc

    # standard
    path_cats = '/home/mehdi/data/eboss/data/v7_2/'
    dNGC = EbossCat(f'{path_cats}eBOSS_QSO_full_NGC_v7_2.dat.fits', zmin=0.8, zmax=3.5)
    rNGC = EbossCat(f'{path_cats}eBOSS_QSO_full_NGC_v7_2.ran.fits', kind='randoms', zmin=0.8, zmax=3.5)
    dSGC = EbossCat(f'{path_cats}eBOSS_QSO_full_SGC_v7_2.dat.fits', zmin=0.8, zmax=3.5)
    rSGC = EbossCat(f'{path_cats}eBOSS_QSO_full_SGC_v7_2.ran.fits', kind='randoms', zmin=0.8, zmax=3.5)

    ngal_ngc = dNGC.to_hp(nside, 0.8, 3.5, raw=2)
    nran_ngc = rNGC.to_hp(nside, 0.8, 3.5, raw=2)
    ngal_sgc = dSGC.to_hp(nside, 0.8, 3.5, raw=2)
    nran_sgc = rSGC.to_hp(nside, 0.8, 3.5, raw=2)
    frac_ngc = nran_ngc / nran_bar
    frac_sgc = nran_sgc / nran_bar
    ngal_tot_ = ngal_ngc + ngal_sgc
    frac_tot_ = frac_ngc + frac_sgc

    ngal_dens = ngal_tot / (frac_tot * pix_area)
    ngal_dens_ = ngal_tot_ / (frac_tot_ * pix_area)
    
    delta_ngal = (ngal_dens_ - ngal_dens)
    
    vmin, vmax = np.percentile(delta_ngal[np.isfinite(delta_ngal)], [1, 99])
    print(vmin, vmax)
    
    mollview(delta_ngal, vmin, vmax,
             r'$\Delta$n$_{{\rmQSO}}$ [deg$^{-2}$] (Standard - NN)', 
             cmap=cmap, colorbar=True)
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)    

    
def p0_demo(fig_path, cap='NGC', sample='main', method='known', show_nn=False):
    # plot P0 
    # FIX covmax
    
    from matplotlib.gridspec import GridSpec
    import nbodykit.lab as nb

    def read_pk(filename):
        d = nb.ConvolvedFFTPower.load(filename)
        p0 = d.poles['power_0'].real #- d.attrs['shotnoise']
        return (d.poles['k'], p0)


    def add_pk(x, y, err, ax1, ax2, **kw):
        ax1.errorbar(x, y, err, capsize=2, **kw)
        ax2.errorbar(x, y, err, capsize=2, **kw, zorder=-1)

    zlim = {'main':'$0.8<z<2.2$',
            'highz':'$2.2<z<3.5$'}
    
    path = '/home/mehdi/data/eboss/data/v7_2/3.0/measurements/spectra'

    covpks_all = np.load('./covpk_ezmocks_ngcsgc_v1.0.npz', allow_pickle=True)
    covpks = covpks_all[f'{cap}_null_standard_512'].item()    
    pk_err = np.sqrt(np.diagonal(covpks['covp0']))

    pks = {}
    pks['systot'] = read_pk(f'{path}/spectra_{cap}_knownsystot_mainhighz_512_v7_2_{sample}.json')
    pks['noweight'] = read_pk(f'{path}/spectra_{cap}_noweight_mainhighz_512_v7_2_{sample}.json')
    pks['nn'] = read_pk(f'{path}/spectra_{cap}_{method}_mainhighz_512_v7_2_{sample}.json')    

    fig = plt.figure(figsize=(8, 5))
    plt.subplots_adjust(wspace=0.0)
    gs  = GridSpec(1, 2, width_ratios=[1, 1], figure=fig)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])


    #--- cosmetics
    ax1.tick_params(direction='in', axis='both', which='both', left=True, right=False)
    ax2.tick_params(direction='in', axis='both', which='both', left=False, right=True)
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)


    xmin, xbreak, xmax = (0.001, 0.02, 0.25)
    ylim = (8.0e3, 3.0e6)
    ax1.set(xlim=(xmin, xbreak), ylim=ylim, yscale='log', xscale='log')
    ax1.set_xticks([0.001, 0.01])
    ax1.set_xticklabels(['0.001', '0.01'])
    ax2.set(xlim=(xbreak, xmax), ylim=ylim, yscale='log')#, xscale='log')
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xticks([0.02, 0.08, 0.14, 0.2])

    ax1.axvline(xbreak, ls=':', color='grey', alpha=0.2)

    ax2.set_xlabel(r'k [h/Mpc]')
    ax1.set_ylabel(r'P$_{0}$(k) [Mpc/h]$^{3}$')
    ax2.xaxis.set_label_coords(0., -0.1) # https://stackoverflow.com/a/25018490/9746916

    #--- plots
    add_pk(*pks['noweight'], pk_err, ax1, ax2, marker='.',  label='Before treatment') # color='#000000',
    add_pk(*pks['systot'], pk_err, ax1, ax2, marker='o', mfc='w',  label='Standard treatment') # ls='--', color='#4b88e3',
    if show_nn:
        add_pk(*pks['nn'], pk_err, ax1, ax2, marker='s', mfc='w', label='Neural Network treatment') # color='#d98404',

    ax2.legend(loc='upper left', frameon=False, 
               bbox_to_anchor=(-0.4, 0.95), 
               title=fr"eBOSS DR16 QSO {cap} ({zlim[sample]})")
    
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')        
    

class NbarCov:
    
    def __init__(self, c, n):
        """ read covariance matrix """
        from glob import glob
        
        path = '/home/mehdi/data/eboss/mocks/1.0/measurements/nnbar/'
        nnbar_null = glob(f'{path}nnbar_{c}_knownsystot_mainhighz_512_v7_0_*_main_{n}.npy')
        self.nmocks = len(nnbar_null)
        print(f'nmocks: {self.nmocks}')

        err_tot = []
        for j, fn in enumerate(nnbar_null):
            d = np.load(fn, allow_pickle=True)

            err_j = []
            for i, di in enumerate(d):
                err_j.append(di['nnbar'] - 1.0)
            err_tot.append(err_j)            
        self.err_tot = np.cov(np.array(err_tot).reshape(self.nmocks, -1), 
                              rowvar=False)

    def get_invcov(self, start, end, return_covmax=False):
        # https://arxiv.org/pdf/astro-ph/0608064.pdf
        err_tot_ = self.err_tot[start:end, start:end]
        
        nbins = err_tot_.shape[0]        
        print(f'nbins: {nbins}')

        hartlop_factor = (self.nmocks - 1.) / (self.nmocks - nbins - 2.)
        covmax = hartlop_factor * err_tot_
        if return_covmax:
            return np.linalg.inv(covmax), covmax
        else:
            return np.linalg.inv(covmax)

    
def plot_overdensity(fig_path, sample='main'):    
    # maps_eboss_v7p2 = [
    #     'star_density', 'ebv', 'loghi',
    #     'sky_g', 'sky_r', 'sky_i', 'sky_z',
    #     'depth_g_minus_ebv','depth_r_minus_ebv', 
    #     'depth_i_minus_ebv', 'depth_z_minus_ebv', 
    #     'psf_g', 'psf_r', 'psf_i', 'psf_z',
    #      'run', 'airmass'
    # ]
    
    def chi2(y, invcov):
        return np.dot(y, np.dot(invcov, y))  
    
    def read_nnbar(fn, ix=[1]):

        nnbar = np.load(fn, allow_pickle=True)

        out = []
        for i in ix:
            out.append([nnbar[i]['bin_avg'], nnbar[i]['nnbar']-1]) #, nnbar[i]['nnbar_err']])
        return out
    
    maps = ['Nstar', 'E(B-V)', 'Sky-i', 'Depth-g', 'PSF-i']
    ixx = [0, 1, 5, 7, 13]    
    zlabels = {'main':'0.8<z<2.2',
               'highz':'2.2<z<3.5'}

    nnbar = {}
    path = '/home/mehdi/data/eboss/data/v7_2/3.0/measurements/nnbar/'
    for cap in ['NGC', 'SGC']:
        nnbar[f'noweight_{cap}'] = read_nnbar(f'{path}nnbar_{cap}_noweight_mainhighz_512_v7_2_{sample}_512.npy', ixx)
        nnbar[f'systot_{cap}'] = read_nnbar(f'{path}nnbar_{cap}_knownsystot_mainhighz_512_v7_2_{sample}_512.npy', ixx)
        nnbar[f'nn_{cap}'] = read_nnbar(f'{path}nnbar_{cap}_known_mainhighz_512_v7_2_{sample}_512.npy', ixx)

    incov_ngc = NbarCov('NGC', '512')
    incov_sgc = NbarCov('SGC', '512')
    
    inv_cov = {}
    inv_cov['NGC'] = []
    inv_cov['SGC'] = [] 
    
    for ix in ixx:
        inv_cov['SGC'].append(incov_sgc.get_invcov(ix*8, (ix+1)*8, return_covmax=True))
        inv_cov['NGC'].append(incov_ngc.get_invcov(ix*8, (ix+1)*8, return_covmax=True))
        

    fig, ax = plt.subplots(figsize=(20, 7), nrows=2, ncols=5, sharey='row')
    fig.subplots_adjust(wspace=0.0, hspace=0.15)

    ax = ax.flatten()


    markers = {'noweight':'.',
             'systot':'o',
             'nn':'s'}

    colors = {'noweight':'#000000',
             'systot':'#4b88e3',
             'nn':'#d98404'}

    labels = {'noweight':'Before treatment',
             'systot':'Standard treatment',
             'nn':'Neural Network treatment'} 

    for k, cap in enumerate(['NGC', 'SGC']):

        for i, name in enumerate(['noweight', 'systot', 'nn']):

            nbar = nnbar[f'{name}_{cap}']

            ls = '-' if name=='noweight' else 'none'

            for j, nbar_j in enumerate(nbar):
                jk = 5*k+j
                
                invcov = inv_cov[cap][j][0]
                covmax = np.sqrt(np.diag(inv_cov[cap][j][1]))
                
                ebar = ax[jk].errorbar(*nbar_j, yerr=covmax, label=labels[name], 
                        marker=markers[name], #color=colors[name],
                        mfc='w', capsize=2, alpha=0.8, ls=ls
                       )

                chi2v = chi2(nbar_j[1], invcov)
                
                xtext = 0.05 if jk in [3, 8] else 0.38
                ax[jk].text(xtext, 0.9-i*0.085, r'$\chi^{2}/dof$ = $%.2f/%d$'%(chi2v,len(nbar_j[1])),
                        color=ebar[0].get_color(), transform=ax[jk].transAxes)


        ax[5*k].set_ylabel(r'$\delta$')
        ax[5*k].text(0.1, 0.2, cap, transform=ax[5*k].transAxes, color='grey')
        
        #ax[5*k+4].legend(loc='lower left', title=f'{cap} {zlabels[sample]}', 
        #                 frameon=False, fontsize=13)
    ax[1].legend(**dict(ncol=3, frameon=False,
                 bbox_to_anchor=(0, 1.05, 3, 0.4), loc="lower left",
                 mode="expand", borderaxespad=0))


    for axi in ax:
        axi.locator_params(tight=True, nbins=6, prune='both')
        axi.tick_params(direction='in', which='both', axis='both', right=True)    
        axi.axhline(0, ls=':', color='grey') 

    for i in range(5):    
        ax[5+i].set(xlabel=maps[i])    
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')

def nbar_covmax(fig_path, cap='NGC', nside=512):
    from glob import glob
    def read_nnbar(path, ix=None):
        d = np.load(path, allow_pickle=True)
        nnbar = []
        if ix is None:        
            for di in d:
                nnbar.append(di['nnbar']-1)
        else:
            for i in ix:
                di = d[i]
                nnbar.append(di['nnbar']-1)
        return np.array(nnbar).flatten()

    ix = None
    path_ = '/home/mehdi/data/eboss/mocks/1.0/measurements/nnbar/'
    mocks = glob(f'{path_}nnbar_{cap}_knownsystot_mainhighz_512_v7_0_*_main_{nside}.npy')
    print('len(nbars):', len(mocks), cap)
    nmocks = len(mocks)
    err_tot = []
    for j, fn in enumerate(mocks):
        err_j = read_nnbar(fn, ix=ix)
        err_tot.append(err_j)            
    err_tot = np.array(err_tot)
    print(err_tot.shape)

    nmocks, nbins = err_tot.shape
    covmax_ = np.cov(err_tot, rowvar=False)
    hartlapf = (nmocks - 1.) / (nmocks - nbins - 2.)
    covmax = hartlapf*covmax_    
    
    maps = ['Nstar', 'EBV', 'logHI']
    maps += ['-'.join([s, b]) for s in ['sky', 'depth', 'psf'] for b in 'griz']
    maps += ['run', 'airmass']

    fig, ax = plt.subplots(figsize=(8, 7))

    fig.subplots_adjust(wspace=0.0, hspace=0.0)

    covmap = ax.imshow(covmax*1.0e5, origin='lower', cmap=plt.cm.bwr, vmin=-1.1, vmax=1.1)
    xticks = np.array([i*8+4 for i in range(nbins//8)])
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
    # Minor ticks, see, e.g., https://stackoverflow.com/a/38994970
    ax.set_xticks(xticks-4, minor=True)
    ax.set_yticks(xticks-4, minor=True)

    ax.tick_params(direction='in', which='both', axis='both')
    ax.set_xticklabels(maps, rotation=90, fontsize=14)
    ax.set_yticklabels(maps, fontsize=14)
    ax.grid(which='minor', color='grey', linestyle=':')
    cbar = fig.colorbar(covmap, ax=ax, extend='both', shrink=0.8, pad=0.02, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar.set_label(r'10$^{5}$ Covariance Matrix', fontsize=15)
    
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    
def nnbarchi2pdf_mocks_data(fig_path, cap='NGC', 
                            xlim1=(40., 300.), xlim2=(2400., 2800.),
                            xticks2=[2500, 2700],
                            ix=None):
    from glob import glob
    from matplotlib.gridspec import GridSpec
    from scipy.special import gamma

    def chi2_pdf(x, k):
        """ Chi2 pdf 
        """
        k2 = k / 2.
        n_ = np.power(x, k2-1.)
        d_ = np.power(2., k2)*gamma(k2)
        return np.exp(-0.5*x)*n_/d_    

    def chi2_fn(y, invcov):
        return np.dot(y, np.dot(invcov, y))    

    def read_nnbar(path, ix=None):
        d = np.load(path, allow_pickle=True)
        nnbar = []
        if ix is None:        
            for di in d:
                nnbar.append(di['nnbar']-1)
        else:
            for i in ix:
                di = d[i]
                nnbar.append(di['nnbar']-1)
        return np.array(nnbar).flatten()

    def get_chi2t(path, ix, incov):
        nnbar_ = read_nnbar(path, ix)
        return chi2_fn(nnbar_, invcov)
        
    def get_chi2t_mocks(nside, cap, ix):
        path_ = '/home/mehdi/data/eboss/mocks/1.0/measurements/nnbar/'
        mocks = glob(f'{path_}nnbar_{cap}_knownsystot_mainhighz_512_v7_0_*_main_{nside}.npy')
        print('len(nbars):', len(mocks), cap)
        nmocks = len(mocks)
        err_tot = []
        for j, fn in enumerate(mocks):
            err_j = read_nnbar(fn, ix=ix)
            err_tot.append(err_j)            
        err_tot = np.array(err_tot)
        print(err_tot.shape)

        nbins = err_tot.shape[1]
        hartlapf = (nmocks-1. - 1.) / (nmocks-1. - nbins - 2.)
        indices = [i for i in range(nmocks)]
        chi2s = []
        for i in range(nmocks):
            indices_ = indices.copy()    
            indices_.pop(i)
            nbar_ = err_tot[i, :]
            err_ = err_tot[indices_, :]    
            covmax_ = np.cov(err_, rowvar=False)
            invcov_ = np.linalg.inv(covmax_*hartlapf)
            chi2_ = chi2_fn(nbar_, invcov_)
            chi2s.append(chi2_)       
            
        print(nmocks)
        covmax_ = np.cov(err_tot, rowvar=False)
        hartlapf = (nmocks - 1.) / (nmocks - nbins - 2.)
        invcov_ = np.linalg.inv(covmax_*hartlapf)
        
        return np.array(chi2s), invcov_

    # read covariance matrix
    chi2mocks, invcov = get_chi2t_mocks('512', cap, ix)
    print(np.percentile(chi2mocks, [0, 1, 5, 95, 99, 100]))
    
    path = '/home/mehdi/data/eboss/data/v7_2/3.0/measurements/nnbar/'
    chi2d = {}
    chi2d['noweight'] = get_chi2t(f'{path}nnbar_{cap}_noweight_mainhighz_512_v7_2_main_512.npy', ix, invcov)
    chi2d['standard'] = get_chi2t(f'{path}nnbar_{cap}_knownsystot_mainhighz_512_v7_2_main_512.npy', ix, invcov)
    chi2d['nn'] = get_chi2t(f'{path}nnbar_{cap}_known_mainhighz_512_v7_2_main_512.npy', ix, invcov)
    print(chi2d)
    
    fig = plt.figure(figsize=(8, 5))
    fig.subplots_adjust(wspace=0.03)
    gs  = GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ax1.tick_params(direction='in', axis='both')
    ax2.tick_params(direction='in', axis='both')

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.set(xlim=xlim1, ylim=(0., 200.), xlabel=r'$\chi^{2}_{{\rm tot}}$')
    ax2.set(xlim=xlim2, ylim=(0., 200.), xticks=xticks2)
    ax2.set_yticks([])

    mu, std = np.mean(chi2mocks), np.std(chi2mocks)
    print(f'{mu:.1f} +- {std:.1f}')

    vmin, vmax = 0.9*min(chi2mocks), 1.1*max(chi2mocks)
    bin_width = 3.5*std / np.power(len(chi2mocks), 1./3.) # Scott 1979
    print(bin_width)


    bins = np.arange(vmin, vmax, bin_width)
    #x_ = np.linspace(vmin, vmax, 200)
    #pdf = (1./np.sqrt(2*np.pi)/std)*np.exp(-0.5*((x_-mu)/std)**2)
    #pdf = chi2_pdf(x_, np.floor(mu))*bin_width*len(chi2mocks) 
    
    #ax1.plot(x_, pdf, 'b-')
    ax1.hist(chi2mocks, bins=bins, 
             alpha=0.3, color='b', label=f'{len(chi2mocks)} Null EZMocks', 
             zorder=-10)

    kw2 = dict(rotation=90, fontsize=12)# fontsize, fontweight='bold'
    ls = ['--', '-.', ':']
    for i, (n,v) in enumerate(chi2d.items()):    
        ax_ = ax2 if n=='noweight' else ax1
        ax_.axvline(v, zorder=-1, color='k', 
                   ls=ls[i], alpha=0.5, lw=1) # label='Data %s'%n,

        pval = 100*(chi2mocks > v).mean()
        ax_.text(1.001*v, 10, fr'Data ({n.upper()}) = {v:.1f} ({pval:.1f}%)', **kw2)

    ax1.legend(loc='upper left', frameon=False)

    d = 0.01
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d/2, 1 + d/2), (-d, +d), **kwargs)  # bottom-left diagonal
    kwargs.update(transform=ax2.transAxes)            # switch to the bottom axes
    ax2.plot((-d/2, +d/2), (-d, +d), **kwargs)        # top-left diagonal
    fig.savefig(fig_path, bbox_inches='tight')


def cellchi2pdf_mocks_data(fig_path, cap='NGC', 
                            xlim1=(0., 1000.), xlim2=(2400., 2800.),
                            xticks2=[2500, 2700],
                            ix=None):
    from glob import glob
    from matplotlib.gridspec import GridSpec
    from scipy.special import gamma
    from lssutils.lab import histogram_cell

    def chi2_pdf(x, k):
        """ Chi2 pdf 
        """
        k2 = k / 2.
        n_ = np.power(x, k2-1.)
        d_ = np.power(2., k2)*gamma(k2)
        return np.exp(-0.5*x)*n_/d_    

    def chi2_fn(y, invcov):
        return np.dot(y, np.dot(invcov, y))    

    def read_cl(fn, ix=[i for i in range(17)], colmax=None):

        cl = np.load(fn, allow_pickle=True).item()
        cl_cross = []
        cl_ss = []

        for i in ix:    
            l_, cl_sg_ = histogram_cell(cl['cl_sg'][i]['cl'])
            l_, cl_ss_ = histogram_cell(cl['cl_ss'][i]['cl'])

            cl_ss.append(cl_ss_)
            cl_cross.append(cl_sg_[:colmax]**2/cl_ss_[:colmax])    

        return l_[:colmax], np.array(cl_cross), cl_ss
    
    def read_clmocks(fn, cl_ss, ix=[i for i in range(17)], colmax=None):
        cl = np.load(fn, allow_pickle=True).item()
        cl_cross = []

        for i in ix:    
            l_, cl_sg_ = histogram_cell(cl['cl_sg'][i]['cl'])
            cl_ss_ = cl_ss[i]
            cl_cross.append(cl_sg_[:colmax]**2/cl_ss_[:colmax])    

        return l_[:colmax], np.array(cl_cross)
    
    # data
    p = '/home/mehdi/data/eboss/data/v7_2/3.0/measurements/cl/'
    cl = {}
    cl['noweight'] = read_cl(f'{p}cl_{cap}_noweight_mainhighz_512_v7_2_main_512.npy', colmax=4)
    cl['standard'] = read_cl(f'{p}cl_{cap}_knownsystot_mainhighz_512_v7_2_main_512.npy', colmax=4)
    cl['nn'] = read_cl(f'{p}cl_{cap}_known_mainhighz_512_v7_2_main_512.npy', colmax=4)    
    
    print('ell=', cl['nn'][0])
    # mocks
    p = '/home/mehdi/data/eboss/mocks/1.0/measurements/cl/'
    mocks = glob(f'{p}cl_{cap}_knownsystot_mainhighz_512_v7_0_*')
    print(len(mocks))
    
    clmocks = []
    for mock_i in mocks:
        clmock_ = read_clmocks(mock_i, cl['nn'][2], colmax=4)[1]
        clmocks.append(clmock_.flatten())        
    err_tot = np.array(clmocks)
    nmocks, nbins = err_tot.shape
    hartlapf = (nmocks-1. - 1.) / (nmocks-1. - nbins - 2.)
    indices = [i for i in range(nmocks)]

    # chi2 of mocks
    chi2s = []
    for i in range(nmocks):
        indices_ = indices.copy()    
        indices_.pop(i)

        nbar_ = err_tot[i, :]
        err_ = err_tot[indices_, :]    
        covmax_ = np.cov(err_, rowvar=False)
        invcov_ = np.linalg.inv(covmax_*hartlapf)

        chi2_ = chi2_fn(nbar_, invcov_)
        chi2s.append(chi2_)  
    chi2mocks = np.array(chi2s)        
    print(np.percentile(chi2mocks, [0, 1, 5, 95, 99, 100]))
        
    # chi2 of data    
    covmax_ = np.cov(err_tot, rowvar=False)
    hartlapf = (nmocks - 1.) / (nmocks - nbins - 2.)
    invcov_ = np.linalg.inv(covmax_*hartlapf)
    
    chi2d = {}
    for i, (name, cl_) in enumerate(cl.items()):
        chi2d[name] = chi2_fn(cl_[1].flatten(), invcov_)
    print(chi2d)
    
    fig = plt.figure(figsize=(8, 5))
    fig.subplots_adjust(wspace=0.03)
    gs  = GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ax1.tick_params(direction='in', axis='both')
    ax2.tick_params(direction='in', axis='both')

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.set(xlim=xlim1, ylim=(0.7, 900.), yscale='log', xlabel=r'$\chi^{2}_{{\rm tot}}$')
    ax2.set(xlim=xlim2, ylim=(0.7, 900.), yscale='log', xticks=xticks2)
    ax2.tick_params(left=False, which='both')
    ax2.set_yticks([])

    mu, std = np.mean(chi2mocks), np.std(chi2mocks)
    print(f'{mu:.1f} +- {std:.1f}')

    vmin, vmax = 0.9*min(chi2mocks), 1.1*max(chi2mocks)
    bin_width = 3.5*std / np.power(len(chi2mocks), 1./3.) # Scott 1979
    print(bin_width)

    bins = np.arange(vmin, vmax, bin_width)
    #x_ = np.linspace(vmin, vmax, 200)
    #pdf = (1./np.sqrt(2*np.pi)/std)*np.exp(-0.5*((x_-mu)/std)**2)
    #pdf = chi2_pdf(x_, np.floor(mu))*bin_width*len(chi2mocks) 
    
    #ax1.plot(x_, pdf, '-', color='orange')
    ax1.hist(chi2mocks, bins=bins, 
             alpha=0.3, color='orange', label=f'{nmocks} Null EZMocks', 
             zorder=-10)

    kw2 = dict(rotation=90, fontsize=12)# fontsize, fontweight='bold'
    ls = ['--', '-.', ':']
    for i, (n,v) in enumerate(chi2d.items()):    
        ax_ = ax2 if n=='noweight' else ax1
        ax_.axvline(v, zorder=-1, color='k', 
                   ls=ls[i], alpha=0.5, lw=1) # label='Data %s'%n,
        pval = 100*(chi2mocks > v).mean()
        ax_.text(1.0*v, 1, fr'Data ({n.upper()}) = {v:.1f} ({pval:.1f}%)', **kw2)

    ax1.legend(loc='upper left', frameon=False)

    d = 0.01
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d/2, 1 + d/2), (-d, +d), **kwargs)  # bottom-left diagonal
    kwargs.update(transform=ax2.transAxes)            # switch to the bottom axes
    ax2.plot((-d/2, +d/2), (-d, +d), **kwargs)        # top-left diagonal
    fig.savefig(fig_path, bbox_inches='tight')

    

def pcc_wnn_nchains(fig_path, ns='512'):

    from scipy.stats import pearsonr
    import fitsio as ft

    

    c = ['k', 'crimson']    
    mk = ['.', 'x']


    fig, ax = plt.subplots()
    ax.tick_params(direction='in', which='both', top=True, right=True)   

    mp = 'known'
    for j, cap in enumerate(['NGC', 'SGC']):
        
        root_path = f'/home/mehdi/data/eboss/data/v7_2/3.0/{cap}'
        
        for k, s in enumerate(['main', 'highz']):
            
            d = ft.read(f'{root_path}/{ns}/{s}/nn_pnll_{mp}/nn-weights.fits') # (Npix, Nchains)

            fc = 'w' #if k==0 else c[j]
            
            x = []
            pcc =  []
            for i in range(1, 11):

                set_1 = d['weight'][:, 0:i].mean(axis=1)
                set_2 = d['weight'][:, 10:i+10].mean(axis=1)

                pcc_ = pearsonr(set_1, set_2)[0]
                
                x.append(i)
                pcc.append(pcc_)
                #print(f'i={i}, pcc={pcc:.3f}')
                

            ax.plot(x, pcc, color=c[j], marker=mk[k], mfc=fc, ls='None')

        ax.text(0.8, 0.5-j*0.08, f'{cap}', color=c[j], transform=ax.transAxes)

    ax.set(xlabel='Nchains', ylabel=r'PCC', xticks=np.arange(1, 11)) 
    
    lines = ax.get_lines()
    print(len(lines))
    legend1 = plt.legend(lines[:2], ['Main', 'High-z'])
    ax.add_artist(legend1)  
    fig.savefig(fig_path, bbox_inches='tight')

def plot_nstar(fig_path):
    fig, ax = plt.subplots(figsize=(6, 8), nrows=2)
    
    ax[0] = nbarnstar(ax[0])
    ax[1] = p0nstar(ax[1])
    
    fig.align_labels()
    fig.savefig(fig_path, bbox_inches='tight')
    
    
    
def nbarnstar(ax):
    
    incov_ngc = NbarCov('NGC', '512')
    ix = 0                             # nstar
    _, covmax = incov_ngc.get_invcov(ix*8, (ix+1)*8, return_covmax=True)
    y_err = np.sqrt(np.diag(covmax))
        
    #p = '/home/mehdi/data/eboss/data/v7_2/3.0/measurements/nnbar/'
    #d = np.load(f'{p}nnbar_NGC_main_512_nstar.npy', allow_pickle=True).item()

    d = {}
    p = '/home/mehdi/data/eboss/data/v7_2/3.0/measurements/nnbar/'
    d['Standard'] = np.load(f'{p}nnbar_NGC_knownsystot_mainhighz_512_v7_2_main_512.npy', allow_pickle=True)
    p = '/home/mehdi/data/eboss/data/v7_2/1.0/measurements/nnbar/'
    d['NN (known)'] = np.load(f'{p}nnbar_NGC_known_mainhighz_512_v7_2_main_512.npy', allow_pickle=True)
    d['NN (all)'] = np.load(f'{p}nnbar_NGC_all_mainhighz_512_v7_2_main_512.npy', allow_pickle=True)
    d['NN (known+sdss)'] = np.load(f'{p}nnbar_NGC_known_mainstar_512_v7_2_main_512.npy', allow_pickle=True)
    d['NN (known+gaia)'] = np.load(f'{p}nnbar_NGC_known_mainstarg_512_v7_2_main_512.npy', allow_pickle=True)
    
    
    ls = 4*['-', '--', '-.']
    mk = 4*['o', 's', '^', '.', 'x', '*']
    si = r'Nstar [Gaia DR2]'
    for j, (ni,di) in enumerate(d.items()):        
        #if ni == 'nn-pnll':        
        #    ax.errorbar(di[ix]['bin_avg'], di[ix]['nnbar']-1, y_err, 
        #                  capsize=3, marker=mk[j], mfc='w', ls=ls[j], label=ni, alpha=0.8)        
        #else:
        ax.plot(di[ix]['bin_avg'], di[ix]['nnbar']-1,
                marker=mk[j], mfc='w', ls=ls[j], label=ni) #

    ax.fill_between(di[ix]['bin_avg'], -y_err, y_err, color='grey', alpha=0.1)        

#     for j, ni in enumerate(['standard', 'NN-known', 'NN-all', 'NN-known+sdss', 'NN-known+gaia']):  
#         di = d[ni]
            
#         #if ni == 'NN-known+gaia':        
#         #    ax.errorbar(di['bin_avg'], di['nnbar']-1, y_err, 
#         #                  capsize=3, marker=mk[j], mfc='w', ls=ls[j], label=ni, alpha=0.8)        
#         #else:
#         ax.plot(di['bin_avg'], di['nnbar']-1,
#                     marker=mk[j], mfc='w', ls=ls[j], label=ni, alpha=0.8)
#     ax.fill_between(di['bin_avg'], -y_err, y_err, color='grey', alpha=0.1)
    
    ax.axhline(0.0, ls=':', lw=1, color='k')
    ax.set_xlabel(f'{si}')
    ax.legend(fontsize=12, ncol=2, frameon=False)   
    ax.set_ylim(-0.025, 0.025)
    ax.tick_params(direction='in', which='both', axis='both', right=True, top=True)    
    ax.set_ylabel(r'$\delta$')
    #fig.savefig(fig_path, bbox_inches='tight')
    return ax
    
    
def p0nstar(ax):
    import nbodykit.lab as nb
    def read(file):
        dt = nb.ConvolvedFFTPower.load(file)
        return (dt.poles['k'], dt.poles['power_0'].real-dt.attrs['shotnoise']) 
    
    cap = 'NGC'
    #pathm = '/home/mehdi/data/eboss/mocks/1.0/measurements/spectra/'
    #pkcov = np.loadtxt(f'{pathm}/spectra_{cap}_knownsystot_mainhighz_512_v7_0_1to1000_6600_512_main.dat')
    #pk_err = np.sqrt(np.diagonal(pkcov))
    covpks_all = np.load('./covpk_ezmocks_ngcsgc_v1.0.npz', allow_pickle=True)
    covpks = covpks_all[f'{cap}_null_standard_512'].item()    
    pk_err = np.sqrt(np.diagonal(covpks['covp0']))  

    
    p_ = '/home/mehdi/data/eboss/data/v7_2/3.0/measurements/spectra/'    
    p = '/home/mehdi/data/eboss/data/v7_2/1.0/measurements/spectra/'

    d = {}
    #d['noweight'] = read(f'{p_}spectra_NGC_noweight_mainhighz_512_v7_2_main.json')
    d['Standard'] = read(f'{p_}spectra_NGC_knownsystot_mainhighz_512_v7_2_main.json')          
    d['NN (known)'] = read(f'{p}spectra_NGC_known_mainhighz_512_v7_2_main.json')
    d['NN (all)'] = read(f'{p}spectra_NGC_all_mainhighz_512_v7_2_main.json')    
    d['NN (known+sdss)'] = read(f'{p}spectra_NGC_known_mainstar_512_v7_2_main.json')
    d['NN (known+gaia)'] = read(f'{p}spectra_NGC_known_mainstarg_512_v7_2_main.json')

    
    #fig, ax = plt.subplots(figsize=(6, 4), sharey=True)
    #fig.subplots_adjust(wspace=0.0)


    ls = 4*['-', '--', '-.']
    mk = 4*['o', 's', '^', '.', 'x', '*']

    for j, (ni,di) in enumerate(d.items()):   
        
        if ni == 'NN (known+gaia)':        
            ax.errorbar(di[0], 1.0e-4*di[1], yerr=1.0e-4*pk_err, marker=mk[j], mfc='w', ls=ls[j], 
                    label=ni, capsize=3) # alpha=0.8, 
        else:
            ax.plot(di[0], 1.0e-4*di[1], marker=mk[j], mfc='w', ls=ls[j], label=ni) #alpha=0.8)


    ax.set(xlabel='k [h/Mpc]', xscale='log') #yscale='log', 
    ax.legend(fontsize=12, frameon=False, ncol=2)    
    ax.tick_params(direction='in', which='both', axis='both', right=True, top=True)
    ax.set_ylabel(r'P$_{0}$ [$10^{4}$ (Mpc/h)$^{3}$]')
    #fig.savefig(fig_path, dpi=300, bbox_inches='tight')    
    return ax
    
def plot_linnn(fig_path):
    fig, ax = plt.subplots(figsize=(6, 8), nrows=2)
    
    ax[0] = nbarlinnn(ax[0])
    ax[1] = p0linnn(ax[1])
    
    fig.align_labels()
    fig.savefig(fig_path, bbox_inches='tight')    

    
def nbarlinnn(ax):

    incov_ngc = NbarCov('NGC', '512')
    ix = 1 # EBV
    _, covmax = incov_ngc.get_invcov(ix*8, (ix+1)*8, return_covmax=True)
    y_err = np.sqrt(np.diag(covmax))
    
    p = '/home/mehdi/data/eboss/data/v7_2/1.0/measurements/nnbar/'

    d = {}
    d['Standard'] = np.load(f'{p}nnbar_NGC_knownsystot_mainhighz_512_v7_2_main_512.npy', allow_pickle=True)    
    d['LIN (mse)'] = np.load(f'{p}nnbar_NGC_known_mainlinmse_512_v7_2_main_512.npy', allow_pickle=True)
    d['LIN (pnll)'] = np.load(f'{p}nnbar_NGC_known_mainlinp_512_v7_2_main_512.npy', allow_pickle=True)
    d['NN (mse)'] = np.load(f'{p}nnbar_NGC_known_mainmse_512_v7_2_main_512.npy', allow_pickle=True)
    d['NN (pnll)'] = np.load(f'{p}nnbar_NGC_known_mainhighz_512_v7_2_main_512.npy', allow_pickle=True)
    d['NN (pnll-lr)'] = np.load(f'{p}nnbar_NGC_known_mainwocos_512_v7_2_main_512.npy', allow_pickle=True)

    
    #fig, ax = plt.subplots(figsize=(6, 4), sharey=True)
    #fig.subplots_adjust(wspace=0.0)
    
    ls = 4*['-', '--', '-.']
    mk = 4*['o', 's', '^', '.', 'x', '*']
    si = r'E[B-V]'
    for j, (ni,di) in enumerate(d.items()):        
        #if ni == 'nn-pnll':        
        #    ax.errorbar(di[ix]['bin_avg'], di[ix]['nnbar']-1, y_err, 
        #                  capsize=3, marker=mk[j], mfc='w', ls=ls[j], label=ni, alpha=0.8)        
        #else:
        ax.plot(di[ix]['bin_avg'], di[ix]['nnbar']-1,
                marker=mk[j], mfc='w', ls=ls[j], label=ni) #, alpha=0.8)

    ax.fill_between(di[ix]['bin_avg'], -y_err, y_err, color='grey', alpha=0.1)        
    ax.axhline(0.0, ls=':', lw=1, color='k')
    ax.set_xlabel(f'{si}')
    ax.legend(fontsize=11, frameon=False, ncol=2, loc='upper right')    
    ax.set_ylabel(r'$\delta$')
    ax.set_ylim(-0.025, 0.025)
    ax.set_xlim(xmax=0.055)
    ax.tick_params(direction='in', which='both', axis='both', right=True, top=True)    
    #fig.savefig(fig_path, bbox_inches='tight')    
    return ax
    
def p0linnn(ax):
    
    
    import nbodykit.lab as nb
    def read(file):
        dt = nb.ConvolvedFFTPower.load(file)
        return (dt.poles['k'], dt.poles['power_0'].real-dt.attrs['shotnoise']) 
    
    cap = 'NGC'
    #pathm = '/home/mehdi/data/eboss/mocks/1.0/measurements/spectra/'
    #pkcov = np.loadtxt(f'{pathm}/spectra_{cap}_knownsystot_mainhighz_512_v7_0_1to1000_6600_512_main.dat')
    #pk_err = np.sqrt(np.diagonal(pkcov))    
    covpks_all = np.load('./covpk_ezmocks_ngcsgc_v1.0.npz', allow_pickle=True)
    covpks = covpks_all[f'{cap}_null_standard_512'].item()    
    pk_err = np.sqrt(np.diagonal(covpks['covp0']))      
    
    p = '/home/mehdi/data/eboss/data/v7_2/1.0/measurements/spectra/'
    p_ = '/home/mehdi/data/eboss/data/v7_2/3.0/measurements/spectra/'    
    d = {}

    d['Standard'] = read(f'{p_}spectra_NGC_knownsystot_mainhighz_512_v7_2_main.json')  
    #d['noweight'] = read(f'{p_}spectra_NGC_noweight_mainhighz_512_v7_2_main.json')
    d['LIN (mse)'] = read(f'{p}spectra_NGC_known_mainlinmse_512_v7_2_main.json')
    d['LIN (pnll)'] = read(f'{p}spectra_NGC_known_mainlinp_512_v7_2_main.json')
    d['NN (mse)'] = read(f'{p}spectra_NGC_known_mainmse_512_v7_2_main.json')
    d['NN (pnll)'] = read(f'{p}spectra_NGC_known_mainhighz_512_v7_2_main.json')
    d['NN (pnll-lr)'] = read(f'{p}spectra_NGC_known_mainwocos_512_v7_2_main.json')
    
    
    
    #fig, ax = plt.subplots(figsize=(6, 4), sharey=True)
    #fig.subplots_adjust(wspace=0.0)


    ls = 4*['-', '--', '-.']
    mk = 4*['o', 's', '^', '.', 'x', '*']

    for j, (ni,di) in enumerate(d.items()):        
        if ni == 'NN (pnll-lr)':
            ax.errorbar(di[0], 1.0e-4*di[1], 1.0e-4*pk_err, marker=mk[j], capsize=3,
                        mfc='w', ls=ls[j], label=ni) #alpha=0.8)
        else:
            ax.plot(di[0], 1.0e-4*di[1], marker=mk[j], mfc='w', ls=ls[j], label=ni)#, alpha=0.8)


    ax.set(xlabel='k [h/Mpc]', xscale='log')
    ax.legend(fontsize=12, frameon=False, ncol=2)    
    ax.set_ylabel(r'P$_{0}$ [$10^{4}$ (Mpc/h)$^{3}$]')
    ax.tick_params(direction='in', which='both', axis='both', right=True, top=True)
    #fig.savefig(fig_path, dpi=300, bbox_inches='tight')        
    return ax


def plot_nsidexz(fig_path):
    fig, ax = plt.subplots(figsize=(6, 8), nrows=2)
    
    ax[0] = nbarnside(ax[0])
    ax[1] = p0nside(ax[1])
    
    fig.align_labels()
    fig.savefig(fig_path, bbox_inches='tight') 
    
    
def nbarnside(ax):

    incov_ngc = NbarCov('NGC', '512')
    ix = 7 # depth-g
    _, covmax = incov_ngc.get_invcov(ix*8, (ix+1)*8, return_covmax=True)
    y_err = np.sqrt(np.diag(covmax))
    
    p = '/home/mehdi/data/eboss/data/v7_2/3.0/measurements/nnbar/'
    d = {}
    d['Standard'] = np.load(f'{p}nnbar_NGC_knownsystot_mainhighz_512_v7_2_main_512.npy', allow_pickle=True)
    d['NN (512-1z)'] = np.load(f'{p}nnbar_NGC_known_mainhighz_512_v7_2_main_512.npy', allow_pickle=True)
    d['NN (512-2z)'] = np.load(f'{p}nnbar_NGC_known_lowmidhighz_512_v7_2_main_512.npy', allow_pickle=True)
    d['NN (256-1z)'] = np.load(f'{p}nnbar_NGC_known_mainhighz_256_v7_2_main_512.npy', allow_pickle=True)
    
    #fig, ax = plt.subplots(figsize=(6, 4), sharey=True)
    #fig.subplots_adjust(wspace=0.0)
    
    ls = 4*['-', '--', '-.']
    mk = 4*['o', 's', '^', '.', 'x', '*']
    si = r'Depth-g'
    for j, (ni,di) in enumerate(d.items()):        
        #if ni == 'NN 512-1z':        
        #    ax.errorbar(di[ix]['bin_avg'], di[ix]['nnbar']-1, y_err, 
        #                  capsize=3, marker=mk[j], mfc='w', ls=ls[j], label=ni, alpha=0.8)        
        #else:
        ax.plot(di[ix]['bin_avg'], di[ix]['nnbar']-1,
                marker=mk[j], mfc='w', ls=ls[j], label=ni)#, alpha=0.8)
        
    ax.fill_between(di[ix]['bin_avg'], -y_err, y_err, color='grey', alpha=0.1)
    ax.axhline(0.0, ls=':', lw=1, color='k')
    ax.set_xlabel(f'{si}')
    ax.set_ylim(-0.025, 0.025)
    ax.tick_params(direction='in', which='both', axis='both', right=True, top=True)    
    ax.legend(fontsize=12, ncol=2, loc='lower right', frameon=False)    
    ax.set_ylabel(r'$\delta$')
    #fig.savefig(fig_path, bbox_inches='tight') 
    return ax

def p0nside(ax):
    import nbodykit.lab as nb
    def read(file):
        dt = nb.ConvolvedFFTPower.load(file)
        return (dt.poles['k'], dt.poles['power_0'].real-dt.attrs['shotnoise']) 
    
    cap = 'NGC'
    #pathm = '/home/mehdi/data/eboss/mocks/1.0/measurements/spectra/'    
    #pkcov = np.loadtxt(f'{pathm}/spectra_{cap}_knownsystot_mainhighz_512_v7_0_1to1000_6600_512_main.dat')
    covpks_all = np.load('./covpk_ezmocks_ngcsgc_v1.0.npz', allow_pickle=True)
    covpks = covpks_all[f'{cap}_null_standard_512'].item()    
    pk_err = np.sqrt(np.diagonal(covpks['covp0']))    
    
    
    p = '/home/mehdi/data/eboss/data/v7_2/3.0/measurements/spectra/'

    d = {}
    d['Standard'] = read(f'{p}spectra_NGC_knownsystot_mainhighz_512_v7_2_main.json')
    d['NN (512-1z)'] = read(f'{p}spectra_NGC_known_mainhighz_512_v7_2_main.json')
    d['NN (512-2z)'] = read(f'{p}spectra_NGC_known_lowmidhighz_512_v7_2_main.json')
    d['NN (256-1z)'] = read(f'{p}spectra_NGC_known_mainhighz_256_v7_2_main.json')    

    
    
    #fig, ax = plt.subplots(figsize=(6, 4), sharey=True)
    #fig.subplots_adjust(wspace=0.0)

    ls = 4*['-', '--', '-.']
    mk = 4*['o', 's', '^', '.', 'x', '*']

    for j, (ni,di) in enumerate(d.items()):        
        if ni == 'NN (256-1z)':
            ax.errorbar(di[0], 1.0e-4*di[1], 1.0e-4*pk_err, marker=mk[j], capsize=3,
                        mfc='w', ls=ls[j], label=ni) #alpha=0.8)
        else:
            ax.plot(di[0], 1.0e-4*di[1], marker=mk[j], mfc='w', ls=ls[j], label=ni) #, alpha=0.8)


    ax.set(xlabel='k [h/Mpc]', xscale='log')
    ax.legend(fontsize=12, frameon=False, ncol=2)   
    ax.tick_params(direction='in', which='both', axis='both', right=True, top=True)    
    ax.set_ylabel(r'P$_{0}$ [$10^{4}$ (Mpc/h)$^{3}$]')
    #fig.savefig(fig_path, dpi=300, bbox_inches='tight')      
    return ax
    

def table_chi2methods():
    from glob import glob
    def chi2_fn(y, invcov):
        return np.dot(y, np.dot(invcov, y))    

    def read_nnbar(path, ix=None):
        d = np.load(path, allow_pickle=True)
        nnbar = []
        if ix is None:        
            for di in d:
                nnbar.append(di['nnbar']-1)
        else:
            for i in ix:
                di = d[i]
                nnbar.append(di['nnbar']-1)
        return np.array(nnbar).flatten()

    def get_chi2t(path, ix, incov):
        nnbar_ = read_nnbar(path, ix)
        return chi2_fn(nnbar_, invcov)

    def get_chi2t_mocks(nside, cap, ix):
        path_ = '/home/mehdi/data/eboss/mocks/1.0/measurements/nnbar/'
        mocks = glob(f'{path_}nnbar_{cap}_knownsystot_mainhighz_512_v7_0_*_main_{nside}.npy')
        print('len(nbars):', len(mocks), cap)
        nmocks = len(mocks)
        err_tot = []
        for j, fn in enumerate(mocks):
            err_j = read_nnbar(fn, ix=ix)
            err_tot.append(err_j)            
        err_tot = np.array(err_tot)
        print(err_tot.shape)

        nbins = err_tot.shape[1]
        hartlapf = (nmocks-1. - 1.) / (nmocks-1. - nbins - 2.)
        indices = [i for i in range(nmocks)]
        chi2s = []
        for i in range(nmocks):
            indices_ = indices.copy()    
            indices_.pop(i)
            nbar_ = err_tot[i, :]
            err_ = err_tot[indices_, :]    
            covmax_ = np.cov(err_, rowvar=False)
            invcov_ = np.linalg.inv(covmax_*hartlapf)
            chi2_ = chi2_fn(nbar_, invcov_)
            chi2s.append(chi2_)       

        print(nmocks)
        covmax_ = np.cov(err_tot, rowvar=False)
        hartlapf = (nmocks - 1.) / (nmocks - nbins - 2.)
        invcov_ = np.linalg.inv(covmax_*hartlapf)

        return np.array(chi2s), invcov_

    # read covariance matrix
    cap = 'NGC'
    ix = None # [i for i in range(1, 17)]
    chi2mocks, invcov = get_chi2t_mocks('512', cap, ix)
    print(np.percentile(chi2mocks, [0, 1, 5, 95, 99, 100]))
    
    
    path = '/home/mehdi/data/eboss/data/v7_2/3.0/measurements/nnbar/'

    chi2d = {}
    chi2d['noweight'] = get_chi2t(f'{path}nnbar_{cap}_noweight_mainhighz_512_v7_2_main_512.npy', ix, invcov)
    chi2d['standard'] = get_chi2t(f'{path}nnbar_{cap}_knownsystot_mainhighz_512_v7_2_main_512.npy', ix, invcov)
    # chi2d['nn'] = get_chi2t(f'{path}nnbar_{cap}_known_mainhighz_512_v7_2_main_512.npy', ix, invcov)
    # chi2d['standard2'] = get_chi2t(f'{path}nnbar_NGC_knownsystot_mainhighz_512_v7_2_main_512.npy', ix, invcov)
    chi2d['NN 512-1z'] = get_chi2t(f'{path}nnbar_NGC_known_mainhighz_512_v7_2_main_512.npy', ix, invcov)
    chi2d['NN 512-2z'] = get_chi2t(f'{path}nnbar_NGC_known_lowmidhighz_512_v7_2_main_512.npy', ix, invcov)
    chi2d['NN 256-1z'] = get_chi2t(f'{path}nnbar_NGC_known_mainhighz_256_v7_2_main_512.npy', ix, invcov)

    p = '/home/mehdi/data/eboss/data/v7_2/1.0/measurements/nnbar/'
    # chi2d['standard1'] = get_chi2t(f'{p}nnbar_NGC_knownsystot_mainhighz_512_v7_2_main_512.npy', ix, invcov)
    chi2d['lin-mse'] = get_chi2t(f'{p}nnbar_NGC_known_mainlinmse_512_v7_2_main_512.npy', ix, invcov)
    chi2d['lin-pnll'] = get_chi2t(f'{p}nnbar_NGC_known_mainlinp_512_v7_2_main_512.npy', ix, invcov)
    chi2d['nn-mse'] = get_chi2t(f'{p}nnbar_NGC_known_mainmse_512_v7_2_main_512.npy', ix, invcov)
    chi2d['nn-pnll'] = get_chi2t(f'{p}nnbar_NGC_known_mainhighz_512_v7_2_main_512.npy', ix, invcov)
    chi2d['nn-pnll-wocos'] = get_chi2t(f'{p}nnbar_NGC_known_mainwocos_512_v7_2_main_512.npy', ix, invcov)
    chi2d['nn-pnll-all'] = get_chi2t(f'{p}nnbar_NGC_all_mainhighz_512_v7_2_main_512.npy', ix, invcov)
    chi2d['nn-known+gaia'] = get_chi2t(f'{p}nnbar_NGC_known_mainstarg_512_v7_2_main_512.npy', ix, invcov)
    chi2d['nn-known+sdss'] = get_chi2t(f'{p}nnbar_NGC_known_mainstar_512_v7_2_main_512.npy', ix, invcov)
    
    for n, v in chi2d.items():
        print(f'{n}, {v:.4f}')

        
def dpvp(fig_path):
    import sys
    sys.path.append('/home/mehdi/github/LSSutils/scripts/analysis/')
    from compute_mitigationbias import Spectra, solve
    
    ngc1 = Spectra(cap='NGC', nside='512')
    ngc1.load('known', '1')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    

    colors = ['#377eb8', '#ff7f00', '#4daf4a',
             '#984ea3', '#a65628', '#f781bf',
              '#999999', '#e41a1c']
    ls = ['-', '--', '-.', ':']
    vs = 0
    x_plot = None
    for j in [0, 1, 2, 3]:

        color = colors[j]

        bins_p = np.percentile(ngc1.pks[:, vs, j], [0, 20, 40, 60, 80, 100])
        bins_p[-1] += 1.0e-4
        if x_plot is None:
            x_plot = bins_p

        ix = np.digitize(ngc1.pks[:, vs, j], bins=bins_p)

        x_ = []
        y_ = []
        for i in ix:
            s_i = ix == i
            x_.append(np.median(ngc1.pks[s_i, vs, j], axis=0))
            y_.append(np.median(ngc1.pks[s_i, 2, j], axis=0))

        x_ = np.array(x_)
        y_ = np.array(y_)

        m,b  = solve(x_, y_)            


        ax.scatter(1.0e-4*ngc1.pks[:, vs, j], 1.0e-4*ngc1.pks[:, 2, j], 2, marker='.', alpha=1., color=color, zorder=-10+j)
        if j == 0:
            ax.scatter(1.0e-4*x_, 1.0e-4*y_, 50, marker='s', fc='w', color=color, zorder=10)

        ax.plot(1.0e-4*x_plot, 1.0e-4*(m*x_plot + b), ls=ls[j], 
                color=color, lw=3., alpha=.9, label=f'k = {ngc1.k[j]:.3f} h/Mpc', zorder=-20+j)
        print(ngc1.k[j], m, b) 


    ax.set_xlabel(r'P$_{0, {\rm Truth}}$ [$10^{4}$(Mpc/h)$^{3}$]', fontsize=20)
    ax.set_ylabel(r'$\Delta$P$_{0}$ [$10^{4}$(Mpc/h)$^{3}$]', fontsize=20)
    ax.legend(fontsize=17, handlelength=2.5, frameon=False)
    ax.text(0.1, 0.1, 'NGC EZmocks', color='grey', transform=ax.transAxes, fontsize=20)
    ax.tick_params(direction='in', which='both', top=True, right=True) 
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    
def p0mocks(fig_path):
    
    pkm = np.load('pk_ezmocks_ngcsgc_v1.0.npz', allow_pickle=True)    
    covpk = np.load('covpk_ezmocks_ngcsgc_v1.0.npz', allow_pickle=True)
    
    fig, ax = plt.subplots(figsize=(12, 16), nrows=4, ncols=2, sharex=True, sharey='row')
    ax = ax.flatten()
    fig.subplots_adjust(hspace=0.0, wspace=0.0) # like subplot adjust


    ylabels = {#:r'P [Mpc/h]$^{3}$',
               0:r'P$_{X}$-P$_{\rm Truth}$ [$10^{4}$(Mpc/h)$^{3}$]',
               2:r'(P$_{X}$-P$_{\rm Truth}$)/|P$_{\rm Truth}|$', #[10$^{4}$ (Mpc/h)$^{3}$]
               4:r'$\sigma$P$_{X}$ [$10^{4}$(Mpc/h)$^{3}$]',
               6:r'$\sigma$P$_{X}$/|P$_{X}$|'}

    for j,cap in enumerate(['NGC', 'SGC']):        
        axes = [2*p+j for p in range(4)]

        nmocks = 1000 if cap=='NGC' else 999
        for method, name in zip([f'{cap}_null_standard_512', 
                                 f'{cap}_cont_standard_512',
                                 f'{cap}_cont_nnknown_512',
                                 f'{cap}_null_nnknown_512'], 
                                ['Truth', 
                                 'Cont after Standard',
                                 'Cont after NN', 
                                 'Truth after NN']):

            k = pkm[method][:, 0]
            ptruth = 1.0e-4*pkm[f'{cap}_null_standard_512'][:, 2]
            pkm_ = 1.0e-4*pkm[method][:, 2]
            dp = (pkm_-ptruth)
            dpr = (pkm_-ptruth)/abs(ptruth)
            sigP = 1.0e-4*np.sqrt(np.diagonal(covpk[method].item()['covp0']))[:]
            sigPr = sigP / np.abs(pkm_)

            mk = '.' if method!=f'{cap}_null_standard_512' else 'None'
            for i,y in zip(axes,
                           [dp, dpr, sigP, sigPr]):
                ax[i].plot(k, y, marker=mk, mfc='w', label=name)


            if method == f'{cap}_null_standard_512':
                ax[axes[0]].fill_between(k, -sigP, sigP, alpha=0.1, zorder=-10, color='#377eb8')        
                ax[axes[0]].fill_between(k, -sigP/np.sqrt(nmocks), sigP/np.sqrt(nmocks), 
                                         alpha=0.2, zorder=-8, color='#377eb8')            
                ax[axes[1]].fill_between(k, -sigP/abs(ptruth), sigP/abs(ptruth), alpha=0.1, zorder=-10, color='#377eb8')        
                ax[axes[1]].fill_between(k, -sigP/np.sqrt(nmocks)/abs(ptruth), sigP/np.sqrt(nmocks)/abs(ptruth), 
                                         alpha=0.2, zorder=-8, color='#377eb8')
    ax[1].legend(frameon=False, fontsize=16)
    for i in range(8):
        if i > 5:ax[i].set_xlabel(r'k [h/Mpc]', fontsize=19)
        if i%2==0:ax[i].set_ylabel(ylabels[i], fontsize=19)
        ax[i].set_xscale('log')
        ax[i].tick_params(direction='in', which='both', right=True, top=True)

    fig.align_labels()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')    
    
# def table_chi2():
#     """ chi2 table for main/highz before and after mitigation and 95th mocks 
#     """
#     from glob import glob

#     def get_chi2t(nbar_fn):
#         d = np.load(nbar_fn, allow_pickle=True)
#         chi2_t = 0.0
#         for di in d:
#             res = (di['nnbar'] - 1)/di['nnbar_err']
#             res_sq = res*res
#             chi2_t += res_sq.sum()
#         return chi2_t

#     def get_chi2t_mocks(nside, cap):
#         path = '/home/mehdi/data/eboss/mocks/1.0/measurements/nnbar/'
#         nbars = glob(f'{path}nnbar_{cap}_knownsystot_mainhighz_512_v7_0_*_main_{nside}.npy')
#         print('len(nbars):', len(nbars), cap)

#         chi2_mocks = []
#         for nbar_i in nbars:        
#             chi2_t = get_chi2t(nbar_i)        
#             chi2_mocks.append(chi2_t)
#             #print('.', end='')

#         return np.array(chi2_mocks)
#     print()
    
#     path = '/home/mehdi/data/eboss/data/v7_2/1.0/measurements/nnbar/'

#     chi2_mocks = {}
#     chi2d = {}

#     for cap in ['NGC', 'SGC']:

#         chi2_mocks[cap] = get_chi2t_mocks('512', cap)

#         for sample in ['main', 'highz']:

#             chi2d[f'{cap}_noweight_{sample}'] = get_chi2t(f'{path}nnbar_{cap}_noweight_mainhighz_512_v7_2_{sample}_512.npy')
#             chi2d[f'{cap}_systot_{sample}'] = get_chi2t(f'{path}nnbar_{cap}_knownsystot_mainhighz_512_v7_2_{sample}_512.npy')
#             chi2d[f'{cap}_nn_{sample}'] = get_chi2t(f'{path}nnbar_{cap}_known_mainhighz_512_v7_2_{sample}_512.npy')

#     for cap in ['NGC', 'SGC']:

#         for sample in ['main', 'highz']:

#             msg = f'{cap}, {sample}, '

#             for method in ['noweight', 'systot', 'nn']:
#                 s_ = f'{cap}_{method}_{sample}'
#                 msg += f'& {chi2d[s_]:.1f} '

#             if sample == 'main':
#                 msg += f'& {np.percentile(chi2_mocks[cap], 95):.1f}\\\ \n'
#             else:
#                 msg += '& -- \\\ \n'

#             print(msg, end='')   
            
            

# def p0mocks(fig_path):

#     from glob import glob
#     import nbodykit.lab as nb

#     def p0mocks(maps, nside, ax=None):
#         path = '/home/mehdi/data/eboss/mocks/1.0/measurements/spectra/'
#         pkc = glob(f'{path}spectra_NGC_noweight_mainhighz_512_v7_1_*_main.json')
#         pk0 = glob(f'{path}spectra_NGC_knownsystot_mainhighz_512_v7_0_*_main.json')
#         pk1 = glob(f'{path}spectra_NGC_knownsystot_mainhighz_512_v7_1_*_main.json')
#         pk0n = glob(f'{path}spectra_NGC_{maps}_mainhighz_{nside}_v7_0_*_main.json')
#         pk1n = glob(f'{path}spectra_NGC_{maps}_mainhighz_{nside}_v7_1_*_main.json')


#         if ax is None:
#             fig, ax = plt.subplots()

#         c = ['green', 'grey', 'blue', 'orange', 'red']
        
#         ls = 2*[':', '-', '--', '-.', '-']
#         i = 0
#         for ni, pk_list in zip(['Cont', 'Null', 'Cont+Standard', f'Null+NN', f'Cont+NN'],
#                            [pkc, pk0, pk1, pk0n, pk1n]):
#             pk_k = []
#             print(ni, len(pk_list))
#             for pk_fn in pk_list:
#                 pk_ = nb.ConvolvedFFTPower.load(pk_fn)

#                 ki = pk_.poles['k']
#                 pk_i= pk_.poles['power_0'].real-pk_.attrs['shotnoise']+1.0e5
#                 pk_k.append(pk_i)
 
#             ax.plot(ki, np.mean(pk_k, axis=0), color=c[i], alpha=1, label=ni, ls=ls[i])
#             ax.fill_between(ki, *np.percentile(pk_k, [0, 100], axis=0), color=c[i], alpha=0.1)
#             i += 1

#         #ax.grid(ls=':', color='grey', alpha=0.4)
#         #ax.set(xscale='log', yscale='log', xlabel='k', ylabel='P0')
#         #ax.set_ylim(1.0e2, 2.0e6)
#         #ax.legend()
#         return ax

#     fig, ax = plt.subplots(figsize=(6, 4))

#     p0mocks('known', '512', ax=ax)
#     ax.set(xscale='log', yscale='log')
#     ax.grid(True, ls=':', alpha=0.5)
#     ax.legend()
#     ax.tick_params(direction='in', which='both', axis='both', right=True, top=True)
#     ax.set_ylabel(r'P$_{0}~[Mpc/h]^{3}$ + Const.')
#     ax.set_xlabel(r'k [h/Mpc]')
#     fig.savefig(fig_path, bbox_inches='tight')    
    
    
# def sigP0mocks(fig_path):
    
#     import nbodykit.lab as nb
#     from glob import glob

#     def read_pk(addrs):
#         files = glob(addrs)
#         print(len(files))
#         pk_ = []
#         for file_ in files:
#             d = nb.ConvolvedFFTPower.load(file_)

#             k = d.poles['k']
#             pk_.append(d.poles['power_0'].real)#-d.attrs['shotnoise'])

#         return (k, np.array(pk_))

#     maps = 'known'
#     nside = 512
#     path = '/home/mehdi/data/eboss/mocks/1.0/measurements/spectra/'
#     pks = {}
#     pks['Cont'] = read_pk(f'{path}spectra_NGC_noweight_mainhighz_512_v7_1_*_main.json')
#     pks['Null'] = read_pk(f'{path}spectra_NGC_knownsystot_mainhighz_512_v7_0_*_main.json')
#     pks['Cont+Standard'] = read_pk(f'{path}spectra_NGC_knownsystot_mainhighz_512_v7_1_*_main.json')
#     pks['Null+NN'] = read_pk(f'{path}spectra_NGC_{maps}_mainhighz_{nside}_v7_0_*_main.json')
#     pks['Cont+NN'] = read_pk(f'{path}spectra_NGC_{maps}_mainhighz_{nside}_v7_1_*_main.json')    

#     print(pks.keys())


#     fig, ax = plt.subplots(figsize=(6, 4))

#     c = ['green', 'grey', 'blue', 'orange', 'red']

#     ls = 2*[':', '-', '--', '-.', '-']


#     for i, (ni, pki) in enumerate(pks.items()):

#         ax.plot(pki[0], np.std(pki[1], axis=0, ddof=1), label=ni, ls=ls[i], color=c[i])


#     ax.grid(which='both', ls=':', alpha=0.5)    
#     ax.legend(fontsize=10)    
#     ax.set_xscale('log')

#     ax.set_ylabel(r'$\sigma~P_{0}$')
#     ax.set_xlabel(r'k [h/Mpc]')
#     ax.set_yscale('log')
#     ax.tick_params(direction='in', which='both', axis='both', right=True, top=True)
#     fig.savefig(fig_path, bbox_inches='tight')
        
        
    
# def mollweide3maps(fig_path):
    
#     import healpy as hp
#     from lssutils.utils import EbossCat
#     import pandas as pd
#     import lssutils.dataviz as dv
    
#     ''' READ THE FULL CATALOGS
#     '''
#     nran_bar = 65.6 # 5000 per sq deg
#     path_cats = '/home/mehdi/data/eboss/data/v7_2/'
#     dNGC = EbossCat(f'{path_cats}eBOSS_QSO_full_NGC_v7_2.dat.fits', zmin=0.8, zmax=3.5)
#     rNGC = EbossCat(f'{path_cats}eBOSS_QSO_full_NGC_v7_2.ran.fits', kind='randoms', zmin=0.8, zmax=3.5)

#     dSGC = EbossCat(f'{path_cats}eBOSS_QSO_full_SGC_v7_2.dat.fits', zmin=0.8, zmax=3.5)
#     rSGC = EbossCat(f'{path_cats}eBOSS_QSO_full_SGC_v7_2.ran.fits', kind='randoms', zmin=0.8, zmax=3.5)



#     ''' BEFORE CORRECTION
#     '''
#     # NGC
#     ngal_ngc = dNGC.to_hp(512, 0.8, 3.5, raw=1)
#     nran_ngc = rNGC.to_hp(512, 0.8, 3.5, raw=1)

#     # SGC
#     ngal_sgc = dSGC.to_hp(512, 0.8, 3.5, raw=1)
#     nran_sgc = rSGC.to_hp(512, 0.8, 3.5, raw=1)

#     frac_ngc = nran_ngc / nran_bar
#     frac_sgc = nran_sgc / nran_bar

#     ngal_tot = ngal_ngc + ngal_sgc
#     frac_tot = frac_ngc + frac_sgc



#     ''' AFTER CORRECTION
#     '''
#     # NGC
#     ngal_ngc = dNGC.to_hp(512, 0.8, 3.5, raw=2)
#     nran_ngc = rNGC.to_hp(512, 0.8, 3.5, raw=2)

#     # SGC
#     ngal_sgc = dSGC.to_hp(512, 0.8, 3.5, raw=2)
#     nran_sgc = rSGC.to_hp(512, 0.8, 3.5, raw=2)

#     frac_ngc = nran_ngc / nran_bar
#     frac_sgc = nran_sgc / nran_bar

#     ngal_tot_f = ngal_ngc + ngal_sgc
#     frac_tot_f = frac_ngc + frac_sgc

#     ''' COMPUTE DENSITY [PER SQ. DEG.]
#     '''
#     area_1pix = hp.nside2pixarea(512, degrees=True)
#     ngal_dens = ngal_tot / (frac_tot * area_1pix)
#     ngal_dens_f = ngal_tot_f / (frac_tot_f * area_1pix)

#     # print(frac_tot_f.mean(), hp.nside2resol(512, arcmin=True))
#     # (0.1251878738403319, 6.870972823634812)

#     ''' EBV
#     '''
#     df = pd.read_hdf('~/data/templates/SDSS_WISE_HI_imageprop_nside512.h5', 'templates')
#     ebv = hp.ud_grade(df.ebv, nside_out=512)
#     ebv[~(frac_tot>0)] = np.nan # mask out 

#     fig = plt.figure(figsize=(7, 11)) # matplotlib is doing the mollveide projection
#     ax  = fig.add_subplot(311, projection='mollweide')
#     ax1 = fig.add_subplot(312, projection='mollweide')
#     ax2 = fig.add_subplot(313, projection='mollweide')

#     spacing = 0.01
#     plt.subplots_adjust(bottom=spacing, top=1-spacing, 
#                         left=spacing, right=1-spacing,
#                         hspace=0.0)


#     kw = dict(unit=r'N$_{{\rm QSO}}$ [deg$^{-2}$]', cmap=plt.cm.YlOrRd_r, 
#              vmin=40, vmax=100, #width=6, 
#              extend='both', galaxy=True)

#     dv.mollview(ngal_dens, figax=[fig, ax], **kw)
#     dv.mollview(ngal_dens_f, figax=[fig, ax2], colorbar=True, **kw)
#     dv.mollview(ebv, figax=[fig, ax1], galaxy=True, vmin=0, vmax=0.1, cmap=plt.cm.Reds, unit='')


#     ax.text(0.2, 0.2, 'Before Systot Correction', transform=ax.transAxes)
#     ax1.text(0.2, 0.2, 'E(B-V)', transform=ax1.transAxes)
#     ax2.text(0.2, 0.2, 'After Systot Correction', transform=ax2.transAxes)
#     # ax2.grid(True, ls=':', color='grey', alpha=0.4)

#     fig.savefig(fig_path, bbox_inches='tight', dpi=300, rasterized=True)    
#     return fig, (ax, ax1, ax2)



# def radec_zbins(fig_path):
#     import healpy as hp
#     from lssutils.utils import EbossCat, hpix2radec, shiftra
    
#     nran_bar = 65.6 # exp # of randoms per pixel
    
#     path = '/home/mehdi/data/eboss/data/v7_2/'
#     maps = {}
#     maps['dngc'] = EbossCat(f'{path}eBOSS_QSO_full_NGC_v7_2.dat.fits', zmin=0.8, zmax=3.5)
#     maps['rngc'] = EbossCat(f'{path}eBOSS_QSO_full_NGC_v7_2.ran.fits', zmin=0.8, zmax=3.5, kind='randoms')
#     maps['dsgc'] = EbossCat(f'{path}eBOSS_QSO_full_SGC_v7_2.dat.fits', zmin=0.8, zmax=3.5)
#     maps['rsgc'] = EbossCat(f'{path}eBOSS_QSO_full_SGC_v7_2.ran.fits', zmin=0.8, zmax=3.5, kind='randoms')


#     zcuts = [[0.8, 1.5], [1.5, 2.2], [2.2, 3.5]]

#     ngals = []

#     ncols = 3
#     fig, ax = plt.subplots(ncols=ncols, nrows=2, figsize=(6*ncols, 8), 
#                            sharey='row', sharex='row')
#     fig.subplots_adjust(wspace=0.0)
#     fig.align_labels()
#     ax= ax.flatten()


#     nside = 64
#     vmin = 0.8
#     vmax = 1.2
#     pixarea = hp.nside2pixarea(nside, degrees=True)
#     kw  = dict(cmap=plt.cm.YlOrRd_r, marker='.', rasterized=True, vmin=vmin, vmax=vmax) # vmax=20., vmin=0., 

#     xtext = 0.6
#     for j, cap in enumerate(['ngc', 'sgc']):
#         for i, zcut in enumerate(zcuts):        
#             ix = j*3 + i
#             ngal = maps['d%s'%cap].to_hp(nside, zcut[0], zcut[1], raw=1)
#             nran = maps['r%s'%cap].to_hp(nside, zcut[0], zcut[1], raw=1)

#             mask = nran > 0        
#             frac = nran / nran_bar
#             ngalc = ngal / (frac*pixarea)

#             ngalc = ngalc / ngalc[mask].mean()

#             #np.random.shuffle(mask)
#             #mask = mask[:1000]

#             hpix = np.argwhere(mask).flatten()
#             ra, dec = hpix2radec(nside, hpix)

#             mapi = ax[ix].scatter(shiftra(ra), dec, 15, c=ngalc[hpix], **kw, )


#             ax[ix].text(xtext, 0.1, '{}<z<{}'.format(*zcut), 
#                         color='k', transform=ax[ix].transAxes, fontsize=18, alpha=0.8)            

#             ax[ix].tick_params(direction='in', axis='both', which='both', labelsize=15,
#                               right=True, top=True)

#             if ix==4:
#                 ax[ix].set_xlabel('RA [deg]', fontsize=18) # title='{0}<z<{1}'.format(*zlim)

#             if ix%3==0:
#                 ax[ix].set_ylabel('DEC [deg]', fontsize=18)

#     cax = plt.axes([0.91, 0.2, 0.01, 0.6])
#     cbar = fig.colorbar(mapi, cax=cax,
#                  shrink=0.7, ticks=[vmin, 1, vmax], extend='both')
#     cbar.set_label(label=r'$n/\bar{n}$', size=20)
#     cbar.ax.tick_params(labelsize=15)
#     fig.savefig(fig_path, bbox_inches='tight')    
#     return fig, ax


# def kmean_jackknife(fig_path):

#     from lssutils.utils import EbossCat, KMeansJackknifes
#     path = '/home/mehdi/data/eboss/data/v7_2/'

#     nran_bar = 65.6 # for nside=512
#     njack = 20
#     randoms = EbossCat(f'{path}eBOSS_QSO_full_NGC_v7_2.ran.fits', 
#                            kind='randoms', zmin=0.8, zmax=3.5)

#     nranw = randoms.to_hp(512, zmin=0.8, zmax=3.5, raw=2)
#     mask = nranw > 0 
#     frac = nranw/nran_bar


#     jk = KMeansJackknifes(mask, frac)
#     jk.build_masks(njack)
    
#     fig, ax = jk.visualize()
#     ax.tick_params(direction='in', axis='both', 
#                     which='both', right=True, top=True)
#     fig.savefig(fig_path, bbox_inches='tight', rasterized=True)    
    
#     return fig, ax


# def train_val_losses_256vs512(fig_path, nchains=1, npartitions=1, alpha=1,
#                              cap='NGC', maps='all', sample='main'):
    
#     fig, ax = plt.subplots()

#     j = 0
#     c = ['k', 'crimson']

#     for nside in ['256', '512']:

#         metrics = np.load(f'/home/mehdi/data/eboss/data/v7_2/1.0/{cap}/'\
#                           +nside+f'/{sample}/nn_pnnl_{maps}/metrics.npz', allow_pickle=True)

#         taining_loss = metrics['losses'].item()['train']
#         valid_loss = metrics['losses'].item()['valid']


#         for k in range(npartitions):

#             base_train_loss = metrics['stats'].item()[k]['base_train_loss']
#             base_val_loss = metrics['stats'].item()[k]['base_valid_loss']

#             for i in range(nchains):

#                 ax.plot(np.array(taining_loss[k][i])-base_train_loss, color=c[j], ls='-', lw=1, alpha=alpha)
#                 ax.plot(np.array(valid_loss[k][i])-base_val_loss, color=c[j], ls='--', lw=2, alpha=alpha)

#         ax.text(0.75, 0.25+j*0.4, f'NSIDE={nside}', color=c[j], transform=ax.transAxes)

#         j += 1


#     ax.set_ylim(-0.5, .1)
#     ax.axhline(0, ls=':', color='grey')
#     ax.set(xlabel='Epoch', ylabel=r'$\Delta$PNLL [NN-baseline]')
#     ax.legend(['Training', 'Validation'])
#     ax.tick_params(direction='in', which='both', axis='both', pad=6, right=True, top=True)
    
#     fig.savefig(fig_path, bbox_inches='tight')
#     return fig, ax



# def train_val_losses_allvsknown(fig_path, nchains=1, npartitions=1, alpha=1,
#                                sample='main', nside='256', cap='NGC'):
#     fig, ax = plt.subplots()


#     j = 0
#     c = ['k', 'crimson']

    
    
#     for maps in ['all', 'known']:

#         path = '/home/mehdi/data/eboss/data/v7_2/1.0/'
#         metrics = np.load(f'{path}{cap}/{nside}/{sample}/nn_pnnl_{maps}/metrics.npz', allow_pickle=True)

#         taining_loss = metrics['losses'].item()['train']
#         valid_loss = metrics['losses'].item()['valid']


#         for k in range(npartitions):

#             base_train_loss = metrics['stats'].item()[k]['base_train_loss']
#             base_val_loss = metrics['stats'].item()[k]['base_valid_loss']

#             for i in range(nchains):

#                 ax.plot(np.array(taining_loss[k][i])-base_train_loss, color=c[j], ls='-', lw=1, alpha=alpha)
#                 ax.plot(np.array(valid_loss[k][i])-base_val_loss, color=c[j], ls='--', lw=2, alpha=alpha)

#         ax.text(0.8, 0.7-j*0.08, f'{maps}', color=c[j], transform=ax.transAxes)

#         j += 1


#     ax.set_ylim(-0.5, .1)
#     ax.axhline(0, ls=':', color='grey')
#     ax.set(xlabel='Epoch', ylabel=r'$\Delta$PNLL [NN-baseline]')
#     ax.legend(['Training', 'Validation'], loc=1)
#     ax.tick_params(direction='in', which='both', axis='both', pad=6, right=True, top=True)
    
#     fig.savefig(fig_path, bbox_inches='tight')
#     return fig, ax

# def mollweide_templates(fig_path):
#     import pandas as pd
#     from lssutils.dataviz import mollview

#     templates = pd.read_hdf('/home/mehdi/data/templates/SDSS_WISE_HI_imageprop_nside512.h5', 'templates')


#     fig = plt.figure(figsize=(5, 3.5))
#     ax0  = fig.add_axes([0.0, 0, 1., 1.],       projection='mollweide')
#     ax1  = fig.add_axes([0.9, 0, 1., 1.], projection='mollweide')
#     ax2  = fig.add_axes([1.8, 0, 1., 1.], projection='mollweide')
#     ax3  = fig.add_axes([0.0,-0.5, 1., 1.], projection='mollweide')
#     ax4  = fig.add_axes([0.9,-0.5, 1., 1.], projection='mollweide')
#     ax5  = fig.add_axes([1.8,-0.5, 1., 1.], projection='mollweide')
#     fig.subplots_adjust(hspace=0, wspace=0)

#     ax = [ax0, ax1, ax2, ax3, ax4, ax5]

#     names = ['sky_i', 'psf_i', 'depth_g_minus_ebv', 'ebv', 'run', 'airmass']

#     for i,name in enumerate(names):

#         good = np.isfinite(templates[name])
#         vmin, vmax = np.percentile(templates[name][good], [5, 95])

#         mapi = templates[name].copy().values
#         mapi[~good] = np.inf

#         mollview(mapi, vmin=vmin, vmax=vmax,
#                 cmap=plt.cm.viridis, figax=[fig, ax[i]], unit='', galaxy=False)


#         ax[i].set(xticks=[], yticks=[])
#         ax[i].axis('off')
#         ax[i].text(0.4, 0.35, names[i], transform=ax[i].transAxes)    
        
#     fig.savefig(fig_path, bbox_inches='tight')
#     return fig, ax



# def stdwnn_epoch(fig_path, cap='NGC', ns='256', s='main'):
#     import fitsio as ft
    
#     root_path = f'/home/mehdi/data/eboss/data/v7_2/1.0/{cap}'
#     c = ['k', 'crimson']    
#     mk = ['w', 'crimson']
    
#     fig, ax = plt.subplots()
#     ax.tick_params(direction='in', which='both', top=True, right=True)   
    
#     for j, mp in enumerate(['all', 'known']):
        
#         wnn = ft.read(f'{root_path}/{ns}/{s}/nn_pnnl_{mp}/nn-weights.fits') # (Npix, Nchains)
        
#         for i in range(1, wnn['weight'].shape[1]+1):
#             wnn_mean = wnn['weight'][:, :i].mean(axis=1)
#             ax.scatter(i, np.std(wnn_mean), color=c[j], marker='o', fc=mk[j])
            
#         ax.text(0.8, 0.7-j*0.08, f'{mp}', color=c[j], transform=ax.transAxes)

#     ax.set(xlabel='Nchains', ylabel=r'std(Nqso$_{\rm NN}$)', xticks=np.arange(1, 21))    
#     fig.savefig(fig_path, bbox_inches='tight')
    
#     return fig, ax


# def cell_ngc_main_nchain1(fig_path):
#     from lssutils.lab import histogram_cell
    
#     path = '/home/mehdi/data/eboss/data/v7_2/1.0/measurements/cl/'
#     d = np.load(f'{path}cl_ngc_main_known_512_nchains.npz', allow_pickle=True)
    
#     cl_avg = d['cl_avg'].item()
#     cl_def = d['cl_def'].item()
#     cl_old = d['cl_old'].item()
    
#     i = -1
#     fig, ax = plt.subplots()
#     num = len(cl_avg.keys())-2
#     print(num)
#     for idx in cl_avg:

#         if idx == -1:
#             continue
#         print(idx, end=' ')

#         cl_i = cl_avg[idx]

#         elb, clb = histogram_cell(cl_i['cl_gg']['cl'])

#         #plt.loglog(cl_i['cl_gg']['l'], 
#         #      cl_i['cl_gg']['cl'], 
#         #      color=plt.cm.jet(i/num),
#         #      alpha=0.2)

#         ls = '-' #'None' if idx != 19 else '-'
#         ax.plot(elb, clb, label='N=%d'%idx,
#                   color=plt.cm.jet(i/num), 
#                  ls=ls, marker='.')
#         i += 1

#     elbd, clbd = histogram_cell(cl_avg[-1]['cl_gg']['cl'])
#     ax.plot(elbd, clbd, ls='-', marker='.', color='k', label='No weight')

#     elbd, clbd = histogram_cell(cl_def['cl_gg']['cl'])
#     ax.plot(elbd, clbd, ls='--', marker='s', color='grey', mfc='w', label='Systot')

#     elbo, clbo = histogram_cell(cl_old['cl_gg']['cl'])
#     ax.plot(elbo, clbo, ls=':', marker='^', color='grey',label='NN (Rezaie+2020)')


#     ax.legend()
#     lines = ax.get_lines()
#     legend1 = plt.legend(lines[:-3], ["N=%d"%d for d in range(1,21)], bbox_to_anchor=(1., 1.02), fontsize=7.8)
#     legend2 = plt.legend(lines[-3:], ["No Weight", "Systot", "NN (Rezaie+2020)"], loc=1)
#     ax.add_artist(legend1)
#     ax.add_artist(legend2)


#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_ylim(1.0e-6, 3.0e-4)
#     ax.set_ylabel(r'C$_{\ell}$')
#     ax.set_xlabel(r'$\ell$')
#     ax.tick_params(direction='in', which='both', 
#                    axis='both', pad=6, right=True, top=True)
#     fig.savefig(fig_path, bbox_inches='tight')    
#     return fig, ax


# def cell_ngc_main_nchain2(fig_path):
#     from lssutils.lab import histogram_cell

#     path = '/home/mehdi/data/eboss/data/v7_2/1.0/measurements/cl/'
#     d = np.load(f'{path}cl_ngc_main_known_512_nchains.npz',
#                allow_pickle=True)
    
#     cl_ind = d['cl_indv'].item()
#     cl_avg = d['cl_avg'].item()
#     cl_def = d['cl_def'].item()

#     fig, ax  = plt.subplots()
    
#     keys = list(cl_ind.keys())
#     num = len(keys)
#     clbs = []

#     for idx in keys:
#         print(idx, end=' ')
#         cl_i = cl_ind[idx]
#         elb, clb = histogram_cell(cl_i['cl_gg']['cl'])
#         ls = '-' #'None' if idx != 19 else '-'
#         ax.plot(elb, clb,
#                   color='b', 
#                  ls=ls, marker='o', alpha=0.1, lw=1, zorder=-1)

#         clbs.append(clb)

#     ax.plot(elb, np.mean(clbs, axis=0), color='b', label='avg. of Cls', zorder=1)

#     elb, clb = histogram_cell(cl_avg[19]['cl_gg']['cl'])
#     ax.plot(elb, clb, color='crimson', ls='-', marker='.', label="avg. of Wsys' (N=20)", zorder=1)

#     elb, clb = histogram_cell(cl_avg[-1]['cl_gg']['cl'])
#     ax.plot(elb, clb, color='k', ls='-', marker='.', label="No Weight", zorder=-1)


#     elbd, clbd = histogram_cell(cl_def['cl_gg']['cl'])
#     ax.loglog(elbd, clbd, ls='--', marker='s', mfc='w', color='grey', label='Systot', zorder=-1)

#     ax.legend()
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_ylim(1.0e-6, 3.0e-4)
#     ax.set_ylabel(r'C$_{\ell}$')
#     ax.set_xlabel(r'$\ell$')
#     ax.tick_params(direction='in', which='both', 
#                    axis='both', pad=6, right=True, top=True)
#     fig.savefig(fig_path, bbox_inches='tight')    
    
#     return fig, ax


# def p0_512vs256_knownvsall_ngcvssgc(fig_path):
    
#     import nbodykit.lab as nb
#     def read_pk(filename):
#         pk = nb.ConvolvedFFTPower.load(filename)
#         return (pk.poles['k'], pk.poles['power_0'].real-pk.attrs['shotnoise'])

#     pks = {}

#     path_spectra = '/home/mehdi/data/eboss/data/v7_2/1.0/measurements/spectra/'

#     for cap in ['NGC', 'SGC']:
#         for zbin in ['main', 'highz']:

#             for maps in ['all', 'known']:
#                 for nside in ['256', '512']:

#                     name = '_'.join([cap, zbin, maps, nside])
#                     pks[name] = read_pk(f'{path_spectra}spectra_{cap}_{maps}_mainhighz_{nside}_v7_2_{zbin}.json')
#                     print('.', end='')

#             pks[f'{cap}_{zbin}_systot'] = read_pk(f'{path_spectra}spectra_{cap}_knownsystot_mainhighz_512_v7_2_{zbin}.json')



#     # creat figure
#     fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 8),
#                           sharex=True, sharey='row')
#     fig.subplots_adjust(hspace=0.02, wspace=0)
#     ax = ax.flatten()

#     # add cosmetics
#     for i, axi in enumerate(ax):
#         axi.tick_params(direction='in', which='both', axis='both', 
#                         pad=6, right=True, top=True)
#         axi.set(xscale='log')
#         axi.grid(True, ls=':', color='grey', alpha=0.2)

#         if i in [0, 2]:
#             axi.set_ylabel(r'P$_{0}$(k) [Mpc/h]$^{3}$')
#         if i in [2, 3]:
#             axi.set_xlabel(r'k [h/Mpc]')


#     colors = {'all':'#4b88e3',
#               'known':'#d98404',
#               'systot':'#000000'}

#     markers = {'all':'o',
#               'known':'s'}

#     styles = {'512':'-',
#               '256':'--'}

#     rows = {'NGC':0,
#             'SGC':1}

#     cols = {'main':0,
#            'highz':1}

#     zlim = {'main':'0.8<z<2.2',
#            'highz':'2.2<z<3.5'}

#     for cap in ['NGC', 'SGC']:
#         for zbin in ['main', 'highz']:    

#             ix, iy = rows[cap], cols[zbin]
#             ik = ix*2+iy

#             if ik < 2:
#                 ax[ik].text(0.3, 0.9, ' '.join([zbin.upper(), zlim[zbin]]), 
#                            transform=ax[ik].transAxes)

#             if ik in [0, 2]:
#                 ax[ik].text(0.1, 0.1, cap, transform=ax[ik].transAxes)

#             lb = 'SYSTOT' if ik==0 else None
#             ax[ik].plot(*pks[f'{cap}_{zbin}_systot'], ls='-', 
#                         color=colors['systot'], marker='.', label=lb)

#             for maps in ['all', 'known']:
#                 for nside in ['256', '512']:

#                     name = '_'.join([cap, zbin, maps, nside])

#                     ls = styles[nside]
#                     color = colors[maps]
#                     mk = markers[maps]

#                     if (ik==1) and (maps=='all'):
#                         lb = 'NSIDE=%s'%nside  
#                     elif (ik==0) and (nside=='512'):
#                         lb = '%s'%(maps.upper())
#                     else:
#                         lb = None

#                     ax[ik].plot(*pks[name], ls=ls, color=color, marker=mk, mfc='w', label=lb)

#     for ik in [0, 1]:
#         ax[ik].legend(frameon=False, numpoints=2)
    
#     fig.savefig(fig_path, bbox_inches='tight')    
#     return fig, ax

# def p2_512vs256_knownvsall_ngcvssgc(fig_path):
    
#     import nbodykit.lab as nb
#     def read_pk(filename):
#         pk = nb.ConvolvedFFTPower.load(filename)
#         return (pk.poles['k'], pk.poles['power_2'].real)

#     pks = {}

#     path_spectra = '/home/mehdi/data/eboss/data/v7_2/1.0/measurements/spectra/'

#     for cap in ['NGC', 'SGC']:
#         for zbin in ['main', 'highz']:

#             for maps in ['all', 'known']:
#                 for nside in ['256', '512']:

#                     name = '_'.join([cap, zbin, maps, nside])
#                     pks[name] = read_pk(f'{path_spectra}spectra_{cap}_{maps}_mainhighz_{nside}_v7_2_{zbin}.json')
#                     print('.', end='')

#             pks[f'{cap}_{zbin}_systot'] = read_pk(f'{path_spectra}spectra_{cap}_knownsystot_mainhighz_512_v7_2_{zbin}.json')



#     # creat figure
#     fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 8),
#                           sharex=True, sharey='row')
#     fig.subplots_adjust(hspace=0.02, wspace=0)
#     ax = ax.flatten()

#     # add cosmetics
#     for i, axi in enumerate(ax):
#         axi.tick_params(direction='in', which='both', axis='both', 
#                         pad=6, right=True, top=True)
#         axi.set(xscale='log')
#         axi.grid(True, ls=':', color='grey', alpha=0.2)

#         if i in [0, 2]:
#             axi.set_ylabel(r'P$_{2}$(k) [Mpc/h]$^{3}$')
#         if i in [2, 3]:
#             axi.set_xlabel(r'k [h/Mpc]')


#     colors = {'all':'#4b88e3',
#               'known':'#d98404',
#               'systot':'#000000'}

#     markers = {'all':'o',
#               'known':'s'}

#     styles = {'512':'-',
#               '256':'--'}

#     rows = {'NGC':0,
#             'SGC':1}

#     cols = {'main':0,
#            'highz':1}

#     zlim = {'main':'0.8<z<2.2',
#            'highz':'2.2<z<3.5'}

#     for cap in ['NGC', 'SGC']:
#         for zbin in ['main', 'highz']:    

#             ix, iy = rows[cap], cols[zbin]
#             ik = ix*2+iy

#             if ik < 2:
#                 ax[ik].text(0.3, 0.9, ' '.join([zbin.upper(), zlim[zbin]]), 
#                            transform=ax[ik].transAxes)

#             if ik in [0, 2]:
#                 ax[ik].text(0.1, 0.1, cap, transform=ax[ik].transAxes)

#             lb = 'SYSTOT' if ik==0 else None
#             ax[ik].plot(*pks[f'{cap}_{zbin}_systot'], ls='-', 
#                         color=colors['systot'], marker='.', label=lb)

#             for maps in ['all', 'known']:
#                 for nside in ['256', '512']:

#                     name = '_'.join([cap, zbin, maps, nside])

#                     ls = styles[nside]
#                     color = colors[maps]
#                     mk = markers[maps]

#                     if (ik==1) and (maps=='all'):
#                         lb = 'NSIDE=%s'%nside  
#                     elif (ik==0) and (nside=='512'):
#                         lb = '%s'%(maps.upper())
#                     else:
#                         lb = None

#                     ax[ik].plot(*pks[name], ls=ls, color=color, marker=mk, mfc='w', label=lb)

#     for ik in [0, 1]:
#         ax[ik].legend(frameon=False, numpoints=2)
    
#     fig.savefig(fig_path, bbox_inches='tight')    
#     return fig, ax

 

# def __read_nnbar(nnbar_filename):
#     d = np.load(nnbar_filename, allow_pickle=True)
#     dd = {}
#     for i in range(len(d)):
#         dname = d[i]['sys']
#         dd[dname] = (d[i]['bin_avg'], d[i]['nnbar'], d[i]['nnbar_err'])
#     return dd    


# def chi2_from_nbar(fig_path):

#     nnbars = {}

#     path_nnbar = '/home/mehdi/data/eboss/data/v7_2/1.0/measurements/nnbar/'

#     for cap in ['NGC', 'SGC']:
#         for zbin in ['main', 'highz']:

#             for maps in ['all', 'known']:
#                 for nside in ['512']: # '256', 

#                     name = '_'.join([cap, zbin, maps, nside])
#                     #nnbars[name] = __read_nnbar(f'{path_nnbar}nnbar_{cap}_{maps}_mainhighz_{nside}_v7_2_{zbin}.npy')
#                     nnbars[name] = __read_nnbar(f'{path_nnbar}nnbar_{cap}_{maps}_lowmidhighz_{nside}_v7_2_{zbin}.npy')
#                     print('.', end='')

#             nnbars[f'{cap}_{zbin}_systot'] = __read_nnbar(f'{path_nnbar}nnbar_{cap}_knownsystot_mainhighz_512_v7_2_{zbin}.npy')
#             nnbars[f'{cap}_{zbin}_noweight'] = __read_nnbar(f'{path_nnbar}nnbar_{cap}_noweight_mainhighz_512_v7_2_{zbin}.npy')


#     chi2_fn = lambda y, ye:(((y-1.0)/ye)**2).sum()
#     sysm_n_all = nnbars['NGC_main_all_512'].keys()


#     nnbars_chi2 = {}
#     for key in nnbars:
#         chi_l = []

#         for sysm_n in sysm_n_all:

#             chi_l.append(chi2_fn(nnbars[key][sysm_n][1], nnbars[key][sysm_n][2]))

#         nnbars_chi2[key] = chi_l
#         print('.', end='')


#     assert len(nnbars['NGC_highz_all_512']['run'][0])==8


#     fig, ax = plt.subplots(nrows=2, ncols=2, 
#                            figsize=(16, 6), 
#                            sharex=True, sharey='row')
#     fig.subplots_adjust(hspace=0.0, wspace=0.0)
#     ax = ax.flatten()

#     x_ = np.arange(len(sysm_n_all))
#     dx = 0.18
#     width=0.18
#     shift = -0.3

#     titles = {'main':'0.8<z<2.2',
#              'highz':'2.2<z<3.5'}

#     lgnarg = dict(ncol=3, frameon=False,
#                  bbox_to_anchor=(0.2, 1.01, 1.5, 0.4), loc="lower left",
#                 mode="expand", borderaxespad=0)


#     for k, cap in enumerate(['NGC', 'SGC']):

#         for j, zbin in enumerate(['main', 'highz']):
#             ix = 2*k + j

#             for i, nside in enumerate(['512']):#, '256']):
#                 # 
#                 ax[ix].bar(x_+(2*i+1)*dx+shift, nnbars_chi2[f'{cap}_{zbin}_known_{nside}'], width=width, label=f'{nside}-known')
#                 ax[ix].bar(x_+(2*i+2)*dx+shift, nnbars_chi2[f'{cap}_{zbin}_all_{nside}'], width=width, label=f'{nside}-all')
#                 print(cap, zbin, nside, 'known', sum(nnbars_chi2[f'{cap}_{zbin}_known_{nside}']))
#                 print(cap, zbin, nside, 'all', sum(nnbars_chi2[f'{cap}_{zbin}_all_{nside}']))

#             ax[ix].bar(x_+shift, nnbars_chi2[f'{cap}_{zbin}_systot'], width=width, label='SYSTOT', color='grey')
#             #ax[ix].bar(x_+shift-dx, nnbars_chi2[f'{cap}_{zbin}_noweight'], width=0.2)

#             #ax[ix].text(0.1, .9, 'Nbins=8', color='grey', transform=ax[ix].transAxes)        
#             ax[ix].text(0.7, 0.9, f'{cap} {titles[zbin]}', transform=ax[ix].transAxes)
#             #ax[ix].set_ylim(0, 50)    
#             ax[ix].axhline(8, alpha=0.2, color='grey')    
#             ax[ix].tick_params(direction='in', axis='both', right=True, top=True)
#             print()
#         print(5*'-')

#     ax[1].text(15., 9, 'Nbins=8', color='grey')    
#     ax[0].legend(**lgnarg) #(title=f'NGC {titles[zbin]}', frameon=False)

#     for ix in [2, 3]:
#         ax[ix].set_xticks(x_)
#         ax[ix].set_xticklabels(sysm_n_all, rotation=90)

#     for ix in [0, 2]:
#         ax[ix].set_ylabel(r'$\chi^{2}(s)$')

#     fig.savefig(fig_path, bbox_inches='tight')



#     for k, cap in enumerate(['NGC', 'SGC']):
#         print(cap)
#         for j, zbin in enumerate(['main', 'highz']):        
#             line1 = ''
#             line2 = ''
#             line3 = ''
#             values = ''            
#             #print(zbin, end=' & ')
#             line1 += f'{zbin} '

#             # add default
#             chi2t = sum(nnbars_chi2[f'{cap}_{zbin}_noweight'])
#             values += f' {chi2t:.1f} &'
#             line3 += 'noweight &'

#             chi2t = sum(nnbars_chi2[f'{cap}_{zbin}_systot'])
#             values += f' {chi2t:.1f} &'
#             line3 += 'systot &'        

#             for i, nside in enumerate(['512']):#, '256']):
#                 #print(nside, end=' ')
#                 line2 += f'{nside} '
#                 for l, maps in enumerate(['known', 'all']):
#                     #print(maps, end=' ')
#                     line3 += f' {maps} &'

#                     chi2t = sum(nnbars_chi2[f'{cap}_{zbin}_{maps}_{nside}'])
#                     values += f' {chi2t:.1f} &'
                
#                 if nside == '512':
#                     line3 += ' &'
#                     values += ' &'
            

#             print('\n-----\n')            
#             print(line1)
#             print(line2)
#             print(line3)
#             print(values)    
        
#     return fig, ax