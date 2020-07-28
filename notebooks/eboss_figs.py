import matplotlib.pyplot as plt
import numpy as np


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
    ax.step(nbar_ngc[:,0], nbar_ngc[:, 3]/1.0e-5, label='NGC', ls='--', color='royalblue', **kw)
    ax.step(nbar_sgc[:,0], nbar_sgc[:, 3]/1.0e-5, label='SGC', ls='-', color='darkorange', **kw)

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
    ax.set(xlabel='z', ylabel=r'10$^{5} \times$ n(z) [h/Mpc]$^{3}$', 
           ylim=(0, 2.5), xlim=(-0.2, 4))
    ax.legend()
    fig.savefig(fig_path, bbox_inches='tight')
    return fig, ax



def mollweide3maps(fig_path):
    
    import healpy as hp
    from LSSutils.utils import EbossCat
    import pandas as pd
    import LSSutils.dataviz as dv
    
    ''' READ THE FULL CATALOGS
    '''
    nran_bar = 65.6 # 5000 per sq deg
    path_cats = '/home/mehdi/data/eboss/data/v7_2/'
    dNGC = EbossCat(f'{path_cats}eBOSS_QSO_full_NGC_v7_2.dat.fits', zmin=0.8, zmax=3.5)
    rNGC = EbossCat(f'{path_cats}eBOSS_QSO_full_NGC_v7_2.ran.fits', kind='randoms', zmin=0.8, zmax=3.5)

    dSGC = EbossCat(f'{path_cats}eBOSS_QSO_full_SGC_v7_2.dat.fits', zmin=0.8, zmax=3.5)
    rSGC = EbossCat(f'{path_cats}eBOSS_QSO_full_SGC_v7_2.ran.fits', kind='randoms', zmin=0.8, zmax=3.5)



    ''' BEFORE CORRECTION
    '''
    # NGC
    ngal_ngc = dNGC.to_hp(512, 0.8, 3.5, raw=1)
    nran_ngc = rNGC.to_hp(512, 0.8, 3.5, raw=1)

    # SGC
    ngal_sgc = dSGC.to_hp(512, 0.8, 3.5, raw=1)
    nran_sgc = rSGC.to_hp(512, 0.8, 3.5, raw=1)

    frac_ngc = nran_ngc / nran_bar
    frac_sgc = nran_sgc / nran_bar

    ngal_tot = ngal_ngc + ngal_sgc
    frac_tot = frac_ngc + frac_sgc



    ''' AFTER CORRECTION
    '''
    # NGC
    ngal_ngc = dNGC.to_hp(512, 0.8, 3.5, raw=2)
    nran_ngc = rNGC.to_hp(512, 0.8, 3.5, raw=2)

    # SGC
    ngal_sgc = dSGC.to_hp(512, 0.8, 3.5, raw=2)
    nran_sgc = rSGC.to_hp(512, 0.8, 3.5, raw=2)

    frac_ngc = nran_ngc / nran_bar
    frac_sgc = nran_sgc / nran_bar

    ngal_tot_f = ngal_ngc + ngal_sgc
    frac_tot_f = frac_ngc + frac_sgc

    ''' COMPUTE DENSITY [PER SQ. DEG.]
    '''
    area_1pix = hp.nside2pixarea(512, degrees=True)
    ngal_dens = ngal_tot / (frac_tot * area_1pix)
    ngal_dens_f = ngal_tot_f / (frac_tot_f * area_1pix)

    # print(frac_tot_f.mean(), hp.nside2resol(512, arcmin=True))
    # (0.1251878738403319, 6.870972823634812)

    ''' EBV
    '''
    df = pd.read_hdf('~/data/templates/SDSS_WISE_HI_imageprop_nside512.h5', 'templates')
    ebv = hp.ud_grade(df.ebv, nside_out=512)
    ebv[~(frac_tot>0)] = np.nan # mask out 

    fig = plt.figure(figsize=(7, 11)) # matplotlib is doing the mollveide projection
    ax  = fig.add_subplot(311, projection='mollweide')
    ax1 = fig.add_subplot(312, projection='mollweide')
    ax2 = fig.add_subplot(313, projection='mollweide')

    spacing = 0.01
    plt.subplots_adjust(bottom=spacing, top=1-spacing, 
                        left=spacing, right=1-spacing,
                        hspace=0.0)


    kw = dict(unit=r'N$_{{\rm QSO}}$ [deg$^{-2}$]', cmap=plt.cm.YlOrRd_r, 
             vmin=40, vmax=100, #width=6, 
             extend='both', galaxy=True)

    dv.mollview(ngal_dens, figax=[fig, ax], **kw)
    dv.mollview(ngal_dens_f, figax=[fig, ax2], colorbar=True, **kw)
    dv.mollview(ebv, figax=[fig, ax1], galaxy=True, vmin=0, vmax=0.1, cmap=plt.cm.Reds, unit='')


    ax.text(0.2, 0.2, 'Before Systot Correction', transform=ax.transAxes)
    ax1.text(0.2, 0.2, 'E(B-V)', transform=ax1.transAxes)
    ax2.text(0.2, 0.2, 'After Systot Correction', transform=ax2.transAxes)
    # ax2.grid(True, ls=':', color='grey', alpha=0.4)

    fig.savefig(fig_path, bbox_inches='tight', dpi=300, rasterized=True)    
    return fig, (ax, ax1, ax2)



def radec_zbins(fig_path):
    import healpy as hp
    from LSSutils.utils import EbossCat, hpix2radec, shiftra
    
    nran_bar = 65.6 # exp # of randoms per pixel
    
    path = '/home/mehdi/data/eboss/data/v7_2/'
    maps = {}
    maps['dngc'] = EbossCat(f'{path}eBOSS_QSO_full_NGC_v7_2.dat.fits', zmin=0.8, zmax=3.5)
    maps['rngc'] = EbossCat(f'{path}eBOSS_QSO_full_NGC_v7_2.ran.fits', zmin=0.8, zmax=3.5, kind='randoms')
    maps['dsgc'] = EbossCat(f'{path}eBOSS_QSO_full_SGC_v7_2.dat.fits', zmin=0.8, zmax=3.5)
    maps['rsgc'] = EbossCat(f'{path}eBOSS_QSO_full_SGC_v7_2.ran.fits', zmin=0.8, zmax=3.5, kind='randoms')


    zcuts = [[0.8, 1.5], [1.5, 2.2], [2.2, 3.5]]

    ngals = []

    ncols = 3
    fig, ax = plt.subplots(ncols=ncols, nrows=2, figsize=(6*ncols, 8), 
                           sharey='row', sharex='row')
    fig.subplots_adjust(wspace=0.0)
    fig.align_labels()
    ax= ax.flatten()


    nside = 64
    vmin = 0.8
    vmax = 1.2
    pixarea = hp.nside2pixarea(nside, degrees=True)
    kw  = dict(cmap=plt.cm.YlOrRd_r, marker='.', rasterized=True, vmin=vmin, vmax=vmax) # vmax=20., vmin=0., 

    xtext = 0.6
    for j, cap in enumerate(['ngc', 'sgc']):
        for i, zcut in enumerate(zcuts):        
            ix = j*3 + i
            ngal = maps['d%s'%cap].to_hp(nside, zcut[0], zcut[1], raw=1)
            nran = maps['r%s'%cap].to_hp(nside, zcut[0], zcut[1], raw=1)

            mask = nran > 0        
            frac = nran / nran_bar
            ngalc = ngal / (frac*pixarea)

            ngalc = ngalc / ngalc[mask].mean()

            #np.random.shuffle(mask)
            #mask = mask[:1000]

            hpix = np.argwhere(mask).flatten()
            ra, dec = hpix2radec(nside, hpix)

            mapi = ax[ix].scatter(shiftra(ra), dec, 15, c=ngalc[hpix], **kw, )


            ax[ix].text(xtext, 0.1, '{}<z<{}'.format(*zcut), 
                        color='k', transform=ax[ix].transAxes, fontsize=18, alpha=0.8)            

            ax[ix].tick_params(direction='in', axis='both', which='both', labelsize=15,
                              right=True, top=True)

            if ix==4:
                ax[ix].set_xlabel('RA [deg]', fontsize=18) # title='{0}<z<{1}'.format(*zlim)

            if ix%3==0:
                ax[ix].set_ylabel('DEC [deg]', fontsize=18)

    cax = plt.axes([0.91, 0.2, 0.01, 0.6])
    cbar = fig.colorbar(mapi, cax=cax,
                 shrink=0.7, ticks=[vmin, 1, vmax], extend='both')
    cbar.set_label(label=r'$n/\bar{n}$', size=20)
    cbar.ax.tick_params(labelsize=15)
    fig.savefig(fig_path, bbox_inches='tight')    
    return fig, ax


def kmean_jackknife(fig_path):

    from LSSutils.utils import EbossCat, KMeansJackknifes
    path = '/home/mehdi/data/eboss/data/v7_2/'

    nran_bar = 65.6 # for nside=512
    njack = 20
    randoms = EbossCat(f'{path}eBOSS_QSO_full_NGC_v7_2.ran.fits', 
                           kind='randoms', zmin=0.8, zmax=3.5)

    nranw = randoms.to_hp(512, zmin=0.8, zmax=3.5, raw=2)
    mask = nranw > 0 
    frac = nranw/nran_bar


    jk = KMeansJackknifes(mask, frac)
    jk.build_masks(njack)
    
    fig, ax = jk.visualize()
    ax.tick_params(direction='in', axis='both', 
                    which='both', right=True, top=True)
    fig.savefig(fig_path, bbox_inches='tight', rasterized=True)    
    
    return fig, ax


def train_val_losses_256vs512(fig_path, nchains=1, npartitions=1, alpha=1):
    
    fig, ax = plt.subplots()

    j = 0
    c = ['k', 'crimson']

    for nside in ['256', '512']:

        metrics = np.load('/home/mehdi/data/eboss/data/v7_2/1.0/NGC/'\
                          +nside+'/main/nn_pnnl_all/metrics.npz', allow_pickle=True)

        taining_loss = metrics['losses'].item()['train']
        valid_loss = metrics['losses'].item()['valid']


        for k in range(npartitions):

            base_train_loss = metrics['stats'].item()[k]['base_train_loss']
            base_val_loss = metrics['stats'].item()[k]['base_val_loss']

            for i in range(nchains):

                ax.plot(np.array(taining_loss[k][i])-base_train_loss, color=c[j], ls='-', lw=1, alpha=alpha)
                ax.plot(np.array(valid_loss[k][i])-base_val_loss, color=c[j], ls='--', lw=2, alpha=alpha)

        ax.text(0.75, 0.15+j*0.28, f'NSIDE={nside}', color=c[j], transform=ax.transAxes)

        j += 1


    ax.set_ylim(-0.12, 0.12)
    ax.axhline(0, ls=':', color='grey')
    ax.set(xlabel='Epoch', ylabel=r'$\Delta$PNLL [NN-baseline]')
    ax.legend(['Training', 'Validation'])
    ax.tick_params(direction='in', which='both', axis='both', pad=6, right=True, top=True)
    
    fig.savefig(fig_path, bbox_inches='tight')
    return fig, ax



def train_val_losses_allvsknown(fig_path, nchains=1, npartitions=1, alpha=1):
    fig, ax = plt.subplots()


    j = 0
    c = ['k', 'crimson']

    nside='256'
    sample='main'
    for maps in ['all', 'known']:

        path = '/home/mehdi/data/eboss/data/v7_2/1.0/'
        metrics = np.load(f'{path}NGC/{nside}/{sample}/nn_pnnl_{maps}/metrics.npz', allow_pickle=True)

        taining_loss = metrics['losses'].item()['train']
        valid_loss = metrics['losses'].item()['valid']


        for k in range(npartitions):

            base_train_loss = metrics['stats'].item()[k]['base_train_loss']
            base_val_loss = metrics['stats'].item()[k]['base_val_loss']

            for i in range(nchains):

                ax.plot(np.array(taining_loss[k][i])-base_train_loss, color=c[j], ls='-', lw=1, alpha=alpha)
                ax.plot(np.array(valid_loss[k][i])-base_val_loss, color=c[j], ls='--', lw=2, alpha=alpha)

        ax.text(0.8, 0.7-j*0.08, f'{maps}', color=c[j], transform=ax.transAxes)

        j += 1


    ax.set_ylim(-0.12, 0.12)
    ax.axhline(0, ls=':', color='grey')
    ax.set(xlabel='Epoch', ylabel=r'$\Delta$PNLL [NN-baseline]')
    ax.legend(['Training', 'Validation'], loc=1)
    ax.tick_params(direction='in', which='both', axis='both', pad=6, right=True, top=True)
    
    fig.savefig(fig_path, bbox_inches='tight')
    return fig, ax

def mollweide_templates(fig_path):
    import pandas as pd
    from LSSutils.dataviz import mollview

    templates = pd.read_hdf('/home/mehdi/data/templates/SDSS_WISE_HI_imageprop_nside512.h5', 'templates')


    fig = plt.figure(figsize=(5, 3.5))
    ax0  = fig.add_axes([0.0, 0, 1., 1.],       projection='mollweide')
    ax1  = fig.add_axes([0.9, 0, 1., 1.], projection='mollweide')
    ax2  = fig.add_axes([1.8, 0, 1., 1.], projection='mollweide')
    ax3  = fig.add_axes([0.0,-0.5, 1., 1.], projection='mollweide')
    ax4  = fig.add_axes([0.9,-0.5, 1., 1.], projection='mollweide')
    ax5  = fig.add_axes([1.8,-0.5, 1., 1.], projection='mollweide')
    fig.subplots_adjust(hspace=0, wspace=0)

    ax = [ax0, ax1, ax2, ax3, ax4, ax5]

    names = ['sky_i', 'psf_i', 'depth_g_minus_ebv', 'ebv', 'run', 'airmass']

    for i,name in enumerate(names):

        good = np.isfinite(templates[name])
        vmin, vmax = np.percentile(templates[name][good], [10, 90])

        mapi = templates[name].copy().values
        mapi[~good] = np.inf

        mollview(mapi, vmin=vmin, vmax=vmax,
                cmap=plt.cm.viridis, figax=[fig, ax[i]], unit='', galaxy=False)

    for i, axi in enumerate(ax):
        axi.set(xticks=[], yticks=[])
        axi.axis('off')
        axi.text(0.4, 0.35, names[i], transform=axi.transAxes)    
        
    fig.savefig(fig_path, bbox_inches='tight')
    return fig, ax