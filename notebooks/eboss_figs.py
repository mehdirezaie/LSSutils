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
    from lssutils.utils import EbossCat
    import pandas as pd
    import lssutils.dataviz as dv
    
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
    from lssutils.utils import EbossCat, hpix2radec, shiftra
    
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

    from lssutils.utils import EbossCat, KMeansJackknifes
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


def train_val_losses_256vs512(fig_path, nchains=1, npartitions=1, alpha=1,
                             cap='NGC', maps='all', sample='main'):
    
    fig, ax = plt.subplots()

    j = 0
    c = ['k', 'crimson']

    for nside in ['256', '512']:

        metrics = np.load(f'/home/mehdi/data/eboss/data/v7_2/1.0/{cap}/'\
                          +nside+f'/{sample}/nn_pnnl_{maps}/metrics.npz', allow_pickle=True)

        taining_loss = metrics['losses'].item()['train']
        valid_loss = metrics['losses'].item()['valid']


        for k in range(npartitions):

            base_train_loss = metrics['stats'].item()[k]['base_train_loss']
            base_val_loss = metrics['stats'].item()[k]['base_valid_loss']

            for i in range(nchains):

                ax.plot(np.array(taining_loss[k][i])-base_train_loss, color=c[j], ls='-', lw=1, alpha=alpha)
                ax.plot(np.array(valid_loss[k][i])-base_val_loss, color=c[j], ls='--', lw=2, alpha=alpha)

        ax.text(0.75, 0.25+j*0.4, f'NSIDE={nside}', color=c[j], transform=ax.transAxes)

        j += 1


    ax.set_ylim(-0.5, .1)
    ax.axhline(0, ls=':', color='grey')
    ax.set(xlabel='Epoch', ylabel=r'$\Delta$PNLL [NN-baseline]')
    ax.legend(['Training', 'Validation'])
    ax.tick_params(direction='in', which='both', axis='both', pad=6, right=True, top=True)
    
    fig.savefig(fig_path, bbox_inches='tight')
    return fig, ax



def train_val_losses_allvsknown(fig_path, nchains=1, npartitions=1, alpha=1,
                               sample='main', nside='256', cap='NGC'):
    fig, ax = plt.subplots()


    j = 0
    c = ['k', 'crimson']

    
    
    for maps in ['all', 'known']:

        path = '/home/mehdi/data/eboss/data/v7_2/1.0/'
        metrics = np.load(f'{path}{cap}/{nside}/{sample}/nn_pnnl_{maps}/metrics.npz', allow_pickle=True)

        taining_loss = metrics['losses'].item()['train']
        valid_loss = metrics['losses'].item()['valid']


        for k in range(npartitions):

            base_train_loss = metrics['stats'].item()[k]['base_train_loss']
            base_val_loss = metrics['stats'].item()[k]['base_valid_loss']

            for i in range(nchains):

                ax.plot(np.array(taining_loss[k][i])-base_train_loss, color=c[j], ls='-', lw=1, alpha=alpha)
                ax.plot(np.array(valid_loss[k][i])-base_val_loss, color=c[j], ls='--', lw=2, alpha=alpha)

        ax.text(0.8, 0.7-j*0.08, f'{maps}', color=c[j], transform=ax.transAxes)

        j += 1


    ax.set_ylim(-0.5, .1)
    ax.axhline(0, ls=':', color='grey')
    ax.set(xlabel='Epoch', ylabel=r'$\Delta$PNLL [NN-baseline]')
    ax.legend(['Training', 'Validation'], loc=1)
    ax.tick_params(direction='in', which='both', axis='both', pad=6, right=True, top=True)
    
    fig.savefig(fig_path, bbox_inches='tight')
    return fig, ax

def mollweide_templates(fig_path):
    import pandas as pd
    from lssutils.dataviz import mollview

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
        vmin, vmax = np.percentile(templates[name][good], [5, 95])

        mapi = templates[name].copy().values
        mapi[~good] = np.inf

        mollview(mapi, vmin=vmin, vmax=vmax,
                cmap=plt.cm.viridis, figax=[fig, ax[i]], unit='', galaxy=False)


        ax[i].set(xticks=[], yticks=[])
        ax[i].axis('off')
        ax[i].text(0.4, 0.35, names[i], transform=ax[i].transAxes)    
        
    fig.savefig(fig_path, bbox_inches='tight')
    return fig, ax



def stdwnn_epoch(fig_path, cap='NGC', ns='256', s='main'):
    import fitsio as ft
    
    root_path = f'/home/mehdi/data/eboss/data/v7_2/1.0/{cap}'
    c = ['k', 'crimson']    
    mk = ['w', 'crimson']
    
    fig, ax = plt.subplots()
    ax.tick_params(direction='in', which='both', top=True, right=True)   
    
    for j, mp in enumerate(['all', 'known']):
        
        wnn = ft.read(f'{root_path}/{ns}/{s}/nn_pnnl_{mp}/nn-weights.fits') # (Npix, Nchains)
        
        for i in range(1, wnn['weight'].shape[1]+1):
            wnn_mean = wnn['weight'][:, :i].mean(axis=1)
            ax.scatter(i, np.std(wnn_mean), color=c[j], marker='o', fc=mk[j])
            
        ax.text(0.8, 0.7-j*0.08, f'{mp}', color=c[j], transform=ax.transAxes)

    ax.set(xlabel='Nchains', ylabel=r'std(Nqso$_{\rm NN}$)', xticks=np.arange(1, 21))    
    fig.savefig(fig_path, bbox_inches='tight')
    
    return fig, ax

def pcc_wnn_nchains(fig_path, cap='NGC', ns='256', s='main'):

    from scipy.stats import pearsonr
    import fitsio as ft

    root_path = f'/home/mehdi/data/eboss/data/v7_2/1.0/{cap}'

    c = ['k', 'crimson']    
    mk = ['w', 'crimson']

    fig, ax = plt.subplots()
    ax.tick_params(direction='in', which='both', top=True, right=True)   

    for j, mp in enumerate(['all', 'known']):

        d = ft.read(f'{root_path}/{ns}/{s}/nn_pnnl_{mp}/nn-weights.fits') # (Npix, Nchains)

        for i in range(1, 11):

            set_1 = d['weight'][:, 0:i].mean(axis=1)
            set_2 = d['weight'][:, 10:i+10].mean(axis=1)

            pcc = pearsonr(set_1, set_2)[0]
            #print(f'i={i}, pcc={pcc:.3f}')

            plt.scatter(i, pcc, color=c[j], marker='o', fc=mk[j])

        ax.text(0.8, 0.5-j*0.08, f'{mp}', color=c[j], transform=ax.transAxes)

    ax.set(xlabel='Nchains', ylabel=r'PCC', xticks=np.arange(1, 11)) 
    fig.savefig(fig_path, bbox_inches='tight')

def cell_ngc_main_nchain1(fig_path):
    from lssutils.lab import histogram_cell
    
    path = '/home/mehdi/data/eboss/data/v7_2/1.0/measurements/cl/'
    d = np.load(f'{path}cl_ngc_main_known_512_nchains.npz', allow_pickle=True)
    
    cl_avg = d['cl_avg'].item()
    cl_def = d['cl_def'].item()
    cl_old = d['cl_old'].item()
    
    i = -1
    fig, ax = plt.subplots()
    num = len(cl_avg.keys())-2
    print(num)
    for idx in cl_avg:

        if idx == -1:
            continue
        print(idx, end=' ')

        cl_i = cl_avg[idx]

        elb, clb = histogram_cell(cl_i['cl_gg']['cl'])

        #plt.loglog(cl_i['cl_gg']['l'], 
        #      cl_i['cl_gg']['cl'], 
        #      color=plt.cm.jet(i/num),
        #      alpha=0.2)

        ls = '-' #'None' if idx != 19 else '-'
        ax.plot(elb, clb, label='N=%d'%idx,
                  color=plt.cm.jet(i/num), 
                 ls=ls, marker='.')
        i += 1

    elbd, clbd = histogram_cell(cl_avg[-1]['cl_gg']['cl'])
    ax.plot(elbd, clbd, ls='-', marker='.', color='k', label='No weight')

    elbd, clbd = histogram_cell(cl_def['cl_gg']['cl'])
    ax.plot(elbd, clbd, ls='--', marker='s', color='grey', mfc='w', label='Systot')

    elbo, clbo = histogram_cell(cl_old['cl_gg']['cl'])
    ax.plot(elbo, clbo, ls=':', marker='^', color='grey',label='NN (Rezaie+2020)')


    ax.legend()
    lines = ax.get_lines()
    legend1 = plt.legend(lines[:-3], ["N=%d"%d for d in range(1,21)], bbox_to_anchor=(1., 1.02), fontsize=7.8)
    legend2 = plt.legend(lines[-3:], ["No Weight", "Systot", "NN (Rezaie+2020)"], loc=1)
    ax.add_artist(legend1)
    ax.add_artist(legend2)


    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1.0e-6, 3.0e-4)
    ax.set_ylabel(r'C$_{\ell}$')
    ax.set_xlabel(r'$\ell$')
    ax.tick_params(direction='in', which='both', 
                   axis='both', pad=6, right=True, top=True)
    fig.savefig(fig_path, bbox_inches='tight')    
    return fig, ax


def cell_ngc_main_nchain2(fig_path):
    from lssutils.lab import histogram_cell

    path = '/home/mehdi/data/eboss/data/v7_2/1.0/measurements/cl/'
    d = np.load(f'{path}cl_ngc_main_known_512_nchains.npz',
               allow_pickle=True)
    
    cl_ind = d['cl_indv'].item()
    cl_avg = d['cl_avg'].item()
    cl_def = d['cl_def'].item()

    fig, ax  = plt.subplots()
    
    keys = list(cl_ind.keys())
    num = len(keys)
    clbs = []

    for idx in keys:
        print(idx, end=' ')
        cl_i = cl_ind[idx]
        elb, clb = histogram_cell(cl_i['cl_gg']['cl'])
        ls = '-' #'None' if idx != 19 else '-'
        ax.plot(elb, clb,
                  color='b', 
                 ls=ls, marker='o', alpha=0.1, lw=1, zorder=-1)

        clbs.append(clb)

    ax.plot(elb, np.mean(clbs, axis=0), color='b', label='avg. of Cls', zorder=1)

    elb, clb = histogram_cell(cl_avg[19]['cl_gg']['cl'])
    ax.plot(elb, clb, color='crimson', ls='-', marker='.', label="avg. of Wsys' (N=20)", zorder=1)

    elb, clb = histogram_cell(cl_avg[-1]['cl_gg']['cl'])
    ax.plot(elb, clb, color='k', ls='-', marker='.', label="No Weight", zorder=-1)


    elbd, clbd = histogram_cell(cl_def['cl_gg']['cl'])
    ax.loglog(elbd, clbd, ls='--', marker='s', mfc='w', color='grey', label='Systot', zorder=-1)

    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1.0e-6, 3.0e-4)
    ax.set_ylabel(r'C$_{\ell}$')
    ax.set_xlabel(r'$\ell$')
    ax.tick_params(direction='in', which='both', 
                   axis='both', pad=6, right=True, top=True)
    fig.savefig(fig_path, bbox_inches='tight')    
    
    return fig, ax


def p0_512vs256_knownvsall_ngcvssgc(fig_path):
    
    import nbodykit.lab as nb
    def read_pk(filename):
        pk = nb.ConvolvedFFTPower.load(filename)
        return (pk.poles['k'], pk.poles['power_0'].real-pk.attrs['shotnoise'])

    pks = {}

    path_spectra = '/home/mehdi/data/eboss/data/v7_2/1.0/measurements/spectra/'

    for cap in ['NGC', 'SGC']:
        for zbin in ['main', 'highz']:

            for maps in ['all', 'known']:
                for nside in ['256', '512']:

                    name = '_'.join([cap, zbin, maps, nside])
                    pks[name] = read_pk(f'{path_spectra}spectra_{cap}_{maps}_mainhighz_{nside}_v7_2_{zbin}.json')
                    print('.', end='')

            pks[f'{cap}_{zbin}_systot'] = read_pk(f'{path_spectra}spectra_{cap}_knownsystot_mainhighz_512_v7_2_{zbin}.json')



    # creat figure
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 8),
                          sharex=True, sharey='row')
    fig.subplots_adjust(hspace=0.02, wspace=0)
    ax = ax.flatten()

    # add cosmetics
    for i, axi in enumerate(ax):
        axi.tick_params(direction='in', which='both', axis='both', 
                        pad=6, right=True, top=True)
        axi.set(xscale='log')
        axi.grid(True, ls=':', color='grey', alpha=0.2)

        if i in [0, 2]:
            axi.set_ylabel(r'P$_{0}$(k) [Mpc/h]$^{3}$')
        if i in [2, 3]:
            axi.set_xlabel(r'k [h/Mpc]')


    colors = {'all':'#4b88e3',
              'known':'#d98404',
              'systot':'#000000'}

    markers = {'all':'o',
              'known':'s'}

    styles = {'512':'-',
              '256':'--'}

    rows = {'NGC':0,
            'SGC':1}

    cols = {'main':0,
           'highz':1}

    zlim = {'main':'0.8<z<2.2',
           'highz':'2.2<z<3.5'}

    for cap in ['NGC', 'SGC']:
        for zbin in ['main', 'highz']:    

            ix, iy = rows[cap], cols[zbin]
            ik = ix*2+iy

            if ik < 2:
                ax[ik].text(0.3, 0.9, ' '.join([zbin.upper(), zlim[zbin]]), 
                           transform=ax[ik].transAxes)

            if ik in [0, 2]:
                ax[ik].text(0.1, 0.1, cap, transform=ax[ik].transAxes)

            lb = 'SYSTOT' if ik==0 else None
            ax[ik].plot(*pks[f'{cap}_{zbin}_systot'], ls='-', 
                        color=colors['systot'], marker='.', label=lb)

            for maps in ['all', 'known']:
                for nside in ['256', '512']:

                    name = '_'.join([cap, zbin, maps, nside])

                    ls = styles[nside]
                    color = colors[maps]
                    mk = markers[maps]

                    if (ik==1) and (maps=='all'):
                        lb = 'NSIDE=%s'%nside  
                    elif (ik==0) and (nside=='512'):
                        lb = '%s'%(maps.upper())
                    else:
                        lb = None

                    ax[ik].plot(*pks[name], ls=ls, color=color, marker=mk, mfc='w', label=lb)

    for ik in [0, 1]:
        ax[ik].legend(frameon=False, numpoints=2)
    
    fig.savefig(fig_path, bbox_inches='tight')    
    return fig, ax

def p2_512vs256_knownvsall_ngcvssgc(fig_path):
    
    import nbodykit.lab as nb
    def read_pk(filename):
        pk = nb.ConvolvedFFTPower.load(filename)
        return (pk.poles['k'], pk.poles['power_2'].real)

    pks = {}

    path_spectra = '/home/mehdi/data/eboss/data/v7_2/1.0/measurements/spectra/'

    for cap in ['NGC', 'SGC']:
        for zbin in ['main', 'highz']:

            for maps in ['all', 'known']:
                for nside in ['256', '512']:

                    name = '_'.join([cap, zbin, maps, nside])
                    pks[name] = read_pk(f'{path_spectra}spectra_{cap}_{maps}_mainhighz_{nside}_v7_2_{zbin}.json')
                    print('.', end='')

            pks[f'{cap}_{zbin}_systot'] = read_pk(f'{path_spectra}spectra_{cap}_knownsystot_mainhighz_512_v7_2_{zbin}.json')



    # creat figure
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 8),
                          sharex=True, sharey='row')
    fig.subplots_adjust(hspace=0.02, wspace=0)
    ax = ax.flatten()

    # add cosmetics
    for i, axi in enumerate(ax):
        axi.tick_params(direction='in', which='both', axis='both', 
                        pad=6, right=True, top=True)
        axi.set(xscale='log')
        axi.grid(True, ls=':', color='grey', alpha=0.2)

        if i in [0, 2]:
            axi.set_ylabel(r'P$_{2}$(k) [Mpc/h]$^{3}$')
        if i in [2, 3]:
            axi.set_xlabel(r'k [h/Mpc]')


    colors = {'all':'#4b88e3',
              'known':'#d98404',
              'systot':'#000000'}

    markers = {'all':'o',
              'known':'s'}

    styles = {'512':'-',
              '256':'--'}

    rows = {'NGC':0,
            'SGC':1}

    cols = {'main':0,
           'highz':1}

    zlim = {'main':'0.8<z<2.2',
           'highz':'2.2<z<3.5'}

    for cap in ['NGC', 'SGC']:
        for zbin in ['main', 'highz']:    

            ix, iy = rows[cap], cols[zbin]
            ik = ix*2+iy

            if ik < 2:
                ax[ik].text(0.3, 0.9, ' '.join([zbin.upper(), zlim[zbin]]), 
                           transform=ax[ik].transAxes)

            if ik in [0, 2]:
                ax[ik].text(0.1, 0.1, cap, transform=ax[ik].transAxes)

            lb = 'SYSTOT' if ik==0 else None
            ax[ik].plot(*pks[f'{cap}_{zbin}_systot'], ls='-', 
                        color=colors['systot'], marker='.', label=lb)

            for maps in ['all', 'known']:
                for nside in ['256', '512']:

                    name = '_'.join([cap, zbin, maps, nside])

                    ls = styles[nside]
                    color = colors[maps]
                    mk = markers[maps]

                    if (ik==1) and (maps=='all'):
                        lb = 'NSIDE=%s'%nside  
                    elif (ik==0) and (nside=='512'):
                        lb = '%s'%(maps.upper())
                    else:
                        lb = None

                    ax[ik].plot(*pks[name], ls=ls, color=color, marker=mk, mfc='w', label=lb)

    for ik in [0, 1]:
        ax[ik].legend(frameon=False, numpoints=2)
    
    fig.savefig(fig_path, bbox_inches='tight')    
    return fig, ax


def p0_demo(fig_path):
    
    from matplotlib.gridspec import GridSpec
    import nbodykit.lab as nb

    def read_pk(filename):
        d = nb.ConvolvedFFTPower.load(filename)
        return (d.poles['k'], d.poles['power_0'].real - d.attrs['shotnoise'])


    def add_pk(x, y, ax1, ax2, **kw):
        ax1.plot(x, y, **kw)
        ax2.plot(x, y, **kw, zorder=-1, )


    path = '/home/mehdi/data/eboss/data/v7_2/1.0/measurements/spectra'
    pks = {}
    pks['systot'] = read_pk(f'{path}/spectra_NGC_knownsystot_mainhighz_512_v7_2_main.json')
    pks['noweight'] = read_pk(f'{path}/spectra_NGC_noweight_mainhighz_512_v7_2_main.json')
    pks['nn'] = read_pk(f'{path}/spectra_NGC_known_mainhighz_512_v7_2_main.json')    

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
    ylim = (1.0e3, 3.0e6)
    ax1.set(xlim=(xmin, xbreak), ylim=ylim, yscale='log', xscale='log')
    ax2.set(xlim=(xbreak, xmax), ylim=ylim, yscale='log')#, xscale='log')
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xticks([0.02, 0.05, 0.1, 0.15, 0.2, 0.25])

    ax1.axvline(xbreak, ls=':', color='grey', alpha=0.2)

    ax2.set_xlabel(r'k [h/Mpc]')
    ax1.set_ylabel(r'P$_{0}$(k) [Mpc/h]$^{3}$')
    ax2.xaxis.set_label_coords(0., -0.1) # https://stackoverflow.com/a/25018490/9746916

    #--- plots
    add_pk(*pks['noweight'], ax1, ax2, marker='.', color='#000000', label='Before treatment')
    add_pk(*pks['systot'], ax1, ax2, marker='o', mfc='w', ls='--', color='#4b88e3', label='Standard treatment')
    add_pk(*pks['nn'], ax1, ax2, marker='s', mfc='w', color='#d98404', label='Neural Network treatment')

    ax2.legend(frameon=False, title=r'eBOSS DR16 QSO NGC ($0.8<z<2.2$)')

    fig.savefig(fig_path, dpi=300, bbox_inches='tight')    

def __read_nnbar(nnbar_filename):
    d = np.load(nnbar_filename, allow_pickle=True)
    dd = {}
    for i in range(len(d)):
        dname = d[i]['sys']
        dd[dname] = (d[i]['bin_avg'], d[i]['nnbar'], d[i]['nnbar_err'])
    return dd    


def chi2_from_nbar(fig_path):

    nnbars = {}

    path_nnbar = '/home/mehdi/data/eboss/data/v7_2/1.0/measurements/nnbar/'

    for cap in ['NGC', 'SGC']:
        for zbin in ['main', 'highz']:

            for maps in ['all', 'known']:
                for nside in ['512']: # '256', 

                    name = '_'.join([cap, zbin, maps, nside])
                    #nnbars[name] = __read_nnbar(f'{path_nnbar}nnbar_{cap}_{maps}_mainhighz_{nside}_v7_2_{zbin}.npy')
                    nnbars[name] = __read_nnbar(f'{path_nnbar}nnbar_{cap}_{maps}_lowmidhighz_{nside}_v7_2_{zbin}.npy')
                    print('.', end='')

            nnbars[f'{cap}_{zbin}_systot'] = __read_nnbar(f'{path_nnbar}nnbar_{cap}_knownsystot_mainhighz_512_v7_2_{zbin}.npy')
            nnbars[f'{cap}_{zbin}_noweight'] = __read_nnbar(f'{path_nnbar}nnbar_{cap}_noweight_mainhighz_512_v7_2_{zbin}.npy')


    chi2_fn = lambda y, ye:(((y-1.0)/ye)**2).sum()
    sysm_n_all = nnbars['NGC_main_all_512'].keys()


    nnbars_chi2 = {}
    for key in nnbars:
        chi_l = []

        for sysm_n in sysm_n_all:

            chi_l.append(chi2_fn(nnbars[key][sysm_n][1], nnbars[key][sysm_n][2]))

        nnbars_chi2[key] = chi_l
        print('.', end='')


    assert len(nnbars['NGC_highz_all_512']['run'][0])==8


    fig, ax = plt.subplots(nrows=2, ncols=2, 
                           figsize=(16, 6), 
                           sharex=True, sharey='row')
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    ax = ax.flatten()

    x_ = np.arange(len(sysm_n_all))
    dx = 0.18
    width=0.18
    shift = -0.3

    titles = {'main':'0.8<z<2.2',
             'highz':'2.2<z<3.5'}

    lgnarg = dict(ncol=3, frameon=False,
                 bbox_to_anchor=(0.2, 1.01, 1.5, 0.4), loc="lower left",
                mode="expand", borderaxespad=0)


    for k, cap in enumerate(['NGC', 'SGC']):

        for j, zbin in enumerate(['main', 'highz']):
            ix = 2*k + j

            for i, nside in enumerate(['512']):#, '256']):
                # 
                ax[ix].bar(x_+(2*i+1)*dx+shift, nnbars_chi2[f'{cap}_{zbin}_known_{nside}'], width=width, label=f'{nside}-known')
                ax[ix].bar(x_+(2*i+2)*dx+shift, nnbars_chi2[f'{cap}_{zbin}_all_{nside}'], width=width, label=f'{nside}-all')
                print(cap, zbin, nside, 'known', sum(nnbars_chi2[f'{cap}_{zbin}_known_{nside}']))
                print(cap, zbin, nside, 'all', sum(nnbars_chi2[f'{cap}_{zbin}_all_{nside}']))

            ax[ix].bar(x_+shift, nnbars_chi2[f'{cap}_{zbin}_systot'], width=width, label='SYSTOT', color='grey')
            #ax[ix].bar(x_+shift-dx, nnbars_chi2[f'{cap}_{zbin}_noweight'], width=0.2)

            #ax[ix].text(0.1, .9, 'Nbins=8', color='grey', transform=ax[ix].transAxes)        
            ax[ix].text(0.7, 0.9, f'{cap} {titles[zbin]}', transform=ax[ix].transAxes)
            #ax[ix].set_ylim(0, 50)    
            ax[ix].axhline(8, alpha=0.2, color='grey')    
            ax[ix].tick_params(direction='in', axis='both', right=True, top=True)
            print()
        print(5*'-')

    ax[1].text(15., 9, 'Nbins=8', color='grey')    
    ax[0].legend(**lgnarg) #(title=f'NGC {titles[zbin]}', frameon=False)

    for ix in [2, 3]:
        ax[ix].set_xticks(x_)
        ax[ix].set_xticklabels(sysm_n_all, rotation=90)

    for ix in [0, 2]:
        ax[ix].set_ylabel(r'$\chi^{2}(s)$')

    fig.savefig(fig_path, bbox_inches='tight')



    for k, cap in enumerate(['NGC', 'SGC']):
        print(cap)
        for j, zbin in enumerate(['main', 'highz']):        
            line1 = ''
            line2 = ''
            line3 = ''
            values = ''            
            #print(zbin, end=' & ')
            line1 += f'{zbin} '

            # add default
            chi2t = sum(nnbars_chi2[f'{cap}_{zbin}_noweight'])
            values += f' {chi2t:.1f} &'
            line3 += 'noweight &'

            chi2t = sum(nnbars_chi2[f'{cap}_{zbin}_systot'])
            values += f' {chi2t:.1f} &'
            line3 += 'systot &'        

            for i, nside in enumerate(['512']):#, '256']):
                #print(nside, end=' ')
                line2 += f'{nside} '
                for l, maps in enumerate(['known', 'all']):
                    #print(maps, end=' ')
                    line3 += f' {maps} &'

                    chi2t = sum(nnbars_chi2[f'{cap}_{zbin}_{maps}_{nside}'])
                    values += f' {chi2t:.1f} &'
                
                if nside == '512':
                    line3 += ' &'
                    values += ' &'
            

            print('\n-----\n')            
            print(line1)
            print(line2)
            print(line3)
            print(values)    
        
    return fig, ax