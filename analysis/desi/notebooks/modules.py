import sys
import numpy as np
import fitsio as ft
import healpy as hp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import getdist

from matplotlib.gridspec import GridSpec
from getdist import plots, MCSamples
from glob import glob

sys.path.insert(0, '/users/PHS0336/medirz90/github/LSSutils')
import lssutils.utils as ut
import lssutils.dataviz as dv
from lssutils.io import read_nnbar, read_nbmocks, read_chain
from lssutils.stats.pcc import pcc
from lssutils.stats.window import WindowSHT
from lssutils.theory.cell import (init_sample, SurveySpectrum, Spectrum, bias_model_lrg)

dv.setup_color()



# --- helper functions
def print_stats(stats):
    for s, v in stats.items():
        msg = r"{0:40s}& ${1:6.2f}$& ${2:6.2f}$& ${3:6.2f}<\fnl<{4:6.2f}$& ${5:6.2f}<\fnl<{6:6.2f}$\\"\
                .format(s, v[0], v[1], v[2], v[3], v[4], v[5])
        print(msg)     


class MCMC(MCSamples):
     def __init__(self, path_to_mcmc, read_kw=dict(), mc_kw=dict()):
        self.stats, chains = read_chain(path_to_mcmc, **read_kw)
        MCSamples.__init__(self, samples=chains, **mc_kw)
        

def read_clx(fn, bins=None):

    cl = np.load(fn, allow_pickle=True).item()
    
    cl_cross = []
    cl_ss = []

    for i in range(len(cl['cl_sg'])):    
        __, cl_sg_ = ut.histogram_cell(cl['cl_sg'][i]['l'], cl['cl_sg'][i]['cl'], bins=bins)
        __, cl_ss_ = ut.histogram_cell(cl['cl_ss'][i]['l'], cl['cl_ss'][i]['cl'], bins=bins)

        cl_ss.append(cl_ss_)
        cl_cross.append(cl_sg_**2/cl_ss_)    

    return np.array(cl_cross).flatten()


def read_clxmocks(list_clx, bins=None):
    
    err_mat = []    
    for i, clx_i in enumerate(list_clx):
        
        err_i  = read_clx(clx_i, bins=bins)
        err_mat.append(err_i)
        
        if (i % (len(list_clx)//10)) == 0:
            print(f'{i}/{len(list_clx)}')

    err_mat = np.array(err_mat)
    print(err_mat.shape)
    return err_mat

def combine_nn(maps, output):
    
    hpmap = np.zeros(12*256*256)
    counts = np.zeros(12*256*256)
    for map_ in maps:
        d_ = ft.read(map_)
        counts[d_['hpix']] += 1.0
        hpmap[d_['hpix']] += d_['weight'].mean(axis=1)

    hpmap = hpmap / counts
    hpmap[~(counts > 0.0)] = hp.UNSEEN
    hp.write_map(output, hpmap, fits_IDL=False)
    print(f'wrote {output}')        
    

def bin_clmock(fnl, survey, iscont, method, ell_edges, log=True):
    print(ell_edges)
    p = '/fs/ess/PHS0336/data/lognormal/v3/clustering/'
    cl_files = glob(f'{p}clmock_{iscont}_*_lrg_{fnl}_{survey}_256_{method}.npy')
    print(fnl, survey, iscont, method)
    if log:        
        file_out1 = f'{p}logclmock_{iscont}_lrg_{fnl}_{survey}_256_{method}_mean.npz'
        file_out2 = f'{p}logclmock_{iscont}_lrg_{fnl}_{survey}_256_{method}_cov.npz'    
    else:
        file_out1 = f'{p}clmock_{iscont}_lrg_{fnl}_{survey}_256_{method}_mean.npz'
        file_out2 = f'{p}clmock_{iscont}_lrg_{fnl}_{survey}_256_{method}_cov.npz'        
    print(len(cl_files), cl_files[0])
    assert len(cl_files) == 1000
    
    cl_gg = []
    cl_ggb = []    
    for file_i in cl_files:
        cl_i = np.load(file_i, allow_pickle=True).item()
        cl_gg.append(cl_i['cl_gg']['cl'])        
        lb, clb = ut.histogram_cell(cl_i['cl_gg']['l'], cl_i['cl_gg']['cl'], bins=ell_edges)
        cl_ggb.append(clb)

    cl_gg = np.array(cl_gg)
    cl_ggb = np.array(cl_ggb)
    nmocks, nell = cl_gg.shape
    nbins = cl_ggb.shape[1]
    print(nmocks, nell, nbins)
    
    hf = (nmocks - 1.0)/(nmocks - nbins - 2.0)
    if log:    
        print('going to do log transform')
        cl_cov = np.cov(np.log10(cl_ggb), rowvar=False)*hf / nmocks
        cl_mean = np.log10(cl_ggb.mean(axis=0))
    else:
        cl_cov = np.cov(cl_ggb, rowvar=False)*hf / nmocks
        cl_mean = cl_ggb.mean(axis=0)
        
    inv_cov = np.linalg.inv(cl_cov)
    print(f'Hartlap with #mocks ({nmocks}) and #bins ({nbins}): {hf:.2f}' )    
    np.savez(file_out1, **{'el_edges':ell_edges, 'el_bin':lb, 'cl':cl_mean})
    np.savez(file_out2, **{'el_edges':ell_edges, 'el_bin':lb, 'clcov':cl_cov})    
    print('wrote', file_out1, file_out2)    

    
    
def create_cl(save=False):
    z, b, dNdz = init_sample(kind='lrg', plot=True)

    model = Spectrum()
    model.add_tracer(z, b, dNdz, p=1.0)
    model.add_kernels(np.arange(2000))


    fg, ax = plt.subplots()

    el = np.arange(2000)
    for fnl, name in zip([-200, -100, 0, 100, 200],
                         ['ne200', 'ne100', 'zero', 'po100', 'po200']):

        cl_raw = model(el, fnl=fnl, b=1.43)
        ax.plot(cl_raw, label='%.1f'%fnl)
        if save:            
            np.savetxt(f'/users/PHS0336/medirz90/github/flask/data/desiCl{name}-f1z1f1z1.dat', 
                       np.column_stack([el, cl_raw]), header='# el -- Cel')

    ax.set(xlim=(-1, 22), yscale='log', xlabel=r'$\ell$', 
           ylim=(2.0e-7, 3.0e-2), ylabel=r'$C_\ell$')
    ax.legend(ncol=2)
    #bbox_to_anchor=(1.5, 1.))
    #ax[1].set(xlim=(1, 700))  
    
    
def plot_radec():
    fig, ax = plt.subplots(ncols=2, figsize=(12, 4), sharex=True, sharey=True)
    
    for r in ['bmzls', 'ndecals', 'sdecals']:
        table = f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_{r}_256.fits'
        dt = ft.read(table)
        ra, dec = ut.hpix2radec(256, dt['hpix'])
        ax[0].scatter(ut.shiftra(ra[::50]), np.sin(np.radians(dec[::50])), 1, marker='.')
        
    for r in ['bmzls', 'ndecalsc', 'sdecalsc']:
        table = f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_{r}_256.fits'
        dt = ft.read(table)
        ra, dec = ut.hpix2radec(256, dt['hpix'])
        ax[1].scatter(ut.shiftra(ra[::50]), np.sin(np.radians(dec[::50])), 1, marker='.')        

    for a in ax:
        a.set_xlabel('RA [deg]')
        a.set_ylabel('sin(DEC [deg])')
    
    
    
def combine_regions():
    bmzls = ft.read('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_bmzls_256.fits')
    ndecals = ft.read('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_ndecals_256.fits')

    common = np.intersect1d(bmzls['hpix'], ndecals['hpix'])
    ngc = np.concatenate([bmzls, ndecals])

    mask_comm = np.in1d(ngc['hpix'], common)
    ngc_comm = ngc[mask_comm]
    ngc_comm = np.sort(ngc_comm, order='hpix')
    commons = np.zeros(common.size, dtype=ngc_comm.dtype)

    for key in commons.dtype.names:

        if key=='hpix':
            commons[key] = ngc_comm[key][:-1:2]
        elif key in ['label', 'fracgood']:
            commons[key] = (ngc_comm[key][:-1:2]+ngc_comm[key][1::2])
        elif key=='features':
            commons[key] = (ngc_comm[key][:-1:2]+ngc_comm[key][1::2]) / 2.0
        else:
            print(key)

    ngc_unique = ngc[~mask_comm]
    combined = np.concatenate([ngc_unique, commons])
    assert ngc.size ==2*commons.size+ngc_unique.size        

    ft.write('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_ngc_256.fits', combined)


    # Combine NGC and DECaLS S into desi
    ngc = ft.read('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_ngc_256.fits')
    sdecals = ft.read('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_sdecals_256.fits')

    common = np.intersect1d(sdecals['hpix'], ngc['hpix'])
    assert len(common) == 0
    desi = np.concatenate([ngc, sdecals])

    ft.write('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_desi_256.fits', desi)

    # Combine DECaLS N and DECaLS S into DECaLS
    ndecals = ft.read('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_ndecals_256.fits')
    sdecals = ft.read('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_sdecals_256.fits')

    common = np.intersect1d(sdecals['hpix'], ndecals['hpix'])
    assert len(common) == 0
    decals = np.concatenate([ndecals, sdecals])

    ft.write('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_decals_256.fits', decals)
    
def combine_regions2():
    """
        Apply dec cuts to get the '***c_256.fits' files 
        
    """
    bmzls = ft.read('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_bmzls_256.fits')
    ndecals = ft.read('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_ndecalsc_256.fits')

    common = np.intersect1d(bmzls['hpix'], ndecals['hpix'])
    ngc = np.concatenate([bmzls, ndecals])

    mask_comm = np.in1d(ngc['hpix'], common)
    ngc_comm = ngc[mask_comm]
    ngc_comm = np.sort(ngc_comm, order='hpix')
    commons = np.zeros(common.size, dtype=ngc_comm.dtype)

    for key in commons.dtype.names:

        if key=='hpix':
            commons[key] = ngc_comm[key][:-1:2]
        elif key in ['label', 'fracgood']:
            commons[key] = (ngc_comm[key][:-1:2]+ngc_comm[key][1::2])
        elif key=='features':
            commons[key] = (ngc_comm[key][:-1:2]+ngc_comm[key][1::2]) / 2.0
        else:
            print(key)

    ngc_unique = ngc[~mask_comm]
    combined = np.concatenate([ngc_unique, commons])
    assert ngc.size ==2*commons.size+ngc_unique.size        

    ft.write('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_ngcc_256.fits', combined)


    # Combine NGC and DECaLS S into desi
    ngc = ft.read('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_ngcc_256.fits')
    sdecals = ft.read('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_sdecalsc_256.fits')

    common = np.intersect1d(sdecals['hpix'], ngc['hpix'])
    assert len(common) == 0
    desi = np.concatenate([ngc, sdecals])

    ft.write('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_desic_256.fits', desi)    
    
def plot_ngal():
    desi = ft.read('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_desi_256.fits')
    ng  = ut.make_hp(256, desi['hpix'], desi['label']/(desi['fracgood']*hp.nside2pixarea(256, True)), np.inf)
    print(np.mean(ng[desi['hpix']]))

    dv.mollview(ng, 400, 1200, r'Target Density [deg$^{-2}$]', 
            cmap='YlOrRd_r', colorbar=True, galaxy=True)
    plt.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/lrgdens.pdf', bbox_inches='tight')    

def plot_npred():
    hp_known = hp.read_map('/fs/ess/PHS0336/data/rongpu/imaging_sys/regression/0.57.0/linp_lrg_desic_known.hp256.fits')
    hp_known1 = hp.read_map('/fs/ess/PHS0336/data/rongpu/imaging_sys/regression/0.57.0/linp_lrg_desic_known1.hp256.fits')
    hp_all = hp.read_map('/fs/ess/PHS0336/data/rongpu/imaging_sys/regression/0.57.0/linp_lrg_desic_all.hp256.fits')
    hp_nknown1 = hp.read_map('/fs/ess/PHS0336/data/rongpu/imaging_sys/regression/0.57.0/dnnp_lrg_desic_known1.hp256.fits')

    sf = hp.nside2pixarea(256, True)
    for hp_i in [hp_all, hp_known, hp_known1, hp_nknown1]:
        is_g = hp_i != hp.UNSEEN
        print(np.percentile(hp_i[is_g]/sf, [1, 99]))

    fig = plt.figure(figsize=(6, 7))
    ax0  = fig.add_axes([0., 1.,  1., 1], projection='mollweide')
    ax1  = fig.add_axes([0,  0.6, 1., 1],  projection='mollweide')
    ax2  = fig.add_axes([0., 0.2, 1., 1],  projection='mollweide')
    ax3  = fig.add_axes([0., -0.2, 1., 1], projection='mollweide')

    kw = {'vmin':760., 'vmax':890., 'cmap':dv.mycolor(), 'in_deg':True}
    dv.mollview(hp_known,  figax=[fig, ax0], **kw)
    dv.mollview(hp_known1, figax=[fig, ax1], **kw)
    dv.mollview(hp_all,    figax=[fig, ax2], **kw)
    dv.mollview(hp_nknown1, figax=[fig, ax3], 
                colorbar=True, galaxy=False, unit=r'Predicted density [deg$^{-2}$]', **kw)

    for ni, axi in zip(['Conservative I', 'Conservative II', 'Baseline', 'Nonlinear (Cons. II)'], 
                                      [ax0, ax1, ax2, ax3]):
        axi.text(0.4, 0.2, ni, transform=axi.transAxes)
    fig.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/npred.pdf', bbox_inches='tight')    
    
    
def plot_pcc():
    pccs = {}
    for region in ['bmzls', 'ndecalsc', 'sdecalsc']:
        d_ = ft.read(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_{region}_256.fits')
        y_ = pcc(d_['features'], d_['label']/d_['fracgood'], kind='pearson')

        mocks = glob(f'/fs/ess/PHS0336/data/lognormal/v3/hpmaps/lrghp-zero-*-f1z1.fits')[::10]
        print(len(mocks))
        ym_ = []
        for mock in mocks:
            dm_ = hp.read_map(mock, verbose=False)
            ym_.append(pcc(d_['features'], dm_[d_['hpix']], kind='pearson'))

        pccs[region] = y_, np.array(ym_)
        print(region, 'done')
        
    colors = ['C0', 'C1', 'C2']
    names = [r'EBV', r'nStar']+[fr'depth$_{b}$' for b in ['g', 'r', 'z', '{w1}']]\
            + [fr'psfsize$_{b}$' for b in ['g', 'r', 'z']]

    for i, (region, name) in enumerate(zip(['bmzls', 'ndecalsc', 'sdecalsc'],
                                          ['BASS+MzLS', 'DECaLS North', 'DECaLS South'])):
        p_, er_ = pccs[region]

        pcc_ = p_[0]
        err_ = er_[:, 0, :]
        print(err_.shape, len(pcc_))
        pcc_min, pcc_max = np.percentile(err_, [0, 100], axis=0)

        x = np.arange(len(pcc_))+i*0.2
        plt.bar(x, pcc_, width=0.2, alpha=0.6, color=colors[i], label=name)
        plt.plot(x, pcc_min, ls='-', lw=1, color=colors[i], alpha=0.5)
        plt.plot(x, pcc_max, ls='-', lw=1, color=colors[i], alpha=0.5)

    plt.xticks(x, names, rotation=90)
    plt.ylabel('PCC (gal density, Imaging)')
    plt.legend()
    plt.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/pcc.pdf', bbox_inches='tight')
    
    
def plot_corrmax():
    desi = ft.read('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_desic_256.fits')
    names = [r'EBV', r'nStar']+[fr'depth$_{b}$' for b in ['g', 'r', 'z', '{w1}']]\
            + [fr'psfsize$_{b}$' for b in ['g', 'r', 'z']]

    d = pd.DataFrame(data=desi['features'], columns=names)
    corr_cf = d.corr()
    mask = np.ones_like(corr_cf, dtype=np.bool)
    mask[np.tril_indices_from(mask)] = False
    f, ax = plt.subplots(figsize=(6, 4))
    kw = dict(mask=mask, cmap=plt.cm.seismic, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    sns.heatmap(corr_cf, **kw)
    ax.yaxis.set_tick_params(right=False)
    ax.xaxis.set_tick_params(top=False)
    plt.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/pccx.pdf', bbox_inches='tight')
    plt.show()
    
    
def plot_clxtest():
    names = [r'EBV', r'nStar']+[fr'depth$_{b}$' for b in ['g', 'r', 'z', '{w1}']]\
            + [fr'psfsize$_{b}$' for b in ['g', 'r', 'z']]    
    ell_edges = ut.ell_edges[:10]
    print(f'ell edges: {ell_edges}')

    p = '/fs/ess/PHS0336/data/lognormal/v3/clustering/'
    err_0 = read_clxmocks(glob(f'{p}clmock_0_*_lrg_zero_desic_256_noweight.npy'),
                            ell_edges)
    err_100 = read_clxmocks(glob(f'{p}clmock_0_*_lrg_po100_desic_256_noweight.npy'),
                            ell_edges)

    chi2s = {}
    chi2s['fNL=0'] = ut.get_chi2pdf(err_0)
    chi2s['fNL=100'] = ut.get_chi2pdf(err_100)


    err_dr9 = read_clx('/fs/ess/PHS0336/data/rongpu/imaging_sys/clustering/0.57.0/cl_lrg_desic_256_noweight.npy', ell_edges)
    err_dr9all = read_clx('/fs/ess/PHS0336/data/rongpu/imaging_sys/clustering/0.57.0/cl_lrg_desic_256_linp_all.npy', ell_edges)
    err_dr9known = read_clx('/fs/ess/PHS0336/data/rongpu/imaging_sys/clustering/0.57.0/cl_lrg_desic_256_linp_known.npy', ell_edges)
    err_dr9known1 = read_clx('/fs/ess/PHS0336/data/rongpu/imaging_sys/clustering/0.57.0/cl_lrg_desic_256_linp_known1.npy', ell_edges)
    err_dr9nknown1 = read_clx('/fs/ess/PHS0336/data/rongpu/imaging_sys/clustering/0.57.0/cl_lrg_desic_256_dnnp_known1.npy', ell_edges)

    icov, cov_0 = ut.get_inv(err_0, return_cov=True)
    cov_100 = ut.get_inv(err_100, return_cov=True)[1]

    chi2_dr9 = ut.chi2_fn(err_dr9, icov)
    chi2_dr9all = ut.chi2_fn(err_dr9all, icov)
    chi2_dr9known = ut.chi2_fn(err_dr9known, icov)
    chi2_dr9known1 = ut.chi2_fn(err_dr9known1, icov)
    chi2_dr9nknown1 = ut.chi2_fn(err_dr9nknown1, icov)

    fg, ax = plt.subplots(sharex=True, figsize=(14, 6))
    fg.subplots_adjust(hspace=0.)

    err_0m = err_0.mean(axis=0)
    err_100m = err_100.mean(axis=0)

    ell_b = np.arange(err_0m.size)
    err_0e = np.diagonal(cov_0)**0.5
    err_100e = np.diagonal(cov_100)**0.5

    ln1 = ax.fill_between(ell_b, 0.0, 1.+(err_0e/err_0m),  label=r'$f_{\rm NL}=0$', color='k', alpha=0.08)
    ln2 = ax.fill_between(ell_b+0.1, 0.0, (err_100m+err_100e)/err_0m, label=r'$f_{\rm NL}=100$', color='k', alpha=0.04)
    lgn1 = plt.legend(handles=[ln1, ln2], loc='upper right', title='Clean Mocks')

    kw = dict()
    ln3, = ax.plot(ell_b, err_dr9/err_0m,      label='DR9 (Before)', )
    ln4, = ax.plot(ell_b, err_dr9all/err_0m,    label='DR9 (Baseline)')
    ln5, = ax.plot(ell_b, err_dr9known/err_0m,  label='DR9 (Conservative I)')
    ln6, = ax.plot(ell_b, err_dr9known1/err_0m, label='DR9 (Conservative II)')
    ln7, = ax.plot(ell_b, err_dr9nknown1/err_0m, label='DR9 (Nonlinear Cons. II)')
    ax.legend(handles=[ln3, ln4, ln5, ln6, ln7], 
                 bbox_to_anchor=(0, 1.02, 1, 0.4), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=3, frameon=False)

    plt.gca().add_artist(lgn1)
    ax.set_yscale('symlog', linthreshy=10)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200])
    ax.set_ylabel(r'$C_{X, \ell}/C_{X, \ell}^{f_{\rm NL}=0~{\rm mocks}}$')
    for xt in np.arange(0, 82, 9):ax.axvline(xt, ls=':', lw=1, color='grey')
    ax.set_xticks([4.5+i*9 for i in range(len(names))])
    ax.set_xticklabels(names, rotation=90)
    ax.annotate("Linear", (-2, 0.0), (-2, 4), arrowprops=dict(arrowstyle='->', relpos=(0, 0.2)), rotation=90, fontsize=13, color='grey')
    ax.annotate("", (-2, +10.0), (-2, 6.2), arrowprops=dict(arrowstyle='->'), rotation=90, color='grey')
    ax.annotate("Logarithmic", (-2, 10.0), (-2, 25), arrowprops=dict(arrowstyle='->', relpos=(0, 0)), rotation=90, fontsize=13, color='grey')
    ax.annotate("", (-2, 200.0), (-2, 85), arrowprops=dict(arrowstyle='->'), rotation=90, color='grey')  
    fg.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/clx_mocks.pdf', bbox_inches='tight')    
    plt.show()
    
    plt.figure()
    ls = ['-', '--']
    for i, (name_i, chi2_i) in enumerate(chi2s.items()):
        print(np.max(chi2_i), np.min(chi2_i))
        plt.hist(chi2_i, histtype='step', bins=58, ls=ls[i],
                 label=name_i, range=(0, 550.))    
    # plt.yscale('log')

    plt.text(180., 48., f'DR9 (Before) $\chi^{2}$ = {chi2_dr9:.1f}', fontsize=12)
    plt.text(180., 41., f'DR9 (Conservative I) $\chi^{2}$ = {chi2_dr9known:.1f}', fontsize=12)
    plt.text(180., 34., f'DR9 (Conservative II) $\chi^{2}$ = {chi2_dr9known1:.1f}', fontsize=12)
    plt.text(180., 27., f'DR9 (Baseline) $\chi^{2}$ = {chi2_dr9all:.1f}', fontsize=12)
    plt.text(180., 20., f'DR9 (Nonlinear cons. II) $\chi^{2}$ = {chi2_dr9nknown1:.1f}', fontsize=12)
    # plt.yscale('log')
    plt.xticks([0, 100, 200, 300, 400, 500])
    plt.legend(title='Clean Mocks', frameon=False)
    plt.xlabel(r'Cross Spectrum $\chi^{2}$')
    plt.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/chi2test.pdf', bbox_inches='tight')    
    
    
    
def plot_nbartest():
    names = [r'EBV', r'nStar']+[fr'depth$_{b}$' for b in ['g', 'r', 'z', '{w1}']]\
            + [fr'psfsize$_{b}$' for b in ['g', 'r', 'z']]    
    
    err_0 = read_nbmocks(glob('/fs/ess/PHS0336/data/lognormal/v3/clustering/nbarmock_0_*_lrg_zero_desic_256_noweight.npy'))
    err_100 = read_nbmocks(glob('/fs/ess/PHS0336/data/lognormal/v3/clustering/nbarmock_0_*_lrg_po100_desic_256_noweight.npy'))
    icov, cov_0 = ut.get_inv(err_0, return_cov=True)
    cov_100 = ut.get_inv(err_100, return_cov=True)[1]
    
    err_dr9 = read_nnbar('/fs/ess/PHS0336/data/rongpu/imaging_sys/clustering/0.57.0/nbar_lrg_desic_256_noweight.npy')
    err_dr9all = read_nnbar('/fs/ess/PHS0336/data/rongpu/imaging_sys/clustering/0.57.0/nbar_lrg_desic_256_linp_all.npy')
    err_dr9known = read_nnbar('/fs/ess/PHS0336/data/rongpu/imaging_sys/clustering/0.57.0/nbar_lrg_desic_256_linp_known.npy')
    err_dr9known1 = read_nnbar('/fs/ess/PHS0336/data/rongpu/imaging_sys/clustering/0.57.0/nbar_lrg_desic_256_linp_known1.npy')
    err_dr9nknown1 = read_nnbar('/fs/ess/PHS0336/data/rongpu/imaging_sys/clustering/0.57.0/nbar_lrg_desic_256_dnnp_known1.npy') 
    
    chi2_dr9 = ut.chi2_fn(err_dr9, icov)
    chi2_dr9all = ut.chi2_fn(err_dr9all, icov)
    chi2_dr9known = ut.chi2_fn(err_dr9known, icov)
    chi2_dr9known1 = ut.chi2_fn(err_dr9known1, icov)
    chi2_dr9nknown1 = ut.chi2_fn(err_dr9nknown1, icov)
    
    fg, ax = plt.subplots(sharex=True, figsize=(14, 6))
    fg.subplots_adjust(hspace=0.)

    err_0m = err_0.mean(axis=0)
    err_100m = err_100.mean(axis=0)

    ell_b = np.arange(err_0m.size)
    err_0e = np.diagonal(cov_0)**0.5
    err_100e = np.diagonal(cov_100)**0.5

    ln1 = ax.fill_between(ell_b, err_0m-err_0e, err_0m+err_0e,  label='fNL=0', color='k', alpha=0.1)
    ln2 = ax.fill_between(ell_b+0.1, err_100m-err_100e, err_100m+err_100e, label='fNL=100', color='k', alpha=0.05)
    lgn1 = plt.legend(handles=[ln1, ln2], loc='upper right', title='Clean Mocks')

    kw = dict()
    ln3, = ax.plot(ell_b, err_dr9,      label='DR9 (Before)', )
    ln4, = ax.plot(ell_b, err_dr9all,    label='DR9 (Baseline)')
    ln5, = ax.plot(ell_b, err_dr9known,  label='DR9 (Conservative I)')
    ln6, = ax.plot(ell_b, err_dr9known1, label='DR9 (Conservative II)')
    ln7, = ax.plot(ell_b, err_dr9nknown1, label='DR9 (Nonlinear cons. II)')
    ax.legend(handles=[ln3, ln4, ln5, ln6, ln7], 
                 bbox_to_anchor=(0, 1.02, 1, 0.4), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=4, frameon=False)
    plt.gca().add_artist(lgn1)
    ax.set_ylabel('Mean Density Contrast')
    for xt in np.arange(0, 80, 8):
        ax.axvline(xt, ls=':', lw=1, color='k')
    ax.set_xticks([4+i*8 for i in range(len(names))])
    ax.set_xticklabels(names, rotation=90)
    fg.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/nbar_mocks.pdf', bbox_inches='tight')    
    plt.show()
    
    
    chi2s = {}
    chi2s['fNL=0'] = ut.get_chi2pdf(err_0)
    chi2s['fNL=100'] = ut.get_chi2pdf(err_100)
    plt.figure()
    ls = ['-', '--']
    for i, (name_i, chi2_i) in enumerate(chi2s.items()):
        print(np.max(chi2_i), np.min(chi2_i))
        plt.hist(chi2_i, histtype='step', bins=65, ls=ls[i],
                 label=name_i, range=(30, 160.))    

    plt.text(91., 48., f'DR9 (Before) $\chi^{2}$ = {chi2_dr9:.1f}', fontsize=12)
    plt.text(91., 43., f'DR9 (Conservative I) $\chi^{2}$ = {chi2_dr9known:.1f}', fontsize=12)
    plt.text(91., 38., f'DR9 (Conservative II) $\chi^{2}$ = {chi2_dr9known1:.1f}', fontsize=12)
    plt.text(91., 33., f'DR9 (Baseline) $\chi^{2}$ = {chi2_dr9all:.1f}', fontsize=12)
    plt.text(91., 28., f'DR9 (Nonlinear cons. II) $\chi^{2}$ = {chi2_dr9nknown1:.1f}', fontsize=12)
    plt.xlim(26, 210)
    plt.xticks([50, 75, 100, 125, 150, 175, 200])
    plt.legend(title='Clean Mocks', frameon=False)
    plt.xlabel(r'Mean Density $\chi^{2}$')
    plt.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/chi2test2.pdf', bbox_inches='tight')    
 
            
def plot_clhist():
    
    def hist(ax, x, **kw):
        std_ = np.std(x)
        ax.hist(x, **kw)
        ax.text(0.6, 0.8, f'$\sigma$={std_:.2f}', 
                transform=ax.transAxes, color=kw['color'])
        
    print('bins', ut.ell_edges)
    p = '/fs/ess/PHS0336/data/lognormal/v3/clustering/'
   
    cl_ggb = []    
    for file_i in glob(f'{p}clmock_0_*_lrg_zero_desic_256_noweight.npy'):
        cl_i = np.load(file_i, allow_pickle=True).item()
        lb, clb = ut.histogram_cell(cl_i['cl_gg']['l'], cl_i['cl_gg']['cl'], bins=ut.ell_edges)
        cl_ggb.append(clb)
    cl_0 = np.array(cl_ggb)
    
    cl_ggb1 = []    
    for file_i in glob(f'{p}clmock_0_*_lrg_po100_desic_256_noweight.npy'):
        cl_i = np.load(file_i, allow_pickle=True).item()
        lb, clb = ut.histogram_cell(cl_i['cl_gg']['l'], cl_i['cl_gg']['cl'], bins=ut.ell_edges)
        cl_ggb1.append(clb)
    cl_100 = np.array(cl_ggb1)    
    
    fig, ax = plt.subplots(ncols=2, nrows=2, 
                           figsize=(12, 8), sharey=True)
    ax = ax.flatten()
    fig.subplots_adjust(wspace=0.02, hspace=0.3)

    kw = dict(histtype='step', alpha=0.8)
    hist(ax[0], 1.0e5*cl_0[:, 0], color='C0', **kw)
    hist(ax[1], 1.0e5*cl_100[:, 0], color='C0', **kw)
    hist(ax[2], np.log10(cl_0[:, 0]), color='C1', **kw)
    hist(ax[3], np.log10(cl_100[:, 0]), color='C1', **kw)

    ax[0].set(yticks=[], ylabel='Distribution of first bin')
    ax[2].set(yticks=[], ylabel='Distribution of first bin')    
    for i, fnl in zip([0, 1], [0, 100]):        
        ax[i].set(xlabel=r'First bin $C_{\ell}~[x 10^{-5}]$')        
        ax[i].text(0.15, 0.2, fr'$f_{{\rm NL}}={fnl:d}$', transform=ax[i].transAxes)        
        ax[2+i].set(xlabel=r'Log of first bin $C_{\ell}$')
        ax[2+i].text(0.5, 0.2, fr'$f_{{\rm NL}}={fnl:d}$', transform=ax[2+i].transAxes, color='C1')

    fig.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/hist_cl.pdf', bbox_inches='tight')
    
    
def plot_mcmc_mocks():
    stg = {'mult_bias_correction_order':0,'smooth_scale_2D':0.15, 
           'smooth_scale_1D':0.3, 'contours': [0.68, 0.95]}
    mc_kw = dict(names=['fnl', 'b', 'n0'], 
                 labels=['f_{NL}', 'b', '10^{7}n_{0}'], settings=stg) 
    read_kw = dict(ndim=3, iscale=[2])

    po0 = MCMC('/fs/ess/PHS0336/data/lognormal/v3/mcmc/mcmc_0_lrg_pozero_desic_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    lpo0 = MCMC('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logmcmc_0_lrg_pozero_desic_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    po100 = MCMC('/fs/ess/PHS0336/data/lognormal/v3/mcmc/mcmc_0_lrg_po100_desic_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    lpo100 = MCMC('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logmcmc_0_lrg_po100_desic_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    

    stats = {}
    stats[r'log$C_{\ell}$'] = lpo100.stats
    stats[r'$C_{\ell}$'] = po100.stats
    stats[r'log$C_{\ell}$ using $f_{\rm NL}=0$ cov'] = lpo0.stats
    stats[r'$C_{\ell}$ using $f_{\rm NL}=0$ cov'] = po0.stats


    g = plots.get_single_plotter(width_inch=4*1.5)
    g.settings.legend_fontsize = 14
    g.plot_2d([lpo100, po100, lpo0, po0, ], 'fnl', 'b', filled=True, colors=['C0', 'C1', 'C2', 'C3'])
    g.add_x_marker(100)
    g.add_y_marker(1.43)
    g.get_axes().set_ylim(1.426, 1.434)
    g.get_axes().set_xlim(100-2.2, 100+3.2)
    g.add_legend(stats.keys(), colored_text=True, legend_ncol=2)
    ax = g.get_axes()
    ax.text(0.08, 0.08, r'Fitting the mean of $f_{\rm NL}$=100 mocks', 
            transform=ax.transAxes, fontsize=13, color='grey', alpha=0.8)
    ax.text(98, 1.4302, 'Truth', color='grey', fontsize=13, alpha=0.7)
    g.fig.align_labels()  
    g.fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/mcmc_po100.pdf', bbox_inches='tight')
    plt.show()
    
    
    ze = MCMC('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logmcmc_0_lrg_zero_desic_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    nd = MCMC('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logmcmc_0_lrg_zero_ndecalsc_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    sd = MCMC('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logmcmc_0_lrg_zero_sdecalsc_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    bm = MCMC('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logmcmc_0_lrg_zero_bmzls_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    
    stats['0 [DESI]'] = ze.stats
    stats['0 [DECaLS North]'] = nd.stats
    stats['0 [DECaLS South]'] = sd.stats
    stats['0 [BASS+MzLS]'] = bm.stats   
    
    g = plots.get_single_plotter(width_inch=4*1.5)
    g.settings.legend_fontsize = 14
    g.plot_2d([nd, sd, bm, ze], 'fnl', 'b', filled=True)
    g.add_x_marker(0)
    g.add_y_marker(1.43)
    g.get_axes().set_ylim(1.426, 1.434)
    g.get_axes().set_xlim(-2.2, 3.2)    
    ax = g.get_axes()
    ax.text(0.08, 0.92, r'Fitting the mean of $f_{\rm NL}$=0 mocks', 
            transform=ax.transAxes, fontsize=13, color='grey', alpha=0.8)
    ax.text(-2.0, 1.4302, 'Truth', color='grey', fontsize=13, alpha=0.7)
    
    g.add_legend(['DECaLS North', 'DECaLS South', 'BASS+MzLS', r'DESI'], 
                 colored_text=True, legend_loc='lower left')    
    g.fig.align_labels()
    g.fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/mcmc_zero.pdf', bbox_inches='tight')        
    print_stats(stats)    

    
def plot_bestfit():
    
    
    bf = np.load('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logbestfit_0_lrg_zero_desic_256_noweight.npz')
    plt.scatter(*bf['params'][:, :2].T, s=2, marker='.', alpha=0.5, color='C0')
    plt.scatter(*bf['params'].mean(axis=0)[:2],  s=300, marker='*', color='C1', alpha=0.8)
    print(bf['params'].mean(axis=0))
    print(bf['params'].std(axis=0))
    plt.axvline(0, lw=1, ls=':', color='grey', zorder=-10)
    plt.axhline(1.43, lw=1, ls=':', color='grey', zorder=-10)
    plt.text(-75, 1.432, 'Truth', color='grey', alpha=0.8, fontsize=13)
    plt.xlabel(r'$f_{\rm NL}$')
    plt.ylabel('b')
    
    plt.twinx()
    plt.yticks([])
    plt.hist(bf['params'][:, 0], zorder=-10, histtype='step', bins=24, color='C0')
    plt.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/bestfit_zero.pdf', bbox_inches='tight')    
    plt.show()
    
    
    bf = np.load('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logbestfit_0_lrg_po100_desic_256_noweight.npz')
    plt.scatter(*bf['params'][:, :2].T, s=2, marker='.', alpha=0.5, color='C0')
    plt.scatter(*bf['params'].mean(axis=0)[:2],  s=300, marker='*', color='C1', alpha=0.8)
    print(bf['params'].mean(axis=0))
    print(bf['params'].std(axis=0))
    plt.axvline(100, lw=1, ls=':', color='grey', zorder=-10)
    plt.axhline(1.43, lw=1, ls=':', color='grey', zorder=-10)
    plt.text(50, 1.432, 'Truth', color='grey', alpha=0.8, fontsize=13)
    plt.xlabel(r'$f_{\rm NL}$')
    plt.ylabel('b')
    
    plt.twinx()
    plt.yticks([])
    plt.hist(bf['params'][:, 0], zorder=-10, histtype='step', bins=24, color='C0')
    plt.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/bestfit_po100.pdf', bbox_inches='tight')    


    
def plot_mcmc_data():
    stg = {'mult_bias_correction_order':0,'smooth_scale_2D':0.15, 
           'smooth_scale_1D':0.3, 'contours': [0.68, 0.95]}
    mc_kw = dict(names=['fnl', 'b', 'n0'], 
                 labels=['f_{NL}', 'b', '10^{7}n_{0}'], settings=stg) 
    read_kw = dict(ndim=3, iscale=[2])
    
    ze = MCMC('/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/logmcmc_lrg_zero_desic_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    po = MCMC('/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/logmcmc_lrg_zero_desic_linp_all_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    kn = MCMC('/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/logmcmc_lrg_zero_desic_linp_known_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    kn1 = MCMC('/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/logmcmc_lrg_zero_desic_linp_known1_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)        
    knn1 = MCMC('/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/logmcmc_lrg_zero_desic_dnnp_known1_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)            
    knn1b = MCMC('/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/logmcmc_lrg_zero_bmzls_dnnp_known1_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)                
    knn1n = MCMC('/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/logmcmc_lrg_zero_ndecalsc_dnnp_known1_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)                
    knn1s = MCMC('/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/logmcmc_lrg_zero_sdecalsc_dnnp_known1_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)                    
    #knn2 = MCMC('/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/mcmc_lrg_zero_desi_dnnp_known2_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)            
    stats = {}
    stats['No Weight'] = ze.stats
    stats['Baseline'] = po.stats    
    stats['Conservative I'] = kn.stats    
    stats['Conservative II'] = kn1.stats
    stats['Nonlinear (Cons. II)'] = knn1.stats    
    stats['Nonlinear (Cons. II) - BMZLS'] = knn1b.stats    
    stats['Nonlinear (Cons. II) - NDEC'] = knn1n.stats    
    stats['Nonlinear (Cons. II) - SDEC'] = knn1s.stats        
    #stats['Nonlinear (Depths+EBV)'] = knn2.stats    

    # Triangle plot
    g = plots.get_single_plotter(width_inch=3*4*1.5)
    g.settings.legend_fontsize = 14
    #g.plot_2d([ze, po, kn, kn1], 'fnl', 'b', filled=True)
    g.triangle_plot([ze, po, kn, kn1, knn1], filled=True, legend_labels=['No weight', 'Baseline',
                                                                        'Conservative I', 'Conservative II',
                                                                        'Nonlinear (Cons. II)'])
    g.add_legend(['No weight', 'Baseline', 'Conservative I', 'Conservative II', 'Nonlinear (Cons. II)'], 
                 colored_text=True, legend_loc='lower left', )    
    g.fig.align_labels()
    g.fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/mcmc_dr9methods.pdf', bbox_inches='tight')    
    plt.show()
    
    
    # Triangle plot
    g = plots.get_single_plotter(width_inch=3*4*1.5)
    g.settings.legend_fontsize = 14
    #g.plot_2d([ze, po, kn, kn1], 'fnl', 'b', filled=True)
    g.triangle_plot([knn1b, knn1n, knn1s, knn1], filled=True, 
                    legend_labels=['BASS+MzLS', 'DECaLS North', 'DECaLS South', 'DESI'])
    g.add_legend(['BASS+MzLS', 'DECaLS North', 'DECaLS South', 'DESI'], 
                 colored_text=True, legend_loc='lower left', )    
    g.fig.align_labels()
    g.fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/mcmc_dr9regions.pdf', bbox_inches='tight')    
    plt.show()
    
    print_stats(stats)            
    

    nd = MCMC('/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/logmcmc_lrg_zero_ndecals_dnnp_known1_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)            
    ndc = MCMC('/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/logmcmc_lrg_zero_ndecalsc_dnnp_known1_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)            
    ndce = MCMC('/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/logmcmc_lrg_zero_ndecalsc_dnnp_known1ext_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)            
    sd = MCMC('/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/logmcmc_lrg_zero_sdecals_dnnp_known1_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)            
    sdc = MCMC('/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/logmcmc_lrg_zero_sdecalsc_dnnp_known1_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)            
    sdce = MCMC('/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/logmcmc_lrg_zero_sdecalsc_dnnp_known1ext_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)                


    stats = {}
    stats['North'] = ndc.stats
    stats['North [incl. CALIBZ+logHI]'] = ndce.stats    
    stats['North [incl. DEC < -11]'] = nd.stats
    stats['South'] = sdc.stats
    stats['South [incl. CALIBZ+logHI]'] = sdce.stats        
    stats['South [incl. DEC < -30]'] = sd.stats    

    labels = stats.keys()
    # Triangle plot
    g = plots.get_single_plotter(width_inch=4*1.5)
    g.settings.legend_fontsize = 14
    g.plot_2d([ndc, ndce, nd, sdc, sdce, sd], 'fnl', 'b', filled=True, legend_labels=labels)
    g.add_legend(labels, colored_text=True, legend_loc='lower left', )    
    g.fig.align_labels()
    g.fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/mcmc_dr9_cutdec.pdf', bbox_inches='tight')
    plt.show()
    print_stats(stats)
    
    
    

def plot_dr9vsmocks():
    knn1 = np.load('/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/logmcmc_lrg_zero_desic_dnnp_known1_steps10k_walkers50.npz')
    bf = np.load('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logbestfit_0_lrg_zero_desic_256_noweight.npz')

    fig, ax = plt.subplots(ncols=2, figsize=(12, 4), sharey=True)
    fig.subplots_adjust(wspace=0.05)

    ax[0].hist(2*bf['neglog']/1000., histtype='step', bins=18)
    ax[0].axvline(2*knn1['best_fit_logprob'], ls='--', color='C1')

    ax[1].hist(bf['params'][:, 0], histtype='step', bins=18)
    ax[1].axvline(knn1['best_fit'][0], ls='--', color='C1')

    ax[0].set(xlabel=r'min $\chi^{2}$', yticks=[])
    ax[1].set(xlabel=r'$f_{\rm NL}$')

    ax[0].text(0.2, 0.75, 'Mocks', transform=ax[0].transAxes)
    ax[0].text(0.55, 0.75, 'DR9', color='C1', transform=ax[0].transAxes)

    fig.align_xlabels() 
    fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/pdf_dr9vsmocks.pdf', bbox_inches='tight')   
    

    
def plot_model(fnltag='po100'):
    bm = np.load(f'/fs/ess/PHS0336/data/lognormal/v3/mcmc/logmcmc_0_lrg_{fnltag}_desic_256_noweight_steps10k_walkers50.npz')
    zbdndz = init_sample(kind='lrg')
    
    # read survey geometry
    dt = ft.read(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_desic_256.fits')
    w = np.zeros(12*256*256)
    w[dt['hpix']] = 1.0
    weight = hp.ud_grade(w, 1024)
    mask = weight > 0.5

    model = Spectrum()
    model.add_tracer(*zbdndz, p=1.0)
    model.add_kernels(np.arange(2000))


    wind = WindowSHT(weight, mask, np.arange(2048), ngauss=2048)
    fnl, b, noise = bm['best_fit']
    print(fnl, b)
    
    el_g = np.arange(2000)
    cl_bf = model(el_g, fnl=fnl, b=b, noise=noise)

    cl_bfw = wind.convolve(el_g, cl_bf)
    lmax = max(el_g)+1
    cl_bfwic = wind.apply_ic(cl_bfw[:lmax])

    cl_ = np.load(f'/fs/ess/PHS0336/data/lognormal/v3/clustering/logclmock_0_lrg_{fnltag}_desic_256_noweight_mean.npz')
    cl_cov_ = np.load(f'/fs/ess/PHS0336/data/lognormal/v3/clustering/logclmock_0_lrg_{fnltag}_desic_256_noweight_cov.npz')

    el_edges = cl_['el_edges']
    el = cl_['el_bin']
    cl = cl_['cl']
    cl_err = np.diagonal(cl_cov_['clcov'])**0.5

    cl_models = {}
    for name, cl_i in zip(['Best Fit Model', '+ Window Convolution', '+ Integral Constraint'],
                          [cl_bf, cl_bfw[:2000], cl_bfwic[:2000]]):
        cl_models[name] = ut.histogram_cell(el_g, cl_i, bins=el_edges)


    fig = plt.figure(figsize=(7, 7), constrained_layout=False)
    gs = GridSpec(3, 1, figure=fig)

    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[2, 0])

    lw = [0.8, 0.8, 3.]
    ls = ['-', '-', '-']
    al = [1., 1., 0.7]
    for i, (n, v) in enumerate(cl_models.items()):
        kw = dict(label=n, lw=lw[i], ls=ls[i], alpha=al[i])
        ax1.plot(v[0], np.log10(v[1]), **kw)
        ax2.plot(el, np.log10(v[1])-cl, **kw)

    f = np.sqrt(1000.)
    ax1.plot(el, cl, 'C0--', label='Mean of Mocks')
    #ax2.axhline(0.0, color='C0', ls='--')
    ax2.fill_between(el, -cl_err, cl_err, alpha=0.2, color='grey')
    ax2.fill_between(el, -f*cl_err, f*cl_err, alpha=0.05, color='grey')


    ax1.legend(ncol=1)
    ax1.set(xscale='log', ylabel=r'$\logC_{\ell}$') #, yscale='log')
    ax1.tick_params(labelbottom=False)
    ax2.set(xscale='log', xlabel=r'$\ell$', ylabel=r'$\Delta \logC_{\ell}$', xlim=ax1.get_xlim(), ylim=(-0.025, +0.025))

    fig.subplots_adjust(hspace=0.0, wspace=0.02)
    fig.align_labels()
    fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/model_mock.pdf', bbox_inches='tight')     
    
    
def plot_dr9cl():
    zbdndz = init_sample(kind='lrg')

    # read survey geometry
    dt = ft.read(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_desic_256.fits')
    w = np.zeros(12*256*256)
    w[dt['hpix']] = dt['fracgood']
    weight = hp.ud_grade(w, 1024)
    mask = weight > 0.5

    z, b, dNdz = init_sample(kind='lrg')
    model = SurveySpectrum()
    model.add_tracer(z, b, dNdz, p=1.0)
    model.add_kernels(model.el_model)
    model.add_window(weight, mask, np.arange(2048), ngauss=2048)  
    
    fnltag = 'zero'
    cl_ = np.load(f'/fs/ess/PHS0336/data/lognormal/v3/clustering/logclmock_0_lrg_{fnltag}_desic_256_noweight_mean.npz')
    cl_cov_ = np.load(f'/fs/ess/PHS0336/data/lognormal/v3/clustering/logclmock_0_lrg_{fnltag}_desic_256_noweight_cov.npz')

    cl_err = np.diagonal(cl_cov_['clcov']*1000.)**0.5

    mk = ['.', 'o', 'x', '^', 's', '1']
    el_g = np.arange(300)

    plt.figure(figsize=(8, 6))
    ln, = plt.plot(cl_['el_bin'], cl_['cl'], label='Mean of Mocks', alpha=0.5)
    plt.fill_between(cl_['el_bin'], cl_['cl']-cl_err, cl_['cl']+cl_err, alpha=0.1, color=ln.get_color())

    for i, (n, nm) in enumerate(zip(['linp_all', 'linp_known', 'linp_known1', 'dnnp_known1', 'noweight'],
                     ['Baseline', 'Conservative I', 'Conservative II','Nonlinear (Cons. II)', 'No weight'])):
        cl_d = np.load(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/clustering/0.57.0/cl_lrg_desic_256_{n}.npy', allow_pickle=True).item()
        cl_b = np.log10(ut.histogram_cell(cl_d['cl_gg']['l'], cl_d['cl_gg']['cl'], bins=ut.ell_edges)[1])

        bestp = np.load(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/logmcmc_lrg_zero_desic_{n}_steps10k_walkers50.npz')
        fnl, b, noise = bestp['best_fit']
        print(nm, fnl, b)

        cl_bf = np.log10(model(el_g, fnl=fnl, b=b, noise=noise))
        ln = plt.plot(el_g[2:], cl_bf[2:], lw=1, alpha=0.6)
        plt.scatter(cl_['el_bin'], cl_b, label=nm, 
                    marker=mk[i], color=ln[0].get_color(), alpha=0.8)

    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\logC_{\ell}$')
    #plt.ylim(1.2e-6, 2.2e-4)
    plt.xlim(xmin=1.9)
    plt.legend(ncol=1, loc='upper right')
    plt.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/model_dr9.pdf', bbox_inches='tight')         
    
# import sys
# import os
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from matplotlib.backends.backend_pdf import PdfPages

# import numpy as np
# import fitsio as ft
# import healpy as hp
# from glob import glob
# from time import time
# from scipy.optimize import minimize
# import pandas as pd

# HOME = os.getenv('HOME')
# print(f'running on {HOME}')
# sys.path.append(f'{HOME}/github/LSSutils')
# sys.path.append(f'{HOME}/github/sysnetdev')
# import sysnet.sources as src

# from lssutils.dataviz import setup_color, add_locators, mollview, mycolor
# from lssutils.utils import (histogram_cell, maps_dr9, make_hp,
#                             chi2_fn, get_chi2pdf, get_inv, hpix2radec, shiftra, make_overdensity)
# from lssutils.io import (read_nbmocks, read_nnbar, read_window, read_chain)
# from lssutils.theory.cell import (dNdz_model, init_sample, SurveySpectrum, Spectrum, bias_model_lrg)
# from lssutils.extrn.mcmc import Posterior
# from lssutils.extrn import corner
# from lssutils.stats.window import WindowSHT
# from lssutils.stats.pcc import pcc

# import getdist
# from getdist import plots, MCSamples

# ell_edges = np.array([2, 6, 10, 14, 18, 22, 26] 
#                    + [10*i for i in range(3,10)] \
#                    + [100+i*20 for i in range(5)] \
#                    + [200+i*50 for i in range(3)])

# class MCMC(MCSamples):
#      def __init__(self, path_to_mcmc, read_kw=dict(), mc_kw=dict()):
#             self.stats, chains = read_chain(path_to_mcmc, **read_kw)
#             MCSamples.__init__(self, samples=chains, **mc_kw)
            
            
# def bin_clmock(fnl, survey, iscont, method):
#     p = '/fs/ess/PHS0336/data/lognormal/v2/clustering/'
#     ell_edges = np.array([2, 6, 10, 14, 18, 22, 26] 
#                        + [10*i for i in range(3,10)] \
#                        + [100+i*20 for i in range(5)] \
#                        + [200+i*50 for i in range(3)])
#     cl_files = glob(f'{p}clmock_{iscont}_*_lrg_{fnl}_{survey}_256_{method}.npy')
#     print(fnl, survey, iscont, method)
#     file_out = f'{p}clmock_{iscont}_lrg_{fnl}_{survey}_256_{method}_mean.npz'
#     print(len(cl_files), cl_files[0])
#     assert len(cl_files) == 1000
    
#     cl_gg = []
#     cl_ggb = []
    
#     for file_i in cl_files:
#         cl_i = np.load(file_i, allow_pickle=True).item()
#         cl_gg.append(cl_i['cl_gg']['cl'])
        
#         lb, clb = histogram_cell(cl_i['cl_gg']['l'], cl_i['cl_gg']['cl'], bins=ell_edges)
#         cl_ggb.append(clb)

#     cl_gg = np.array(cl_gg)
#     cl_ggb = np.array(cl_ggb)   
#     nmocks, nell = cl_gg.shape
    
    
#     plt.figure()
#     plt.plot(cl_gg.mean(axis=0))
#     plt.plot(lb, cl_ggb.mean(axis=0), marker='o', mfc='w')
#     plt.fill_between(np.arange(nell), *np.percentile(cl_gg, [0, 100], axis=0), alpha=0.1)
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.ylim(1.0e-8, 1.0e-2)
    
#     np.savez(file_out, **{'el_edges':ell_edges, 'el_bin':lb, 'cl':cl_ggb.mean(axis=0)})
#     print('wrote', file_out)


    
    
# def plot_ngmoll():
#     desi = ft.read('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_desi_256.fits')
#     ng  = make_hp(256, desi['hpix'], desi['label']/(desi['fracgood']*hp.nside2pixarea(256, True)), np.inf)
#     print(np.mean(ng[desi['hpix']]))
#     mollview(ng, 400, 1200, r'Observed Galaxy Density [deg$^{-2}$]', 
#                  cmap='YlOrRd_r', colorbar=True, galaxy=True)
#     plt.savefig('figs/nlrg.pdf', bbox_inches='tight')
    
# def plot_nz():
#     nz = np.loadtxt('/fs/ess/PHS0336/data/rongpu/sv3_lrg_dndz_denali.txt')

#     fg, ax = plt.subplots()

#     ax.step(nz[:, 0], nz[:, 2], where='pre', lw=1)#, label='dN/dz')
#     ax.set(xlim=(-0.05, 1.45), xlabel='z', ylabel='dN/dz')
#     ax.text(0.25, 0.7, 'dN/dz', color='C0', transform=ax.transAxes)

#     ax1 = ax.twinx()
#     z_g = np.linspace(0.1, 1.35)

#     ax1.plot(z_g, 1.4*bias_model_lrg(z_g), 'C1--', lw=3, alpha=0.5, zorder=10)#, label='b(z)')
#     ax1.text(0.7, 0.48, 'b(z)$\propto$ D$^{-1}$(z)', color='C1', transform=ax1.transAxes)
#     ax1.set_ylabel('b(z)')
#     ax1.set_ylim((1.3, 3.1))

#     #ax1.legend(loc='upper right')
#     #ax.legend(loc='upper left')

#     fg.savefig('figs/nz_lrg.pdf', bbox_inches='tight')    
    

# def plot_model(fnltag='po100'):
#     bm = np.load(f'/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_{fnltag}_bmzls_noweight_steps10k_walkers50.npz')
#     zbdndz = init_sample(kind='lrg')
#     # read survey geometry
#     dt = ft.read(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_bmzls_256.fits')
#     w = np.zeros(12*256*256)
#     w[dt['hpix']] = 1.0
#     weight = hp.ud_grade(w, 1024)
#     mask = weight > 0.5

#     model = Spectrum()
#     model.add_tracer(*zbdndz, p=1.0)
#     model.add_kernels(np.arange(2000))


#     wind = WindowSHT(weight, mask, np.arange(2048), ngauss=2048)
#     fnl, b, noise = bm['best_fit']
#     print(fnl, b)
    
#     el_g = np.arange(2000)
#     cl_bf = model(el_g, fnl=fnl, b=b, noise=noise)

#     cl_bfw = wind.convolve(el_g, cl_bf)
#     lmax = max(el_g)+1
#     cl_bfwic = wind.apply_ic(cl_bfw[:lmax])

#     cl_ = np.load(f'/fs/ess/PHS0336/data/lognormal/v2/clustering/clmock_{fnltag}_bmzls_mean.npz')
#     cl_cov_ = np.load(f'/fs/ess/PHS0336/data/lognormal/v2/clustering/clmock_{fnltag}_bmzls_cov.npz')

#     el_edges = cl_['el_edges']
#     el = cl_['el_bin']
#     cl = cl_['cl']
#     cl_err = np.diagonal(cl_cov_['clcov']/1000.)**0.5

#     cl_models = {}
#     for name, cl_i in zip(['Best Fit Model', '+ Window Convolution', '+ Integral Constraint'],
#                           [cl_bf, cl_bfw[:2000], cl_bfwic[:2000]]):

#         cl_models[name] = histogram_cell(el_g, cl_i, bins=el_edges)


#     fig = plt.figure(figsize=(5, 5), constrained_layout=False)
#     gs = GridSpec(3, 1, figure=fig)

#     ax1 = fig.add_subplot(gs[:2, 0])
#     ax2 = fig.add_subplot(gs[2, 0])

#     f = 1.0e5
#     lw = [0.8, 0.8, 3.]
#     ls = ['-', '-', '-']
#     al = [1., 1., 0.7]
#     for i, (n, v) in enumerate(cl_models.items()):
#         kw = dict(label=n, lw=lw[i], ls=ls[i], alpha=al[i])
#         ax1.plot(v[0], f*v[1], **kw)
#         ax2.plot(el, v[1]/cl, **kw)

#     ax1.plot(el, f*cl, 'C0--', label='Mean of Mocks')
#     ax2.axhline(1.0, color='C0', ls='--')
#     ax2.fill_between(el, 1-cl_err/cl, 1+cl_err/cl, alpha=0.2)


#     ax1.legend(ncol=1)
#     ax1.set(xscale='log', ylabel=r'$10^{5}C_{\ell}$', yscale='log')
#     ax1.tick_params(labelbottom=False)
#     ax2.set(xscale='log', xlabel=r'$\ell$', ylabel='Ratio', xlim=ax1.get_xlim(), ylim=(0.89, 1.11))

#     fig.subplots_adjust(hspace=0.0, wspace=0.02)
#     fig.align_labels()

#     fig.savefig('figs/model_window.pdf', bbox_inches='tight')        
    
# def plot_mcmc_mocks():
#     stg = {'mult_bias_correction_order':0,'smooth_scale_2D':0.15, 'smooth_scale_1D':0.3, 'contours': [0.68, 0.95]}
#     mc_kw = dict(names=['fnl', 'b', 'n0'], 
#                  labels=['f_{NL}', 'b', '10^{7}n_{0}'], settings=stg) 

#     read_kw = dict(ndim=3, iscale=[2])
#     mc_kw2 = dict(names=['fnl', 'b', 'n0', 'b2', 'n02'], 
#                   labels=['f_{NL}', 'b1', '10^{7}n_{0}', 'b2', '10^{7}n_{0}2'], settings=stg)
#     read_kw2 = dict(ndim=5, iscale=[2, 4])
#     mc_kw3 = dict(names=['fnl', 'b', 'n0', 'b2', 'n02', 'b3', 'n03'], 
#                   labels=['f_{NL}', 'b1', '10^{7}n_{0}', 'b2', '10^{7}n_{0}2', 'b3', '10^{7}n_{0}3'], settings=stg)
#     read_kw3 = dict(ndim=7, iscale=[2, 4, 6])
#     fs = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_fullsky_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
#     bm = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_bmzls_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
#     nd = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_ndecals_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
#     sd = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_sdecals_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
#     bn = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_bmzlsndecals_noweight_steps10k_walkers50.npz', mc_kw=mc_kw2, read_kw=read_kw2)
#     joint = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_bmzlsndecalssdecals_noweight_steps10k_walkers50.npz', mc_kw=mc_kw3, read_kw=read_kw3)

#     stats = {}
#     stats['Full Sky'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_fullsky_noweight_steps10k_walkers50.npz', **read_kw)[0]
#     stats['F. Sky [BMzLS scaled]'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_fullskyscaled_noweight_steps10k_walkers50.npz', **read_kw)[0]
#     stats['BASS/MzLS'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_bmzls_noweight_steps10k_walkers50.npz', **read_kw)[0]
#     stats['DECaLS North'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_ndecals_noweight_steps10k_walkers50.npz', **read_kw)[0]
#     stats['DECaLS South'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_sdecals_noweight_steps10k_walkers50.npz', **read_kw)[0]

#     stats['NGC'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_ngc_noweight_steps10k_walkers50.npz', **read_kw)[0]
#     stats['Joint (BMzLS+DECaLS N)'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_bmzlsndecals_noweight_steps10k_walkers50.npz', **read_kw2)[0]

#     stats['Joint (DECaLS N+DECaLS S)'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_ndecalssdecals_noweight_steps10k_walkers50.npz', **read_kw2)[0]
#     stats['DESI'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_desi_noweight_steps10k_walkers50.npz', **read_kw)[0]
#     stats['Joint (BMzLS+DECaLS)'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_bmzlsndecalssdecals_noweight_steps10k_walkers50.npz', **read_kw3)[0]    

#     # Triangle plot
#     g = plots.get_single_plotter(width_inch=4*1.5)
#     g.settings.legend_fontsize = 14

#     g.plot_2d([bm, nd, sd, fs], 'fnl', 'b', filled=False)
#     g.add_x_marker(0)
#     g.add_y_marker(1.43)
#     g.add_legend(['BASS/MzLS', 'DECaLS North', 'DECaLS South', 'Full Sky'], colored_text=True, legend_loc='lower left')
#     #prsubplots(g.subplots[0, 0].get_xlim())
#     # g.subplots[0, 0].set_xticklabels([])
#     g.fig.align_labels()
#     g.fig.savefig('figs/fnl2dmocks_area.pdf', bbox_inches='tight')    
    
#     # Triangle plot
#     g = plots.get_single_plotter(width_inch=4*1.5)
#     g.settings.legend_fontsize = 14

#     g.plot_1d([bm, bn, joint], 'fnl', filled=False)
#     g.add_x_marker(0)
#     # g.add_y_marker(1.4262343145500318)
#     g.add_legend(['BASS/MzLS', 'BASS/MzLs+DECaLS North', 'BASS/MzLS+DECaLS (North+South)'], colored_text=True)

#     g.subplots[0, 0].set_xticks([-100, -75, -50, -25, 0, 25., 50.])
#     # g.subplots[0, 0].set_xticklabels([])
#     g.fig.align_labels()
#     g.fig.savefig('figs/fnl1dmocks_joint.pdf', bbox_inches='tight')    
    
#     pstats = pd.DataFrame(stats,
#                       index=['MAP [scipy]', 'MAP [chain]', 'Mean [chain]',
#                              'Median [chain]', '16th', '84th']).T
    
#     bm = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_po100_bmzls_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
#     nd = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_po100_ndecals_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
#     sd = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_po100_sdecals_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    

#     # Triangle plot
#     g = plots.get_single_plotter(width_inch=4*1.5)
#     g.settings.legend_fontsize = 14

#     g.plot_2d([bm, nd, sd], 'fnl', 'b', filled=False)
#     g.add_x_marker(100)
#     g.add_y_marker(1.43)
#     g.add_legend(['BASS/MzLS', 'DECaLS North', 'DECaLS South'], colored_text=True, legend_loc='lower left')
#     #prsubplots(g.subplots[0, 0].get_xlim())
#     # g.subplots[0, 0].set_xticklabels([])
#     g.fig.align_labels()
#     g.fig.savefig('figs/fnl2dmocks_po100.pdf', bbox_inches='tight')    
#     return pstats


# def plot_mcmc_mocks_wsys(region, fnltag='zero'):
#     stg = {'mult_bias_correction_order':0,'smooth_scale_2D':0.15,
#            'smooth_scale_1D':0.3, 'contours': [0.68, 0.95]}
#     mc_kw = dict(names=['fnl', 'b', 'n0'], 
#                  labels=['f_{NL}', 'b', '10^{7}n_{0}'], settings=stg) 

#     read_kw = dict(ndim=3, iscale=[2])



#     fnl_ = 0. if fnltag=='zero' else 100.
#     mcmcs = {}


        
#     names = ['Truth', 'No correction', 'All Maps'] # 'I: EBV', 'II: I+psfdepthg', 'III: II+nstar', 'IV: III+psfsize-g', 'V:IV+galdepthz', 
#     keys  = ['noweight',  'nn_all'] # 'nn_known1', 'nn_known2', 'nn_known3', 'nn_known4', 'nn_known5',

#     if region == 'bmzls':
#         names += ['I: EBV',   'II: I+psfdepthg', 'III: II+nstar', 'IV: III+psfsize-g', 'V:IV+galdepthz']
#         keys += ['nn_known1', 'nn_known2',       'nn_known3',     'nn_known4',         'nn_known5']
#     if region == 'ndecals':
#         names += ['V']
#         keys += ['nn_known5']
#     # keys = ['noweight', 'nn_known1', 'nn_known4', 
#     #         'nn_known5', 'nn_known6', 'nn_known7', 'nn_all']
#     # names = ['Truth', 'Contaminated', 'I: EBV', 'II: I+gdepthg', 'III: II+nstar',
#     #          'IV: III+psfsize-g', 'V:IV+gdepth-rz', 'All maps']

#     mcmcs['Truth'] = MCMC(f'/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_0_lrg_{fnltag}_{region}_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
#     for key in keys:
#         mcmcs[key] = MCMC(f'/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_1_lrg_{fnltag}_{region}_{key}_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)

#     y = []
#     for n,v in mcmcs.items():
#         y.append(v)



#     # Triangle plot
#     g = plots.get_single_plotter(width_inch=4*1.5)
#     g.settings.legend_fontsize = 14

#     g.plot_2d(y, 'fnl', 'b', filled=False)
#     g.add_x_marker(fnl_)
#     g.add_y_marker(1.43)
#     g.add_legend(names, colored_text=True)
#     #prsubplots(g.subplots[0, 0].get_xlim())
#     # g.subplots[0, 0].set_xticklabels([])
#     g.fig.align_labels()
#     # g.fig.savefig('figs/fnl2dmocks_mock1.pdf', bbox_inches='tight')    
    
#     # Triangle plot
#     g = plots.get_single_plotter(width_inch=4*1.5)
#     g.settings.legend_fontsize = 14

#     g.plot_1d(y, 'fnl', filled=False)
#     g.add_x_marker(fnl_)
#     # g.add_y_marker(1.43)
#     g.add_legend(names, colored_text=True)
#     #prsubplots(g.subplots[0, 0].get_xlim())
#     # g.subplots[0, 0].set_xticklabels([])
#     g.fig.align_labels()
#     # g.fig.savefig('figs/fnl2dmocks_mock1.pdf', bbox_inches='tight')        
    
    
#     stats = {}
#     for i, (n,v) in enumerate(mcmcs.items()):
#         stats[names[i]] = v.stats

#     pstats = pd.DataFrame(stats,
#                       index=['MAP [scipy]', 'MAP [chain]', 'Mean [chain]',
#                              'Median [chain]', '16th', '84th']).T

#     return pstats



# def add_scale(ax):
#     xx = ax.get_xticks()
#     dx = xx[1]-xx[0]
#     y = ax.get_yticks()[2]
#     ax.arrow(xx[-3], y, dx, 0, )
#     ax.annotate(f"{dx:.0f}", (xx[-3]+0.25*dx, y))
#     ax.set_xticklabels([])
    
    
# def plot_mcmc_dr9methods():
    
#     titles = {'noweight':'No Correction', 
#              'nn_known':'Conservative',
#              'nn_all':'Extreme'}

#     stg = {'mult_bias_correction_order':0,'smooth_scale_2D':0.15, 'smooth_scale_1D':0.3, 'contours': [0.68, 0.95]}
#     mc_kw = dict(names=['fnl', 'b', 'n0'], 
#                  labels=['f_{NL}', 'b', '10^{7}n_{0}'], settings=stg) 

#     read_kw = dict(ndim=3, iscale=[2])
#     mc_kw2 = dict(names=['fnl', 'b', 'n0', 'b2', 'n02'], 
#                   labels=['f_{NL}', 'b1', '10^{7}n_{0}', 'b2', '10^{7}n_{0}2'], settings=stg)
#     read_kw2 = dict(ndim=5, iscale=[2, 4])
#     mc_kw3 = dict(names=['fnl', 'b', 'n0', 'b2', 'n02', 'b3', 'n03'], 
#                   labels=['f_{NL}', 'b1', '10^{7}n_{0}', 'b2', '10^{7}n_{0}2', 'b3', '10^{7}n_{0}3'], settings=stg)
#     read_kw3 = dict(ndim=7, iscale=[2, 4, 6])

#     xlim = None

#     with PdfPages('figs/fnl2d_dr9_methods.pdf') as pdf:

#         for r in ['noweight', 'nn_known', 'nn_all']:

#             noweight = MCMC(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/mcmc_lrg_zero_bmzls_{r}_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
#             nnknown = MCMC(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/mcmc_lrg_zero_ndecals_{r}_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
#             nnall = MCMC(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/mcmc_lrg_zero_sdecals_{r}_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)


#             # Triangle plot
#             g = plots.get_single_plotter(width_inch=4*1.5)
#             g.settings.legend_fontsize = 14

#             g.plot_2d([noweight, nnknown, nnall], 'fnl', 'b', filled=True)
#             g.add_legend(['BASS/MzLS', 'DECaLS North', 'DECaLS South'], colored_text=True, title=titles[r])
#             g.subplots[0, 0].set_ylim((1.23, 1.6))
#             add_scale(g.subplots[0, 0])
#             g.fig.align_labels()
#             pdf.savefig(bbox_inches='tight')
            
# def plot_mcmc_dr9regions():
#     titles = {'bmzls':'BASS/MzLS', 
#              'ndecals':'DECaLS North',
#              'sdecals':'DECaLS South'}

#     stg = {'mult_bias_correction_order':0,'smooth_scale_2D':0.15, 'smooth_scale_1D':0.3, 'contours': [0.68, 0.95]}
#     mc_kw = dict(names=['fnl', 'b', 'n0'], 
#                  labels=['f_{NL}', 'b', '10^{7}n_{0}'], settings=stg) 

#     read_kw = dict(ndim=3, iscale=[2])
#     mc_kw2 = dict(names=['fnl', 'b', 'n0', 'b2', 'n02'], 
#                   labels=['f_{NL}', 'b1', '10^{7}n_{0}', 'b2', '10^{7}n_{0}2'], settings=stg)
#     read_kw2 = dict(ndim=5, iscale=[2, 4])
#     mc_kw3 = dict(names=['fnl', 'b', 'n0', 'b2', 'n02', 'b3', 'n03'], 
#                   labels=['f_{NL}', 'b1', '10^{7}n_{0}', 'b2', '10^{7}n_{0}2', 'b3', '10^{7}n_{0}3'], settings=stg)
#     read_kw3 = dict(ndim=7, iscale=[2, 4, 6])

#     xlim = None

#     with PdfPages('figs/fnl2d_dr9_regions.pdf') as pdf:

#         for r in ['bmzls', 'ndecals', 'sdecals']:

#             noweight = MCMC(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/mcmc_lrg_zero_{r}_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
#             nnknown = MCMC(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/mcmc_lrg_zero_{r}_nn_known_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
#             nnall = MCMC(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/mcmc_lrg_zero_{r}_nn_all_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)


#             # Triangle plot
#             g = plots.get_single_plotter(width_inch=4*1.5)
#             g.settings.legend_fontsize = 14
#             g.plot_2d([noweight, nnknown, nnall], 'fnl', 'b', filled=True)
#             g.add_legend(['No Correction', 'Conservative', 'Extreme'], colored_text=True, title=titles[r])
#             g.subplots[0, 0].set_ylim((1.23, 1.6))
#             add_scale(g.subplots[0, 0])
#             g.fig.align_labels()
#             pdf.savefig(bbox_inches='tight')
            
            
            
# def plot_mcmc_dr9joint():
#     path_ = '/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/'
#     titles = {'nn_known':'Conservative', 
#              'nn_all':'Extreme'}

#     stg = {'mult_bias_correction_order':0,'smooth_scale_2D':0.15, 'smooth_scale_1D':0.3, 'contours': [0.68, 0.95]}
#     mc_kw = dict(names=['fnl', 'b', 'n0'], 
#                  labels=['f_{NL}', 'b', '10^{7}n_{0}'], settings=stg) 

#     read_kw = dict(ndim=3, iscale=[2])
#     mc_kw2 = dict(names=['fnl', 'b', 'n0', 'b2', 'n02'], 
#                   labels=['f_{NL}', 'b1', '10^{7}n_{0}', 'b2', '10^{7}n_{0}2'], settings=stg)
#     read_kw2 = dict(ndim=5, iscale=[2, 4])
#     mc_kw3 = dict(names=['fnl', 'b', 'n0', 'b2', 'n02', 'b3', 'n03'], 
#                   labels=['f_{NL}', 'b1', '10^{7}n_{0}', 'b2', '10^{7}n_{0}2', 'b3', '10^{7}n_{0}3'], settings=stg)
#     read_kw3 = dict(ndim=7, iscale=[2, 4, 6])

#     xlim = None

#     with PdfPages('figs/fnl2d_dr9_joint.pdf') as pdf:
#         for r in ['nn_known', 'nn_all']:
#             noweight = MCMC(f'{path_}mcmc_lrg_zero_bmzls_{r}_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
#             nnknown = MCMC(f'{path_}mcmc_lrg_zero_bmzlsndecals_{r}_steps10k_walkers50.npz', mc_kw=mc_kw2, read_kw=read_kw2)
#             nnall = MCMC(f'{path_}mcmc_lrg_zero_bmzlsndecalssdecals_{r}_steps10k_walkers50.npz', mc_kw=mc_kw3, read_kw=read_kw3)

#             # Triangle plot
#             g = plots.get_single_plotter(width_inch=4*1.5)
#             g.settings.legend_fontsize = 14

#             g.plot_1d([noweight, nnknown, nnall], 'fnl', filled=False)
#             g.add_legend(['BASS/MzLS', '+ DECaLS North', '+ DECaLS (North + South)'], 
#                          colored_text=True, title=titles[r], legend_loc='lower left')
#             add_scale(g.subplots[0, 0])

#             g.fig.align_labels()
#             pdf.savefig(bbox_inches='tight')
            
# def plot_mcmc_dr9joint_bench():
#     path_ = '/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/'
#     titles = {'nn_known':'Conservative', 
#              'nn_all':'Extreme'}

#     stg = {'mult_bias_correction_order':0,'smooth_scale_2D':0.15, 'smooth_scale_1D':0.3, 'contours': [0.68, 0.95]}
#     mc_kw3 = dict(names=['fnl', 'b', 'n0', 'b2', 'n02', 'b3', 'n03'], 
#                   labels=['f_{NL}', 'b1', '10^{7}n_{0}', 'b2', '10^{7}n_{0}2', 'b3', '10^{7}n_{0}3'], settings=stg)
#     read_kw3 = dict(ndim=7, iscale=[2, 4, 6])

#     xlim = None

#     with PdfPages('figs/fnl2d_dr9_joint_bench.pdf') as pdf:
#         nn_ = MCMC(f'{path_}mcmc_lrg_zero_bmzlsndecalssdecals_nn_known_steps10k_walkers50.npz', mc_kw=mc_kw3, read_kw=read_kw3)
#         nn = MCMC(f'{path_}mcmc_lrg_zero_bmzlsndecalssdecals_nn_all_steps10k_walkers50.npz', mc_kw=mc_kw3, read_kw=read_kw3)

#         # Triangle plot
#         g = plots.get_single_plotter(width_inch=4*1.5)
#         g.settings.legend_fontsize = 14

#         g.plot_1d([nn, nn_], 'fnl', filled=False)
#         g.add_legend(['Extreme', 'Conservative'], 
#                      colored_text=True, legend_loc='lower left')
#         add_scale(g.subplots[0, 0])

#         g.fig.align_labels()
#         pdf.savefig(bbox_inches='tight')            
        
        
# def plot_nbar_mock(names, labels):
#     p_ = '/fs/ess/PHS0336/data/lognormal/v2/clustering/'
    
#     def errorbar(axi, *arrays, **kwargs):
#         _, y, ye = arrays
#         chi2 = (((y-1)/ye)**2).sum()
#         axi.errorbar(*arrays, **kwargs, label=fr'$\chi^{2}$ = {chi2:.1f}')


#     for jj in range(1):


#         fg, ax = plt.subplots(nrows=3, ncols=5, figsize=(25, 14), sharey=True)
#         ax = ax.flatten()
#         fg.subplots_adjust(wspace=0.0)
#         for k, r in enumerate(['bmzls']):#,'ndecals', 'sdecals']):
#             nbars = {}
#             for n in names:
#                 nbars[n] = np.load(f'{p_}nbarmock_1_1_lrg_zero_{r}_256_{n}.npy', allow_pickle=True)        

#             for i in range(13):
#                 for j, n in enumerate(nbars.keys()):
#                     #if j > jj:break
#                     errorbar(ax[i], nbars[n][i]['bin_avg'], nbars[n][i]['nnbar'], nbars[n][i]['nnbar_err'], capsize=4)
#                     if i==12:
#                         ax[13].text(0.2, 0.8-0.1*j-0.1*k, f'{labels[j]} {r}', 
#                                     transform=ax[13].transAxes, color='C%s'%j)
#                 ax[i].axhline(1.0, ls=':', lw=1)
#                 ax[i].set_xlabel(nbars[n][i]['sys'])
#                 ax[i].legend(ncol=2)
#                 ax[i].set_ylim(0.82, 1.18)

#         # plt.plot(nbars['nn'])
#         ax[5].set_ylabel('Mean Density')

#         # plt.plot(nbars['noweight'])
#         fg.show()        
        
    
# def plot_clmocks_wsys():
#     survey = 'bmzls'
#     p = '/fs/ess/PHS0336/data/lognormal/v2/clustering/'
#     def add_plot(ax1, ax2, fnl, iscont, method, **kwargs):
#         file_out = f'{p}clmock_{iscont}_lrg_{fnl}_{survey}_256_{method}_mean.npz'
#         cl = np.load(file_out)
#         ax1.plot(cl['el_bin'], cl['cl'], **kwargs)

#         file_out = f'{p}clmock_0_lrg_{fnl}_{survey}_256_noweight_mean.npz'
#         cl_ = np.load(file_out)
#         ax2.plot(cl['el_bin'], cl['cl']/cl_['cl'], **kwargs)


#     labels = ['No correction', 'I: EBV', 'II: I+psfdepthg', 
#               'III: II+nstar', 'IV: III+psfsize-g', 'V:IV+galdepthz', 'All Maps']
#     fig = plt.figure(figsize=(16, 8), constrained_layout=False)
#     gs = GridSpec(3, 2, figure=fig)

#     ax1 = fig.add_subplot(gs[:2, 0])
#     ax2 = fig.add_subplot(gs[2, 0])

#     ax3 = fig.add_subplot(gs[:2, 1])
#     ax4 = fig.add_subplot(gs[2, 1])


#     # fnl=0
#     add_plot(ax1, ax2, 'zero', 0, 'noweight',  label='Truth',   marker='', alpha=0.5, lw=4)
#     add_plot(ax1, ax2, 'zero', 1, 'noweight',  label='Cont',    marker='x', alpha=0.8)
#     add_plot(ax1, ax2, 'zero', 1, 'nn_known1', label=labels[1], marker='+', alpha=0.8)
#     add_plot(ax1, ax2, 'zero', 1, 'nn_known2', label=labels[2], marker='s', alpha=0.8)
#     add_plot(ax1, ax2, 'zero', 1, 'nn_known3', label=labels[3], marker='<', alpha=0.8)
#     add_plot(ax1, ax2, 'zero', 1, 'nn_known4', label=labels[4], marker='.', alpha=0.8)
#     add_plot(ax1, ax2, 'zero', 1, 'nn_known5', label=labels[5], marker='*', alpha=0.8)
#     add_plot(ax1, ax2, 'zero', 1, 'nn_all',    label=labels[6], marker='>', alpha=0.8)

#     # fnl=100
#     add_plot(ax3, ax4, 'po100', 0, 'noweight',  label='Truth',   marker='', alpha=0.5, lw=4)
#     add_plot(ax3, ax4, 'po100', 1, 'noweight',  label='Cont',    marker='x', alpha=0.8)
#     add_plot(ax3, ax4, 'po100', 1, 'nn_known1', label=labels[1], marker='+', alpha=0.8)
#     add_plot(ax3, ax4, 'po100', 1, 'nn_known2', label=labels[2], marker='s', alpha=0.8)
#     add_plot(ax3, ax4, 'po100', 1, 'nn_known3', label=labels[3], marker='<', alpha=0.8)
#     add_plot(ax3, ax4, 'po100', 1, 'nn_known4', label=labels[4], marker='.', alpha=0.8)
#     add_plot(ax3, ax4, 'po100', 1, 'nn_known5', label=labels[5], marker='*', alpha=0.8)
#     add_plot(ax3, ax4, 'po100', 1, 'nn_all',    label=labels[6], marker='>', alpha=0.8)


#     for ax, ax_ in [(ax1, ax2), (ax3, ax4)]:
#         ax.set(xscale='log', yscale='log',  xlim=(3, 310), ylim=(1.1e-6, 4.2e-4))
#         ax_.set(xscale='log', xlabel=r'$\ell$', xlim=(3, 310), ylim=(-0.1, 2.1))
#         ax.set_xticklabels([])

#     ax1.grid(True, which='both', alpha=0.2)
#     ax3.grid(True, which='both', alpha=0.2)
#     ax1.set_ylabel(r'$C_{\ell}$')
#     ax2.set_ylabel(r'$C_{X}/C_{\rm Truth}$')

#     ax1.legend(loc=1, ncol=2, title=r'$f_{\rm NL}=0$')
#     ax3.legend(loc=1, ncol=2, title=r'$f_{\rm NL}=100$')


#     fig.subplots_adjust(hspace=0.0, wspace=0.1)
#     fig.align_labels()

#     fig.savefig('./figs/cl_mocks.png', dpi=300)    
    
        
# ## Mean Density Test
# def chi2_fn(y, invcov):
#     return np.dot(y, np.dot(invcov, y))    

# def get_chi2t(path, invcov):
#     nnbar_ = read_nnbar(path)
#     return chi2_fn(nnbar_, invcov)

# def get_chi2t_mocks(cap, iscont, fnl):
#     path_ = '/fs/ess/PHS0336/data/lognormal/v2/clustering/'
#     mocks = glob(f'{path_}nbarmock_{iscont}_*_lrg_{fnl}_{cap}_256_noweight.npy')
#     mocks.sort()
#     mocks = mocks[::-1]
#     print('len(nbars):', len(mocks), cap)
#     print(mocks[0])
#     nmocks = len(mocks)
#     err_tot = []
#     for j, fn in enumerate(mocks):
#         err_j = read_nnbar(fn)
#         err_tot.append(err_j)            
#     err_tot = np.array(err_tot)
#     print(err_tot.shape)

#     nbins = err_tot.shape[1]
#     hartlapf = (nmocks-1. - 1.) / (nmocks-1. - nbins - 2.)
#     indices = [i for i in range(nmocks)]
#     chi2s = []
#     for i in range(nmocks):
#         indices_ = indices.copy()    
#         indices_.pop(i)
#         nbar_ = err_tot[i, :]
#         err_ = err_tot[indices_, :]    
#         covmax_ = np.cov(err_, rowvar=False)
#         invcov_ = np.linalg.inv(covmax_*hartlapf)
#         chi2_ = chi2_fn(nbar_, invcov_)
#         if i==0:print(chi2_)
#         chi2s.append(chi2_)       

#     print(nmocks)
#     covmax_ = np.cov(err_tot, rowvar=False)
#     hartlapf = (nmocks - 1.) / (nmocks - nbins - 2.)
#     invcov_ = np.linalg.inv(covmax_*hartlapf)

#     return np.array(chi2s), invcov_    


# def plot_chi2hist_mock(names, labels, imock=1, region='bmzls', fnltag='zero'):
#     if os.path.exists(f'./tmp_nbarmock_chi2_iscont0_{region}.npz'):
#         chi2ss = np.load(f'./tmp_nbarmock_chi2_iscont0_{region}.npz', allow_pickle=True)
#         print(chi2ss.files)
#         chi2s = chi2ss['0']
#         chi2f = chi2ss['100']
#         invcov = chi2s[1] if fnltag=='zero' else chi2f[1]
#     else:
#         chi2s = get_chi2t_mocks(region, 0, 'zero')
#         chi2f = get_chi2t_mocks(region, 0, 'po100')
#         invcov = chi2s[1] if fnltag=='zero' else chi2f[1]
#         np.savez(f'./tmp_nbarmock_chi2_iscont0_{region}.npz', **{'0':chi2s, '100':chi2f})

#     chi2c = {}
#     p_ = '/fs/ess/PHS0336/data/lognormal/v2/clustering/'
#     print(f'imock: {imock}')
#     for i, n in enumerate(names):
#         chi2c[names[i]] = get_chi2t(f'{p_}nbarmock_1_{imock}_lrg_{fnltag}_{region}_256_{n}.npy', invcov)

#     plt.figure()    
#     plt.hist(chi2s[0], histtype='step', range=(50., 200.), bins=25)
#     plt.hist(chi2f[0], histtype='step', range=(50., 200.), bins=25)
#     for i, (n,v) in enumerate(chi2c.items()):
#         plt.axvline(v, label=labels[i], color='C%d'%i)
#     plt.legend(loc=(1., 0.2))
#     plt.text(120, 150, r'Mocks $f_{\rm NL}$=0')
#     plt.text(120, 135, r'Mocks $f_{\rm NL}$=100', color='C1')
#     plt.xlabel(r'Mean Density Contrast $\chi^{2}$')
#     plt.xscale('log')
#     plt.show()
#     # plt.savefig('figs/nbar_chi2_mock1.pdf', bbox_inches='tight')
#     # plt.yscale('log')        
    
    
# def plot_chi2hist_mocks(name, label, region='bmzls', fnltag='zero'):
#     if os.path.exists(f'./tmp_nbarmock_chi2_iscont0_{region}.npz'):
#         chi2ss = np.load(f'./tmp_nbarmock_chi2_iscont0_{region}.npz', allow_pickle=True)
#         print(chi2ss.files)
#         chi2s = chi2ss['0']
#         chi2f = chi2ss['100']
#         invcov = chi2s[1] if fnltag=='zero' else chi2f[1]
#     else:
#         chi2s = get_chi2t_mocks(region, 0, 'zero')
#         chi2f = get_chi2t_mocks(region, 0, 'po100')
#         invcov = chi2s[1] if fnltag=='zero' else chi2f[1]
#         np.savez(f'./tmp_nbarmock_chi2_iscont0_{region}.npz', **{'0':chi2s, '100':chi2f})

#     chi2c = []
#     p_ = '/fs/ess/PHS0336/data/lognormal/v2/clustering/'
#     for imock in range(1, 1001):
#         chi2c.append(get_chi2t(f'{p_}nbarmock_1_{imock}_lrg_{fnltag}_{region}_256_{name}.npy', invcov))

#     plt.figure()    
#     plt.hist(chi2s[0], histtype='step', range=(50., 200.), bins=25)
#     plt.hist(chi2f[0], histtype='step', range=(50., 200.), bins=25)
#     plt.hist(chi2c, histtype='step', label=label)
#     plt.legend(loc=(1., 0.2))
#     plt.text(120, 150, r'Mocks $f_{\rm NL}$=0')
#     plt.text(120, 135, r'Mocks $f_{\rm NL}$=100', color='C1')
#     plt.xlabel(r'Mean Density Contrast $\chi^{2}$')
#     plt.xscale('log')
#     plt.show()
#     # plt.savefig('figs/nbar_chi2_mock1.pdf', bbox_inches='tight')
#     # plt.yscale('log')            
    
    

# def read_clx(fn, bins=None):

#     cl = np.load(fn, allow_pickle=True).item()
    
#     cl_cross = []
#     cl_ss = []

#     for i in range(len(cl['cl_sg'])):    
#         __, cl_sg_ = histogram_cell(cl['cl_sg'][i]['l'], cl['cl_sg'][i]['cl'], bins=bins)
#         __, cl_ss_ = histogram_cell(cl['cl_ss'][i]['l'], cl['cl_ss'][i]['cl'], bins=bins)

#         cl_ss.append(cl_ss_)
#         cl_cross.append(cl_sg_**2/cl_ss_)    

#     return np.array(cl_cross).flatten()


# def read_clxmocks(list_clx, bins=None):
    
#     err_mat = []    
#     for i, clx_i in enumerate(list_clx):
        
#         err_i  = read_clx(clx_i, bins=bins)
#         err_mat.append(err_i)
        
#         if (i % (len(list_clx)//10)) == 0:
#             print(f'{i}/{len(list_clx)}')

#     err_mat = np.array(err_mat)
#     print(err_mat.shape)
#     return err_mat


# def make_chi2clx(fnl='zero'):
    
#     p = '/fs/ess/PHS0336/data/lognormal/v2/clustering/'
#     err_truth = read_clxmocks(glob(f'{p}clmock_0_*_lrg_{fnl}_bmzls_256_noweight.npy'),
#                             ell_edges[:4])
#     err_cont = read_clxmocks(glob(f'{p}clmock_1_*_lrg_{fnl}_bmzls_256_noweight.npy'),
#                             ell_edges[:4])
#     err_known5 = read_clxmocks(glob(f'{p}clmock_1_*_lrg_{fnl}_bmzls_256_nn_known5.npy'),
#                             ell_edges[:4])

#     chi2_truth = get_chi2pdf(err_truth)
#     chi2_cont = get_chi2pdf(err_cont)
#     chi2_known5 = get_chi2pdf(err_known5)

#     for chi2_i in [chi2_truth, chi2_cont, chi2_known5]:
#         plt.hist(chi2_i, histtype='step', range=(0, 600), bins=60)    
#     plt.show()