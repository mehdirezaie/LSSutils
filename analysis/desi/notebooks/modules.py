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

import warnings
warnings.filterwarnings("ignore")

dv.setup_color()

gratio = 1.3


def plot_nz():
    nz = np.loadtxt('/fs/ess/PHS0336/data/rongpu/sv3_lrg_dndz_denali.txt')

    x1 =  nz[:, 0]
    y1 = nz[:, 2]/(nz[:, 2].sum()*np.diff(nz[:, 0])[0])
    
    fg, ax = plt.subplots()
    ax.step(x1, y1, where='post', lw=1)#, label='dN/dz')
    ax.set(xlim=(-0.05, 1.45), xlabel='z', ylabel='Normalized dN/dz')
    ax.text(0.25, 0.7, 'dN/dz', color='C0', transform=ax.transAxes)

    ax1 = ax.twinx()
    x2 = np.linspace(0.1, 1.38, num=200)
    y2 = 1.4*bias_model_lrg(x2)
    
    ax1.plot(x2, y2, 'C1--', lw=3, alpha=0.5, zorder=10)#, label='b(z)')
    ax1.text(0.72, 0.47, 'b(z)$\propto$ D$^{-1}$(z)', color='C1', transform=ax1.transAxes)
    ax1.set_ylabel('b(z)')
    ax1.set_ylim((1.3, 3.1))

    #ax1.legend(loc='upper right')
    #ax.legend(loc='upper left')
    np.savetxt(f'/users/PHS0336/medirz90/github/dimagfnl/figures/fig1_nz_lrg1.txt', np.column_stack([x1, y1]))
    np.savetxt(f'/users/PHS0336/medirz90/github/dimagfnl/figures/fig1_nz_lrg2.txt', np.column_stack([x2, y2]))               
    fg.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/nz_lrg.pdf', bbox_inches='tight') 


# --- helper functions
def print_stats(stats):
    for s, v in stats.items():
        msg = r"{0:40s}& ${1:6.0f}$& ${2:6.0f}$& ${3:6.0f}<\fnl<{4:6.0f}$& ${5:6.0f}<\fnl<{6:6.0f}$ & {7:6.1f}\\"\
                .format(s, v[0], v[1], v[2], v[3], v[4], v[5], -v[6])
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
        el_b, cl_sg_ = ut.histogram_cell(cl['cl_sg'][i]['l'], cl['cl_sg'][i]['cl'], bins=bins)
        __, cl_ss_ = ut.histogram_cell(cl['cl_ss'][i]['l'], cl['cl_ss'][i]['cl'], bins=bins)
        cl_ss.append(cl_ss_)
        cl_cross.append(cl_sg_**2/cl_ss_)
    return el_b, np.array(cl_cross).flatten()


def read_clxmocks(list_clx, bins=None):
    err_mat = []    
    for i, clx_i in enumerate(list_clx):
        err_i  = read_clx(clx_i, bins=bins)[1]
        err_mat.append(err_i)
        if (i % (len(list_clx)//10)) == 0:print(f'{i}/{len(list_clx)}')
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


def plot_xmaps():
    ydata = []
    
    
    d = ft.read('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_desi_256.fits')

    desi = ft.read('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_desi_256.fits')
    ng  = ut.make_hp(256, desi['hpix'], desi['label']/(desi['fracgood']*hp.nside2pixarea(256, True)), np.inf)
    print(np.mean(ng[desi['hpix']]))
    
    fig = plt.figure(figsize=(6, 7))
    ax = []
    ax.append(fig.add_axes([0., 1.0, 1., 1], projection='mollweide'))
    ax.append(fig.add_axes([0., 0.6, 1., 1],  projection='mollweide'))
    ax.append(fig.add_axes([0., 0.2, 1., 1],  projection='mollweide'))
    ax.append(fig.add_axes([1., 1.0, 1., 1], projection='mollweide'))
    ax.append(fig.add_axes([1., 0.6, 1., 1], projection='mollweide'))
    ax.append(fig.add_axes([1., 0.2, 1., 1], projection='mollweide'))
    ax.append(fig.add_axes([2., 1.0, 1., 1], projection='mollweide'))
    ax.append(fig.add_axes([2., 0.6, 1., 1], projection='mollweide'))
    ax.append(fig.add_axes([2., 0.2, 1., 1], projection='mollweide'))
    ax_map = fig.add_axes([0.5, 1.2, 2., 2.], projection='mollweide')


    names = [r'EBV', r'nStar']+[fr'depth$_{b}$' for b in ['g', 'r', 'z', '{w1}']]\
            + [fr'psfsize$_{b}$' for b in ['g', 'r', 'z']]

    for i, map_i in enumerate(d['features'].T):
        mi = ut.make_hp(256, d['hpix'], map_i, True)
        cmin, cmax = np.percentile(mi[~np.isnan(mi)], [2.5, 97.5])
        dv.mollview(mi, cmin, cmax, names[i], figax=[fig, ax[i]], 
                    cmap='jet', colorbar=False, galaxy=False)
        ax[i].text(0.55, 0.2, names[i], 
                   transform=ax[i].transAxes, alpha=0.8, fontsize=20)
        
        ydata.append(mi)
        
    dv.mollview(ng, 400, 1200, r'LRG Density [deg$^{-2}$]', 
            cmap=dv.mycolor(), colorbar=True, galaxy=False, 
                figax=[fig, ax_map], cax_axes=[1.6, 1.95, 0.6, 0.02]) # 'YlOrRd_r',       
    ydata.append(ng)
    
    for ax_ in ax:    
        ax_.xaxis.set_ticklabels([])
        ax_.yaxis.set_ticklabels([])
        ax_.xaxis.set_ticks([])
        ax_.yaxis.set_ticks([])          
    np.savetxt(f'/users/PHS0336/medirz90/github/dimagfnl/figures/fig2_dr9data.txt', np.column_stack(ydata))
    fig.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/dr9data.pdf', bbox_inches='tight') 
    
    
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
        
    colors = ['C1', 'C2', 'C3']
    names = [r'EBV', r'nStar']+[fr'depth$_{b}$' for b in ['g', 'r', 'z', '{w1}']]\
            + [fr'psfsize$_{b}$' for b in ['g', 'r', 'z']]

    ydata = []
    
    for i, (region, name) in enumerate(zip(['bmzls', 'ndecalsc', 'sdecalsc'],
                                          ['BASS+MzLS', 'DECaLS North', 'DECaLS South'])):
        p_, er_ = pccs[region]

        pcc_ = p_[0]
        err_ = er_[:, 0, :]
        print(err_.shape, len(pcc_))
        pcc_min, pcc_max = np.percentile(err_, [2.5, 97.5], axis=0)

        x = np.arange(len(pcc_))+i*0.2
        plt.bar(x, pcc_, width=0.2, alpha=0.6, color=colors[i], label=name)
        plt.plot(x, pcc_min, ls='-', lw=1, color=colors[i], alpha=0.5)
        plt.plot(x, pcc_max, ls='-', lw=1, color=colors[i], alpha=0.5)
        ydata.append(pcc_)
        ydata.append(pcc_min)
        ydata.append(pcc_max)

    plt.xticks(x, names, rotation=90)
    plt.ylabel('Pearson (LRG, Imaging)')
    lgn = plt.legend()
    for i, txt in enumerate(lgn.get_texts()):
        txt.set_color('C%d'%(i+1))
        
    np.savetxt(f'/users/PHS0336/medirz90/github/dimagfnl/figures/fig3_pcc_1.txt', np.column_stack(ydata))
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
    np.savetxt(f'/users/PHS0336/medirz90/github/dimagfnl/figures/fig3_pcc_2.txt', corr_cf)
    plt.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/pccx.pdf', bbox_inches='tight')
    plt.show()
    
    
def plot_clxtest():
    names = [r'EBV', r'nStar']+[fr'depth$_{b}$' for b in ['g', 'r', 'z', '{w1}']]\
            + [fr'psfsize$_{b}$' for b in ['g', 'r', 'z']]    
    ell_edges = ut.ell_edges[:10]
    print(f'ell edges: {ell_edges}')

    p = '/fs/ess/PHS0336/data/lognormal/v3/clustering/'
    err_0 = read_clxmocks(glob(f'{p}clmock_0_*_lrg_zero_desic_256_noweight.npy'), ell_edges)
    err_100 = read_clxmocks(glob(f'{p}clmock_0_*_lrg_po100_desic_256_noweight.npy'), ell_edges)

    chi2s = {}
    chi2s['fNL=0'] = ut.get_chi2pdf(err_0)
    chi2s['fNL=76.92'] = ut.get_chi2pdf(err_100)

    
    d_ = '/fs/ess/PHS0336/data/rongpu/imaging_sys/clustering/0.57.0/'
    el_b, err_dr9  = read_clx(f'{d_}cl_lrg_desic_256_noweight.npy', ell_edges)
    err_dr9all     = read_clx(f'{d_}cl_lrg_desic_256_linp_all.npy', ell_edges)[1]
    err_dr9known   = read_clx(f'{d_}cl_lrg_desic_256_linp_known.npy', ell_edges)[1]
    err_dr9known1  = read_clx(f'{d_}cl_lrg_desic_256_linp_known1.npy', ell_edges)[1]
    err_dr9nknown1 = read_clx(f'{d_}cl_lrg_desic_256_dnnp_known1.npy', ell_edges)[1]
    err_dr9nknownp = read_clx(f'{d_}cl_lrg_desic_256_dnnp_knownp.npy', ell_edges)[1]    

    err_0m = np.percentile(err_0, [97.5, ], axis=0).flatten()
    err_100m = np.percentile(err_100, [97.5, ], axis=0).flatten()
    icov, cov_0 = ut.get_inv(err_0, return_cov=True)
    cov_100 = ut.get_inv(err_100, return_cov=True)[1]

    chi2_dr9 = ut.chi2_fn(err_dr9, icov)
    chi2_dr9all = ut.chi2_fn(err_dr9all, icov)
    chi2_dr9known = ut.chi2_fn(err_dr9known, icov)
    chi2_dr9known1 = ut.chi2_fn(err_dr9known1, icov)
    chi2_dr9nknown1 = ut.chi2_fn(err_dr9nknown1, icov)
    chi2_dr9nknownp = ut.chi2_fn(err_dr9nknownp, icov)

    
    labels = ['No Weight', 'Linear Two Maps', 'Linear Three Maps', 'Linear Eight Maps', 
              'Nonlinear Three Maps', 'Nonlinear Four Maps']    
    
    fg, ax = plt.subplots(ncols=3, nrows=3, 
                             figsize=(12, 9), sharey=True, sharex=True)
    ax = ax.flatten()
    fg.subplots_adjust(wspace=0.02, hspace=0.02)

    ydata = []

    
    for j, err_j in enumerate([err_dr9, err_dr9known, err_dr9known1, err_dr9all,
                               err_dr9nknown1, err_dr9nknownp]):
        ydata.append(err_j)
        for i, ax_i in enumerate(ax):
            ax[i].semilogy(el_b, err_j[i*9:(i+1)*9], label=labels[j])


    for i, ax_i in enumerate(ax):
        ax_i.fill_between(el_b, err_100m[i*9:(i+1)*9], 
                           color='k', alpha=0.05)    
        ax_i.fill_between(el_b, err_0m[i*9:(i+1)*9], 
                           color='k', alpha=0.10)   
        ax_i.text(0.5, 0.85, f'LRG x {names[i]}', transform=ax_i.transAxes)
        if i > 5: ax_i.set(xlabel=r'$\ell$')

    lgnd = ax[0].legend(ncol=3,frameon=False,
                     bbox_to_anchor=(0, 1.05, 3, 0.4), loc="lower left",
                    mode="expand", borderaxespad=0)

    for i, lgn_tx in enumerate(lgnd.get_texts()):
        lgn_tx.set_color('C%d'%i)
    
    ax[0].set_ylim(9.5e-9, 1.2e-4)
    #ax[3].set_ylabel(r'$C_{s, g}^{2}/C_{s,s}$')        
    ax[3].set_ylabel(r'$\tilde{C}_{x, \ell}$') 
    np.savetxt('/users/PHS0336/medirz90/github/dimagfnl/figures/fig6_clx_mocks.txt', np.column_stack(ydata))
    fg.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/clx_mocks.pdf', bbox_inches='tight')    
    plt.show()
    
    
    xlabel = r'Cross Spectrum $\chi^{2}$'
    xlim1 = (-10, 450)
    xlim2 = (19970, 20110)
    ylim = (0., 120.)

    fig = plt.figure()
    fig.subplots_adjust(wspace=0.03)
    gs  = GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ax1.tick_params(direction='in', axis='both', right=False)
    ax2.tick_params(direction='in', axis='both', which='both', left=False, right=True)

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.set(xlim=xlim1, ylim=ylim, xlabel=xlabel)
    ax2.set(xlim=xlim2, ylim=ylim)#, xticks=xticks2)
    ax2.set_yticklabels([])
    d = 0.01
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d/2, 1 + d/2), (-d, +d), **kwargs)  # bottom-left diagonal
    ax1.plot((1 - d/2, 1 + d/2), (1-d, 1+d), **kwargs)  # bottom-left diagonal
    kwargs.update(transform=ax2.transAxes)            # switch to the bottom axes
    ax2.plot((-d/2, +d/2), (-d, +d), **kwargs)        # top-left diagonal
    ax2.plot((-d/2, +d/2), (1-d, 1+d), **kwargs)        # top-left diagonal

    ls = ['-', '--']
    names = ['$f_\mathrm{NL}$=0', '$f_\mathrm{NL}$=76.9']
    for i, (__, chi2_i) in enumerate(chi2s.items()):
        print(np.max(chi2_i), np.min(chi2_i))
        ax1.hist(chi2_i, histtype='step', bins=58, 
                 ls=ls[i], label=names[i], range=(0, 550.), color='grey') 
    lgn = ax1.legend(title='Clean Mocks', frameon=True)
    for i,txt in enumerate(lgn.get_texts()):
        txt.set_color('grey')

    ax2.axvline(chi2_dr9, lw=1)
    ax2.annotate('No Weight', (chi2_dr9+7, 20), rotation=90, fontsize=13)

    ax1.axvline(chi2_dr9nknown1, lw=1, color='C4', ls='--')
    ax1.annotate('Nonlinear Three Maps', (chi2_dr9nknown1-20, 8), rotation=90, fontsize=13, color='C4')

    ax1.axvline(chi2_dr9known1, lw=1, color='C2', ls='-.')
    ax1.annotate('Linear Three Maps', (chi2_dr9known1+7, 20), rotation=90, fontsize=13, color='C2')

    ax1.axvline(chi2_dr9known, lw=1, color='C1', ls='--')
    ax1.annotate('Linear Two Maps', (chi2_dr9known+7, 20), rotation=90, fontsize=13, color='C1')

    
    for chi_i in [chi2_dr9, chi2_dr9known, chi2_dr9known1, chi2_dr9all, chi2_dr9nknown1, chi2_dr9nknownp]:
        is_gt = np.array(chi2s['fNL=0']) > chi_i
        print('p-value:', is_gt.mean())
    np.savetxt('/users/PHS0336/medirz90/github/dimagfnl/figures/fig8_chi2test_1.txt', 
              [chi2_dr9, chi2_dr9known, chi2_dr9known1, chi2_dr9all, chi2_dr9nknown1, chi2_dr9nknownp])
    np.savetxt('/users/PHS0336/medirz90/github/dimagfnl/figures/fig8_chi2test_2.txt', [chi2s['fNL=0'], chi2s['fNL=76.92']])
    fig.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/chi2test.pdf', bbox_inches='tight')         


def test_chi2lmax():
    chi2_mocks = []
    chi2_data  = []
    ell_maxes = []
    p_ = '/fs/ess/PHS0336/data/rongpu/imaging_sys/clustering/0.57.0/'
    p = '/fs/ess/PHS0336/data/lognormal/v3/clustering/'
    
    for m in [10, 12, 14, 16, 18]:
        
        ell_edges = ut.ell_edges[:m]
        ell_maxes.append(ell_edges.max())        
        
        err_0 = read_clxmocks(glob(f'{p}clmock_0_*_lrg_zero_desic_256_noweight.npy'), bins=ell_edges)
        err_dr9known1 = read_clx(f'{p_}cl_lrg_desic_256_linp_known1.npy', ell_edges)[1]
        err_dr9nknown1 = read_clx(f'{p_}cl_lrg_desic_256_dnnp_known1.npy', ell_edges)[1]
        
        chi2_mocks.append(ut.get_chi2pdf(err_0))
        icov, cov_0 = ut.get_inv(err_0, return_cov=True)
        chi2_data.append(ut.chi2_fn(err_dr9known1, icov))
        chi2_data.append(ut.chi2_fn(err_dr9nknown1, icov))     
        
    chi2_mocks = np.array(chi2_mocks)
    chi2_data = np.array(chi2_data).reshape(-1, 2)
    chi2_min, chi2_16, chi2_median, chi2_84, chi2_max = np.percentile(chi2_mocks, [2.5, 16, 50, 84, 97.5], axis=1)
    
    plt.fill_between(ell_maxes, chi2_min, chi2_max, alpha=0.03, color='C0')
    plt.fill_between(ell_maxes, chi2_16, chi2_84, alpha=0.06, color='C0')
    
    plt.plot(ell_maxes, chi2_median, label='Mocks Median', lw=1, color='C0')
    plt.scatter(ell_maxes, chi2_data[:, 0],  marker='x', alpha=0.5, color='C2')
    plt.scatter(ell_maxes, chi2_data[:, 1], marker='s', alpha=1.0, color='C4')
    lgn = plt.legend(loc=4)

    plt.text(28, 310, 'Linear Three Maps', color='C2', fontsize=13)
    plt.text(40, 175, 'Nonlinear Three Maps', color='C4', fontsize=13)
    plt.text(70, 100, r'Mocks 68\% (95\%)', color='C0', alpha=0.4, fontsize=13)
    plt.xlabel(r'$\ell_{\rm max}$')
    plt.ylabel(r'Cross Spectrum $\chi^{2}$')
    
    np.savetxt('/users/PHS0336/medirz90/github/dimagfnl/figures/figA1_chi2lmax.txt',
              [ell_maxes, chi2_min, chi2_16, chi2_median, chi2_84, chi2_max, chi2_data[:, 0], chi2_data[:, 1]])
    plt.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/chi2lmax.pdf', bbox_inches='tight')    
        
    
    
def plot_nbartest():
    names = [r'EBV [mag]', r'nStar [deg$^{-2}$]']+[fr'depth$_{b}$ [mag]' for b in ['g', 'r', 'z', '{w1}']]\
            + [fr'psfsize$_{b}$ [arcsec]' for b in ['g', 'r', 'z']]    
    
    err_0 = read_nbmocks(glob('/fs/ess/PHS0336/data/lognormal/v3/clustering/nbarmock_0_*_lrg_zero_desic_256_noweight.npy'))
    err_100 = read_nbmocks(glob('/fs/ess/PHS0336/data/lognormal/v3/clustering/nbarmock_0_*_lrg_po100_desic_256_noweight.npy'))
    icov, cov_0 = ut.get_inv(err_0, return_cov=True)
    cov_100 = ut.get_inv(err_100, return_cov=True)[1]
    
    d_ = '/fs/ess/PHS0336/data/rongpu/imaging_sys/clustering/0.57.0/'    
    xbins, err_dr9 = read_nnbar(f'{d_}nbar_lrg_desic_256_noweight.npy', return_bins=True)
    err_dr9all     = read_nnbar(f'{d_}nbar_lrg_desic_256_linp_all.npy')
    err_dr9known   = read_nnbar(f'{d_}nbar_lrg_desic_256_linp_known.npy')
    err_dr9known1  = read_nnbar(f'{d_}nbar_lrg_desic_256_linp_known1.npy')
    err_dr9nknown1 = read_nnbar(f'{d_}nbar_lrg_desic_256_dnnp_known1.npy') 
    err_dr9nknownp = read_nnbar(f'{d_}nbar_lrg_desic_256_dnnp_knownp.npy') 
    
    chi2_dr9 = ut.chi2_fn(err_dr9, icov)
    chi2_dr9all = ut.chi2_fn(err_dr9all, icov)
    chi2_dr9known = ut.chi2_fn(err_dr9known, icov)
    chi2_dr9known1 = ut.chi2_fn(err_dr9known1, icov)
    chi2_dr9nknown1 = ut.chi2_fn(err_dr9nknown1, icov)
    chi2_dr9nknownp = ut.chi2_fn(err_dr9nknownp, icov)
    
    err_0m = np.std(err_0,  axis=0)
    err_100m = np.std(err_100, axis=0)
    
    labels = ['No Weight', 'Linear Two Maps', 'Linear Three Maps', 'Linear Eight Maps',
              'Nonlinear Three Maps', 'Nonlinear Four Maps']
    
    fg, ax = plt.subplots(ncols=3, nrows=3, figsize=(12, 9), sharey='row')
    ax = ax.flatten()
    fg.subplots_adjust(wspace=0.02, hspace=0.35)

    for j, err_j in enumerate([err_dr9, err_dr9known, err_dr9known1, err_dr9all,
                               err_dr9nknown1, err_dr9nknownp]):
        for i, ax_i in enumerate(ax):
            ax[i].plot(xbins[i], err_j[i*8:(i+1)*8], label=labels[j])


    for i, ax_i in enumerate(ax):
        ax_i.fill_between(xbins[i], -err_100m[i*8:(i+1)*8], err_100m[i*8:(i+1)*8], 
                           color='k', alpha=0.05)    
        ax_i.fill_between(xbins[i], -err_0m[i*8:(i+1)*8], err_0m[i*8:(i+1)*8], 
                           color='k', alpha=0.10)   
        ax_i.set_xlabel(names[i])
        ax[3].set_ylabel('Mean Density Contrast')

    lgnd = ax[0].legend(ncol=3,frameon=False,
                     bbox_to_anchor=(0, 1.05, 3, 0.4), loc="lower left",
                    mode="expand", borderaxespad=0)

    for i, lgn_tx in enumerate(lgnd.get_texts()):
        lgn_tx.set_color('C%d'%i)
    fg.align_ylabels()
    np.savetxt('/users/PHS0336/medirz90/github/dimagfnl/figures/fig7_nbar_mocks_1.txt', 
              [err_dr9, err_dr9known, err_dr9known1, err_dr9all, err_dr9nknown1, err_dr9nknownp])
    fg.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/nbar_mocks.pdf', bbox_inches='tight')    
    plt.show()
    
    
    chi2s = {}
    chi2s['fNL=0'] = ut.get_chi2pdf(err_0)
    chi2s['fNL=76.92'] = ut.get_chi2pdf(err_100)
    
    xlabel = r'Mean Density $\chi^{2}$'
    xlim1 = (20, 240)
    xlim2 = (630, 710)
    ylim = (0., 120.)

    fig = plt.figure()
    fig.subplots_adjust(wspace=0.03)
    gs  = GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ax1.tick_params(direction='in', axis='both', right=False)
    ax2.tick_params(direction='in', axis='both', which='both', left=False, right=True)

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.set(xlim=xlim1, ylim=ylim, xlabel=xlabel)
    ax2.set(xlim=xlim2, ylim=ylim)#, xticks=xticks2)
    ax2.set_yticklabels([])
    d = 0.01
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d/2, 1 + d/2), (-d, +d), **kwargs)  # bottom-left diagonal
    ax1.plot((1 - d/2, 1 + d/2), (1-d, 1+d), **kwargs)  # bottom-left diagonal
    kwargs.update(transform=ax2.transAxes)            # switch to the bottom axes
    ax2.plot((-d/2, +d/2), (-d, +d), **kwargs)        # top-left diagonal
    ax2.plot((-d/2, +d/2), (1-d, 1+d), **kwargs)        # top-left diagonal

    ls = ['-', '--']
    names = ['$f_\mathrm{NL}$=0', '$f_\mathrm{NL}$=76.9']
    for i, (__, chi2_i) in enumerate(chi2s.items()):
        print(np.max(chi2_i), np.min(chi2_i))
        ax1.hist(chi2_i, histtype='step', bins=65, 
                 ls=ls[i], label=names[i], range=(30., 160.), color='grey') 
    lgn = ax1.legend(title='Clean Mocks', frameon=True)
    for i,txt in enumerate(lgn.get_texts()):
        txt.set_color('grey')

    ax2.axvline(chi2_dr9, lw=1)
    ax2.annotate('No Weight', (chi2_dr9+5, 20), rotation=90, fontsize=13)

    ax1.axvline(chi2_dr9nknown1, lw=1, color='C4', ls='--')
    ax1.annotate('Nonlinear\n Three Maps', (chi2_dr9nknown1-9, 78), rotation=90, fontsize=13, color='C4')

    ax1.axvline(chi2_dr9known1, lw=1, color='C2', ls='-.')
    ax1.annotate('Linear Three Maps', (chi2_dr9known1+4, 20), rotation=90, fontsize=13, color='C2')

    ax1.axvline(chi2_dr9known, lw=1, color='C1', ls='--')
    ax1.annotate('Linear Two Maps', (chi2_dr9known+4, 20), rotation=90, fontsize=13, color='C1')


    for chi_i in [chi2_dr9, chi2_dr9known, chi2_dr9known1, chi2_dr9all, chi2_dr9nknown1, chi2_dr9nknownp]:
        is_gt = np.array(chi2s['fNL=0']) > chi_i
        print('p-value:', is_gt.mean())    

    np.savetxt('/users/PHS0336/medirz90/github/dimagfnl/figures/fig8_chi2test2_1.txt',
              [chi2_dr9, chi2_dr9known, chi2_dr9known1, chi2_dr9all, chi2_dr9nknown1, chi2_dr9nknownp])
    np.savetxt('/users/PHS0336/medirz90/github/dimagfnl/figures/fig8_chi2test2_2.txt', [chi2s['fNL=0'], chi2s['fNL=76.92']])    
    fig.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/chi2test2.pdf', bbox_inches='tight')    
 
            
def plot_clhist():
    
    def hist(ax, x, scaled, **kw):
        std_ = np.std(x)
        ax.hist(x, **kw)
        if scaled:
            text = f'$\sigma$={std_:.2f} [x$10^{{-5}}$]'
        else:
            text = f'$\sigma$={std_:.2f}'
        ax.text(0.4, 0.8, text, 
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
    hist(ax[0], 1.0e5*cl_0[:, 0], True, color='C0', **kw)
    hist(ax[1], 1.0e5*cl_100[:, 0], True, color='C0', **kw)   
    hist(ax[2], np.log10(cl_0[:, 0]), False, color='C1', **kw)
    hist(ax[3], np.log10(cl_100[:, 0]), False, color='C1', **kw)

    ax[0].set(yticks=[], ylabel='Distribution of first bin')
    ax[2].set(yticks=[], ylabel='Distribution of first bin')    
    for i, fnl in zip([0, 1], [0, 76.92]):        
        ax[i].set(xlabel=r'First bin $C_{\ell}$ [x$10^{-5}$]')        
        ax[i].text(0.15, 0.2, fr'$f_{{\rm NL}}={fnl:.2f}$', transform=ax[i].transAxes)        
        ax[2+i].set(xlabel=r'Log of first bin $C_{\ell}$')
        ax[2+i].text(0.5, 0.2, fr'$f_{{\rm NL}}={fnl:.2f}$', transform=ax[2+i].transAxes, color='C1')

    np.savetxt('/users/PHS0336/medirz90/github/dimagfnl/figures/fig5_hist_cl.txt', np.column_stack([cl_0[:, 0], cl_100[:, 0]]))
    fig.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/hist_cl.pdf', bbox_inches='tight')
    
    
def plot_mcmc_mocks():
    stg = {'mult_bias_correction_order':0,'smooth_scale_2D':0.15, 
           'smooth_scale_1D':0.3, 'contours': [0.68, 0.95]}
    mc_kw = dict(names=['fnl', 'b', 'n0'], 
                 labels=[r'f_{\rm NL}', 'b', '10^{7}n_{0}'], settings=stg) 
    read_kw = dict(ndim=3, iscale=[2])

    po0 = MCMC('/fs/ess/PHS0336/data/lognormal/v3/mcmc/mcmc_0_lrg_pozero_desic_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    lpo0 = MCMC('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logmcmc_0_lrg_pozero_desic_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    po100 = MCMC('/fs/ess/PHS0336/data/lognormal/v3/mcmc/mcmc_0_lrg_po100_desic_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    lpo100 = MCMC('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logmcmc_0_lrg_po100_desic_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    

    stats = {}
    stats[r'Clean $76.92$ & DESI & log$C_{\ell}$'] = lpo100.stats
    stats[r'Clean $76.92$ & DESI & $C_{\ell}$ ']   = po100.stats
    stats[r'Clean $76.92$ & DESI & log$C_{\ell}$ + $f_{\rm NL}=0$ cov '] = lpo0.stats
    stats[r'Clean $76.92$ & DESI & $C_{\ell}$ + $f_{\rm NL}=0$ cov '] = po0.stats


    g = plots.get_single_plotter(width_inch=5)
    g.settings.legend_fontsize = 13
    g.plot_2d([lpo100, po100, lpo0, po0], 'fnl', 'b', filled=True, colors='Dark2')
    g.add_x_marker(100./1.3)
    g.add_y_marker(1.43)
    g.get_axes().set_ylim(1.426, 1.434)
    g.get_axes().set_xlim(74.8, 80.2)
    g.add_legend([r'log$C_{\ell}$', r'$C_{\ell}$', 
                  r'log$C_{\ell}$ + $f_{\rm NL}=0$ cov ',
                  r'$C_{\ell}$ + $f_{\rm NL}=0$ cov '], colored_text=True, 
                legend_loc='lower left')
    ax = g.get_axes()
    ax.text(0.08, 0.92, r'Fitting the mean of $f_{\rm NL}=76.92$ mocks', 
            transform=ax.transAxes, fontsize=13)
    ax.text(98/1.3, 1.4302, 'Truth', color='grey', fontsize=13, alpha=0.7)
    g.fig.align_labels()  
    g.fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/mcmc_po100.pdf', bbox_inches='tight')
    plt.show()
    
    

    bm = MCMC('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logmcmc_0_lrg_zero_bmzls_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    nd = MCMC('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logmcmc_0_lrg_zero_ndecalsc_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    sd = MCMC('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logmcmc_0_lrg_zero_sdecalsc_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    ze = MCMC('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logmcmc_0_lrg_zero_desic_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    
    stats[r'Clean $0$ & DESI         &  log$C_{\ell}$ '] = ze.stats
    stats[r'Clean $0$ & BASS+MzLS    &  log$C_{\ell}$ '] = bm.stats       
    stats[r'Clean $0$ & DECaLS North &  log$C_{\ell}$'] = nd.stats
    stats[r'Clean $0$ & DECaLS South &  log$C_{\ell}$'] = sd.stats

    
    g = plots.get_single_plotter(width_inch=5)
    g.settings.legend_fontsize = 13
    g.plot_2d([bm, nd, sd, ze], 'fnl', 'b', filled=True, colors=['C1', 'C2', 'C3', 'C0'])
    g.add_x_marker(0)
    g.add_y_marker(1.43)
    g.get_axes().set_ylim(1.426, 1.434)
    g.get_axes().set_xlim(-2.2, 3.2)    
    ax = g.get_axes()
    ax.text(0.08, 0.92, r'Fitting the mean of $f_{\rm NL}$=0 mocks', 
            transform=ax.transAxes, fontsize=13)
    ax.text(-2.0, 1.4302, 'Truth', color='grey', fontsize=13, alpha=0.7)
    
    g.add_legend(['BASS+MzLS', 'DECaLS North', 'DECaLS South', r'DESI'], 
                 colored_text=True, legend_loc='lower left')    
    g.fig.align_labels()
    g.fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/mcmc_zero.pdf', bbox_inches='tight')        
    print_stats(stats)    

    
def plot_mcmc_contmocks():
    stg = {'mult_bias_correction_order':0,'smooth_scale_2D':0.15, 
           'smooth_scale_1D':0.3, 'contours': [0.68, 0.95]}
    mc_kw = dict(names=['fnl', 'b', 'n0'], 
                 labels=[r'f_{\rm NL}', 'b', '10^{7}n_{0}'], settings=stg) 
    read_kw = dict(ndim=3, iscale=[2])
    p = '/fs/ess/PHS0336/data/lognormal/v3/mcmc/'
    
    z_now  = MCMC(f'{p}logmcmc_0_lrg_zero_desic_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    z_nn1  = MCMC(f'{p}logmcmc_0_lrg_zero_desic_256_dnnp_known1_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    z_nnp  = MCMC(f'{p}logmcmc_0_lrg_zero_desic_256_dnnp_knownp_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    z_nnap  = MCMC(f'{p}logmcmc_0_lrg_zero_desic_256_dnnp_allp_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    
    #cz_now  = MCMC(f'{p}logmcmc_0_lrg_czero_desic_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    cz_nn1 = MCMC(f'{p}logmcmc_0_lrg_czero_desic_256_dnnp_known1_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    cz_nnp = MCMC(f'{p}logmcmc_0_lrg_czero_desic_256_dnnp_knownp_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)  
    cz_nnap  = MCMC(f'{p}logmcmc_0_lrg_czero_desic_256_dnnp_allp_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)      

    z_nn1_s  = MCMC(f'{p}logmcmc_0_lrg_zero_desic_256_dnnp_known1debiased_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    z_nnp_s  = MCMC(f'{p}logmcmc_0_lrg_zero_desic_256_dnnp_knownpdebiased_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    z_nnap_s  = MCMC(f'{p}logmcmc_0_lrg_zero_desic_256_dnnp_allpdebiased_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)

    
    
    
    stats = {}
    stats[r'Clean $0$ & No Weight']   = z_now.stats
    stats[r'Clean $0$ & Three Maps']  = z_nn1.stats
    stats[r'Clean $0$ & Four Maps']   = z_nnp.stats    
    stats[r'Clean $0$ & Nine Maps']   = z_nnap.stats        
    
    stats[r'Contaminated $0$ & Three Maps'] = cz_nn1.stats
    stats[r'Contaminated $0$ & Four Maps'] = cz_nnp.stats    
    stats[r'Contaminated $0$ & Nine Maps'] = cz_nnap.stats     
    
    

    colors = [plt.cm.Dark2(i) for i in [0, 2, 3, 1, 2, 3, 1]]        
    g = plots.get_single_plotter(width_inch=5)
    g.settings.legend_fontsize = 13
    g.plot_1d([z_now, z_nn1_s, z_nnp_s, z_nnap_s], 'fnl', colors=colors)
    ax = g.get_axes()
    ax.tick_params(top=False, right=False)
    g.add_legend(['No Weight', 'Nonlinear Three Maps', 'Nonlinear Four Maps', 'Nonlinear Nine Maps'], 
                 colored_text=True, legend_loc='upper left')    
    g.fig.align_labels()
    g.fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/mcmc_cont.pdf', bbox_inches='tight')        
    
    
    g = plots.get_single_plotter(width_inch=5)
    g.settings.legend_fontsize = 13
    g.plot_1d([z_now, z_nn1, z_nnp, z_nnap, cz_nn1, cz_nnp, cz_nnap], 'fnl', colors=colors,
             ls = ['-', '-', '-', '-', '--', '--', '--'])
    g.add_legend(['No Weight', 'Nonlinear Three Maps', 'Nonlinear Four Maps', 'Nonlinear Nine Maps',
                 ],
                colored_text=True, legend_loc='upper left') 
    ax = g.get_axes()
    ax.tick_params(top=False, right=False)
    ax.set_xlabel(r'$f_{\rm NL}$ + Mitigation Systematics')
    g.fig.align_labels()
    g.fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/mcmc_contnoshift.pdf', bbox_inches='tight')        

    
    
    

    z_now  = MCMC(f'{p}logmcmc_0_lrg_po100_desic_256_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    z_nn1  = MCMC(f'{p}logmcmc_0_lrg_po100_desic_256_dnnp_known1_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    z_nnp  = MCMC(f'{p}logmcmc_0_lrg_po100_desic_256_dnnp_knownp_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    z_nnap  = MCMC(f'{p}logmcmc_0_lrg_po100_desic_256_dnnp_allp_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    
    cz_nn1 = MCMC(f'{p}logmcmc_0_lrg_cpo100_desic_256_dnnp_known1_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    cz_nnp = MCMC(f'{p}logmcmc_0_lrg_cpo100_desic_256_dnnp_knownp_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    cz_nnap  = MCMC(f'{p}logmcmc_0_lrg_cpo100_desic_256_dnnp_allp_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    
    
    z_nn1_s  = MCMC(f'{p}logmcmc_0_lrg_po100_desic_256_dnnp_known1debiased_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    z_nnp_s  = MCMC(f'{p}logmcmc_0_lrg_po100_desic_256_dnnp_knownpdebiased_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    z_nnap_s  = MCMC(f'{p}logmcmc_0_lrg_po100_desic_256_dnnp_allpdebiased_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    

    stats[r'Clean $76.92$ & No Weight']  = z_now.stats
    stats[r'Clean $76.92$ & Three Maps'] = z_nn1.stats
    stats[r'Clean $76.92$ & Four Maps']  = z_nnp.stats    
    stats[r'Clean $76.92$ & Nine Maps']  = z_nnap.stats   
    
    stats[r'Contaminated $76.92$ & Three Maps'] = cz_nn1.stats
    stats[r'Contaminated $76.92$ & Four Maps']  = cz_nnp.stats    
    stats[r'Contaminated $76.92$ & Nine Maps'] = cz_nnap.stats   
    
    colors = [plt.cm.Dark2(i) for i in [0, 2, 3, 1, 2, 3, 1]]        
    g = plots.get_single_plotter(width_inch=5)
    g.settings.legend_fontsize = 13
    g.plot_1d([z_now, z_nn1_s, z_nnp_s, z_nnap_s], 'fnl', colors=colors)
    ax = g.get_axes()
    ax.tick_params(top=False, right=False)
    g.add_legend(['No Weight', 'Nonlinear Three Maps', 'Nonlinear Four Maps', 'Nonlinear Nine Maps'], 
                 colored_text=True, legend_loc='upper left')    
    g.fig.align_labels()
    g.fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/mcmcp_cont.pdf', bbox_inches='tight')        
    
    
    g = plots.get_single_plotter(width_inch=5)
    g.settings.legend_fontsize = 13
    g.plot_1d([z_now, z_nn1, z_nnp, z_nnap, cz_nn1, cz_nnp, cz_nnap], 'fnl', colors=colors,
             ls = ['-', '-', '-', '-', '--', '--', '--'])
    g.add_legend(['No Weight', 'Nonlinear Three Maps', 'Nonlinear Four Maps', 'Nonlinear Nine Maps',
                 ],
                colored_text=True, legend_loc='upper left') 
    ax = g.get_axes()
    ax.tick_params(top=False, right=False)
    ax.set_xlabel(r'$f_{\rm NL}$ + Mitigation Systematics')
    g.fig.align_labels()
    g.fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/mcmcp_contnoshift.pdf', bbox_inches='tight')     
    
    print_stats(stats)    

    
    
def plot_bestfit():
    bf = np.load('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logbestfit_0_lrg_zero_desic_256_noweight.npz')
    fnl, b = bf['params'][:, :2].T
    fnl /= gratio
    
    map_ = plt.scatter(fnl, b, s=10, 
                       c=2*bf['neglog']/1000, alpha=0.5, cmap='jet', vmin=20, vmax=50)
    #plt.scatter(*bf['params'].mean(axis=0)[:2],  s=500, marker='*', color='C0', alpha=1.0)
    print(np.mean(fnl))
    print(np.std(fnl))
    plt.axvline(0, lw=1, ls=':', color='grey', zorder=-10)
    plt.axhline(1.43, lw=1, ls=':', color='grey', zorder=-10)
    plt.text(22., 1.432, 'Truth', color='grey', alpha=0.8, fontsize=13)
    plt.xlabel(r'$f_{\rm NL}$')
    plt.ylabel('b')
    plt.colorbar(map_, label=r'Min $\chi^{2}$')    
    plt.ylim(1.36, 1.50)    
    plt.twinx()
    plt.xlim(-70, 40)
    plt.yticks([])
    plt.hist(fnl, zorder=-10, histtype='step', bins=24, color='C0')
    plt.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/bestfit_zero.pdf', bbox_inches='tight')    
    plt.show()
    
    
    bf = np.load('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logbestfit_0_lrg_po100_desic_256_noweight.npz')
    fnl, b = bf['params'][:, :2].T
    fnl /= gratio
    
    map_ = plt.scatter(fnl, b, s=10, 
                       c=2*bf['neglog']/1000, alpha=0.5, cmap='jet', vmin=20, vmax=50)
    #plt.scatter(*bf['params'].mean(axis=0)[:2],  s=500, marker='*', color='C0', alpha=1.0)
    print(np.mean(fnl))
    print(np.std(fnl))
    plt.axvline(100/gratio, lw=1, ls=':', color='grey', zorder=-10)
    plt.axhline(1.43, lw=1, ls=':', color='grey', zorder=-10)
    plt.text(110, 1.432, 'Truth', color='grey', alpha=0.8, fontsize=13)
    plt.xlabel(r'$f_{\rm NL}$')
    plt.ylabel('b')
    plt.ylim(1.36, 1.50)    
    plt.colorbar(map_, label=r'Min $\chi^{2}$')
    plt.twinx()
    plt.yticks([])
    plt.xlim(20, 130)       
    plt.hist(fnl, zorder=-10, histtype='step', bins=24, color='C0')
    plt.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/bestfit_po100.pdf', bbox_inches='tight')    
    plt.show()

    
def plot_mcmc_data():
    stg = {'mult_bias_correction_order':0,'smooth_scale_2D':0.3, 
           'smooth_scale_1D':0.3, 'contours': [0.68, 0.95]}
    mc_kw = dict(names=['fnl', 'b', 'n0'], 
                 labels=[r'f_{\rm NL}', 'b', '10^{7}n_{0}'], settings=stg) 
    read_kw = dict(ndim=3, iscale=[2])
    
    mc_kwj = dict(names=['fnl', 'b1', 'n1', 'b2', 'n2', 'b3', 'n3'], 
                 labels=[r'f_{\rm NL}', 'b', '10^{7}n_{0}', 'b', '10^{7}n_{0}', 'b', '10^{7}n_{0}'], settings=stg) 
    read_kwj = dict(ndim=7, iscale=[2, 4, 6])
    
    p = '/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/'
    ze   = MCMC(f'{p}logmcmc_lrg_zero_desic_noweight_steps10k_walkers50_elmin0_p1.0_s0.945.npz', mc_kw=mc_kw, read_kw=read_kw)
    po   = MCMC(f'{p}logmcmc_lrg_zero_desic_linp_all_steps10k_walkers50_elmin0_p1.0_s0.945.npz', mc_kw=mc_kw, read_kw=read_kw)    
    kn   = MCMC(f'{p}logmcmc_lrg_zero_desic_linp_known_steps10k_walkers50_elmin0_p1.0_s0.945.npz', mc_kw=mc_kw, read_kw=read_kw)    
    kn1  = MCMC(f'{p}logmcmc_lrg_zero_desic_linp_known1_steps10k_walkers50_elmin0_p1.0_s0.945.npz', mc_kw=mc_kw, read_kw=read_kw)        
    knn1 = MCMC(f'{p}logmcmc_lrg_zero_desic_dnnp_known1_steps10k_walkers50_elmin0_p1.0_s0.945.npz', mc_kw=mc_kw, read_kw=read_kw)            
    dsl  = MCMC(f'{p}logmcmc_lrg_zero_desicl_dnnp_known1_steps10k_walkers50_elmin0_p1.0_s0.945.npz', mc_kw=mc_kw, read_kw=read_kw)         
    dsf  = MCMC(f'{p}logmcmc_lrg_zero_desicf_dnnp_known1_steps10k_walkers50_elmin0_p1.0_s0.945.npz', mc_kw=mc_kw, read_kw=read_kw)           
    dss  = MCMC(f'{p}logmcmc_lrg_po100_desic_dnnp_known1_steps10k_walkers50_elmin0_p1.0_s0.945.npz', mc_kw=mc_kw, read_kw=read_kw)      
    dsp  = MCMC(f'{p}logmcmc_lrg_zero_desic_dnnp_allp_steps10k_walkers50_elmin0_p1.0_s0.945.npz', mc_kw=mc_kw, read_kw=read_kw)
    dskp  = MCMC(f'{p}logmcmc_lrg_zero_desic_dnnp_knownp_steps10k_walkers50_elmin0_p1.0_s0.945.npz', mc_kw=mc_kw, read_kw=read_kw)
    knn1_s = MCMC(f'{p}logmcmc_lrg_zero_desic_dnnp_known1debiased_steps10k_walkers50_elmin0_p1.0_s0.945.npz', mc_kw=mc_kw, read_kw=read_kw)
    dsp_s  = MCMC(f'{p}logmcmc_lrg_zero_desic_dnnp_allpdebiased_steps10k_walkers50_elmin0_p1.0_s0.945.npz', mc_kw=mc_kw, read_kw=read_kw)
    dskp_s  = MCMC(f'{p}logmcmc_lrg_zero_desic_dnnp_knownpdebiased_steps10k_walkers50_elmin0_p1.0_s0.945.npz', mc_kw=mc_kw, read_kw=read_kw)
    #knn1joint = MCMC(f'{p}logmcmc_lrg_zero_bmzlsndecalscsdecalsc_dnnp_known1_steps10k_walkers50_elmin0.npz', mc_kw=mc_kwj, read_kw=read_kwj)

    knn1b = MCMC(f'{p}logmcmc_lrg_zero_bmzls_dnnp_known1_steps10k_walkers50_elmin0_p1.0_s0.951.npz', mc_kw=mc_kw, read_kw=read_kw)                
    bml   = MCMC(f'{p}logmcmc_lrg_zero_bmzlsl_dnnp_known1_steps10k_walkers50_elmin0_p1.0_s0.951.npz', mc_kw=mc_kw, read_kw=read_kw)  
    bmf   = MCMC(f'{p}logmcmc_lrg_zero_bmzlsf_dnnp_known1_steps10k_walkers50_elmin0_p1.0_s0.951.npz', mc_kw=mc_kw, read_kw=read_kw)  
    bmp  = MCMC(f'{p}logmcmc_lrg_zero_bmzls_dnnp_allp_steps10k_walkers50_elmin0_p1.0_s0.951.npz', mc_kw=mc_kw, read_kw=read_kw)
    bmkp  = MCMC(f'{p}logmcmc_lrg_zero_bmzls_dnnp_knownp_steps10k_walkers50_elmin0_p1.0_s0.951.npz', mc_kw=mc_kw, read_kw=read_kw)
    
    knn1n = MCMC(f'{p}logmcmc_lrg_zero_ndecalsc_dnnp_known1_steps10k_walkers50_elmin0_p1.0_s0.943.npz', mc_kw=mc_kw, read_kw=read_kw)
    ndce  = MCMC(f'{p}logmcmc_lrg_zero_ndecalsc_dnnp_known1ext_steps10k_walkers50_elmin0_p1.0_s0.943.npz', mc_kw=mc_kw, read_kw=read_kw)    
    nd    = MCMC(f'{p}logmcmc_lrg_zero_ndecals_dnnp_known1_steps10k_walkers50_elmin0_p1.0_s0.943.npz', mc_kw=mc_kw, read_kw=read_kw)            
    ndl   = MCMC(f'{p}logmcmc_lrg_zero_ndecalscl_dnnp_known1_steps10k_walkers50_elmin0_p1.0_s0.943.npz', mc_kw=mc_kw, read_kw=read_kw)
    ndf   = MCMC(f'{p}logmcmc_lrg_zero_ndecalscf_dnnp_known1_steps10k_walkers50_elmin0_p1.0_s0.943.npz', mc_kw=mc_kw, read_kw=read_kw)
    ndp  = MCMC(f'{p}logmcmc_lrg_zero_ndecalsc_dnnp_allp_steps10k_walkers50_elmin0_p1.0_s0.943.npz', mc_kw=mc_kw, read_kw=read_kw)
    ndkp  = MCMC(f'{p}logmcmc_lrg_zero_ndecalsc_dnnp_knownp_steps10k_walkers50_elmin0_p1.0_s0.943.npz', mc_kw=mc_kw, read_kw=read_kw)
    
    knn1s = MCMC(f'{p}logmcmc_lrg_zero_sdecalsc_dnnp_known1_steps10k_walkers50_elmin0_p1.0_s0.943.npz', mc_kw=mc_kw, read_kw=read_kw)    
    sdce  = MCMC(f'{p}logmcmc_lrg_zero_sdecalsc_dnnp_known1ext_steps10k_walkers50_elmin0_p1.0_s0.943.npz', mc_kw=mc_kw, read_kw=read_kw)  
    sd    = MCMC(f'{p}logmcmc_lrg_zero_sdecals_dnnp_known1_steps10k_walkers50_elmin0_p1.0_s0.943.npz', mc_kw=mc_kw, read_kw=read_kw)            
    sdl   = MCMC(f'{p}logmcmc_lrg_zero_sdecalscl_dnnp_known1_steps10k_walkers50_elmin0_p1.0_s0.943.npz', mc_kw=mc_kw, read_kw=read_kw)
    sdf   = MCMC(f'{p}logmcmc_lrg_zero_sdecalscf_dnnp_known1_steps10k_walkers50_elmin0_p1.0_s0.943.npz', mc_kw=mc_kw, read_kw=read_kw)
    sdp  = MCMC(f'{p}logmcmc_lrg_zero_sdecalsc_dnnp_allp_steps10k_walkers50_elmin0_p1.0_s0.943.npz', mc_kw=mc_kw, read_kw=read_kw)
    sdkp  = MCMC(f'{p}logmcmc_lrg_zero_sdecalsc_dnnp_knownp_steps10k_walkers50_elmin0_p1.0_s0.943.npz', mc_kw=mc_kw, read_kw=read_kw)
    
    stats1 = {}
    stats1['DESI                      & No Weight'] = ze.stats
    stats1['DESI                      & Nonlinear Three Maps'] = knn1_s.stats    
    stats1['DESI                      & Nonlinear Four Maps'] = dskp_s.stats    
    stats1['DESI                      & Nonlinear Nine Maps'] = dsp_s.stats    


    
    stats = {}
    stats['DESI                      & No Weight'] = ze.stats
    stats['DESI                      & Linear Eight Maps'] = po.stats    
    stats['DESI                      & Linear Two Maps'] = kn.stats    
    stats['DESI                      & Linear Three Maps'] = kn1.stats
    stats['DESI                      & Nonlinear Three Maps'] = knn1.stats    
    stats['DESI (imag. cut)          & Nonlinear Three Maps'] = dsl.stats      
    stats['DESI (comp. cut)          & Nonlinear Three Maps'] = dsf.stats          
    stats['DESI                      & Nonlinear Four Maps'] = dskp.stats    
    stats['DESI                      & Nonlinear Nine Maps'] = dsp.stats    
    stats[r'DESI                     & Nonlinear Three Maps+$f_{\rm NL}=76.92$ Cov'] = dss.stats          

    
    stats['BASS+MzLS                 & Nonlinear Three Maps'] = knn1b.stats    
    stats['BASS+MzLS                 & Nonlinear Four Maps'] = bmkp.stats            
    stats['BASS+MzLS                 & Nonlinear Nine Maps'] = bmp.stats            
    stats['BASS+MzLS (imag. cut)     & Nonlinear Three Maps'] = bml.stats 
    stats['BASS+MzLS (comp. cut)     & Nonlinear Three Maps'] = bmf.stats     

    
    stats['DECaLS North              & Nonlinear Three Maps'] = knn1n.stats    
    stats['DECaLS North              & Nonlinear Four Maps'] = ndkp.stats                
    stats['DECaLS North              & Nonlinear Five Maps'] = ndce.stats 
    stats['DECaLS North              & Nonlinear Nine Maps'] = ndp.stats            
    stats['DECaLS North (no DEC cut) & Nonlinear Three Maps'] = nd.stats
    stats['DECaLS North (imag. cut)  & Nonlinear Three Maps'] = ndl.stats        
    stats['DECaLS North (comp. cut)  & Nonlinear Three Maps'] = ndf.stats        
    
    stats['DECaLS South              & Nonlinear Three Maps'] = knn1s.stats    
    stats['DECaLS South              & Nonlinear Four Maps'] = sdkp.stats                    
    stats['DECaLS South              & Nonlinear Five Maps'] = sdce.stats        
    stats['DECaLS South              & Nonlinear Nine Maps'] = sdp.stats                
    stats['DECaLS South (no DEC cut) & Nonlinear Three Maps'] = sd.stats   
    stats['DECaLS South (imag. cut)  & Nonlinear Three Maps'] = sdl.stats        
    stats['DECaLS South (comp. cut)  & Nonlinear Three Maps'] = sdf.stats            
    
    
    
    # Triangle plot
    colors = [plt.cm.Dark2(i) for i in [0, 1, 2, 3, 4, 2, 3, 4]]        
    g = plots.get_single_plotter(width_inch=5)
    g.settings.legend_fontsize = 13
    g.plot_2d([ze, knn1_s, dskp_s, dsp_s], 
              'fnl', 'b', 
              filled=True,lims=[-100, 200, 1.2, 1.5], colors='Dark2') # 
    g.add_legend(['No weight',                  
                  'Nonlinear Three Maps',
                  'Nonlinear Four Maps',
                  'Nonlinear Nine Maps'], 
                  colored_text=True, legend_loc='lower left')    
    g.fig.align_labels()
    ax = g.get_axes()
    ax.text(0.15, 0.92, 'DR9 DESI Footprint (different methods)', 
            transform=ax.transAxes, fontsize=13)          
    np.savetxt('/users/PHS0336/medirz90/github/dimagfnl/figures/fig11_mcmc_dr9methods.txt',
              [ze.samples[:, 0], knn1_s.samples[:, 0], dskp_s.samples[:, 0], dsp_s.samples[:, 0]])

    g.fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/mcmc_dr9methods.pdf', bbox_inches='tight')    
    plt.show()
    
    g = plots.get_single_plotter(width_inch=5)
    g.settings.legend_fontsize = 13
    ls = ['-', '-', '-', '-', '-', '--', '--', '--']
    #g.plot_1d([ze, kn1, knn1, dskp, dsp, knn1_s, dskp_s, dsp_s], 'fnl',
    #          filled=True,lims=[-50, 170], colors=colors, ls=ls) # 
    g.plot_1d([ze, knn1_s, dskp_s, dsp_s], 'fnl',
                 filled=True,lims=[-100,200], colors=colors, ls=ls) #     
    g.add_legend(['No weight',
                  'Nonlinear Three Maps',
                  'Nonlinear Four Maps',
                  'Nonlinear Nine Maps'],
                  colored_text=True, legend_loc='lower left')    
    g.fig.align_labels()
    ax = g.get_axes()
    ax.tick_params(top=False, right=False)
    ax.text(0.15, 0.92, 'DR9 DESI Footprint (different methods)', 
            transform=ax.transAxes, fontsize=13)    
    g.fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/mcmc_dr9methods1d.pdf', bbox_inches='tight')    
    plt.show()    
    
    g = plots.get_single_plotter(width_inch=5)
    g.settings.legend_fontsize = 13
    ls = ['-', '-', '-', '-', '-', '--', '--', '--']
    g.plot_1d([ze, knn1, dskp, dsp], 'fnl',
                 filled=True,lims=[-100, 200], colors=colors, ls=ls) #     
    g.add_legend(['No weight',
                  'Nonlinear Three Maps',
                  'Nonlinear Four Maps',
                  'Nonlinear Nine Maps'],
                  colored_text=True, legend_loc='lower left')    
    g.fig.align_labels()
    ax = g.get_axes()
    ax.set_xlabel(r'$f_{\rm NL}$ + Mitigation Systematics')    
    ax.tick_params(top=False, right=False)
    ax.text(0.15, 0.92, 'DR9 DESI Footprint (different methods)', transform=ax.transAxes, fontsize=13)   
    #ax.text(0.15, 0.85, 'Not accounted for mitigation bias', 
    #        transform=ax.transAxes, fontsize=13)
    np.savetxt('/users/PHS0336/medirz90/github/dimagfnl/figures/fig12_mcmc_dr9methodsnoshift.txt',
              [ze.samples[:, 0], knn1.samples[:, 0], dskp.samples[:, 0], dsp.samples[:, 0]])    
    g.fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/mcmc_dr9methods1dnoshift.pdf',
                  bbox_inches='tight')              
    
    # Triangle plot
    g = plots.get_single_plotter(width_inch=5)
    g.settings.legend_fontsize = 13
    g.plot_2d([knn1b, knn1n, knn1s, knn1], 'fnl', 'b', filled=True,
              lims=[-100, 200, 1.2, 1.5], colors=['C1', 'C2', 'C3', 'C0'])
    g.add_legend(['BASS+MzLS', 'DECaLS North', 'DECaLS South', 'DESI'], 
                 colored_text=True, legend_loc='lower left')  
    ax = g.get_axes()
    ax.text(0.15, 0.92, 'Nonlinear Three Maps (different regions)', 
            transform=ax.transAxes, fontsize=13)
    #ax.text(0.15, 0.85, 'Not accounted for mitigation bias', 
    #        transform=ax.transAxes, fontsize=13)  
    ax.set_xlabel(r'$f_{\rm NL}$ + Mitigation Systematics')
    g.fig.align_labels()
    np.savetxt('/users/PHS0336/medirz90/github/dimagfnl/figures/fig13_mcmc_dr9regions.txt',
              [knn1b.samples[:, 0], knn1n.samples[:, 0], knn1s.samples[:, 0], knn1.samples[:, 0]])    
    g.fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/mcmc_dr9regions.pdf', bbox_inches='tight')    
    plt.show() 

#     g = plots.get_single_plotter(width_inch=5)
#     g.settings.legend_fontsize = 13
#     g.plot_1d([knn1b, knn1n, knn1s, knn1, knn1joint], 'fnl', filled=True,
#               lims=[-100, 120], colors=['C1', 'C2', 'C3', 'C0', 'C4'])
#     g.add_legend(['BASS+MzLS', 'DECaLS North', 'DECaLS South', 'DESI', 'DESI (joint)'], 
#                  colored_text=True, legend_loc='lower left')  
#     ax = g.get_axes()
#     ax.text(0.15, 0.92, 'Nonlinear Three Maps (different regions)', 
#             transform=ax.transAxes, fontsize=13)
#     #ax.text(0.15, 0.85, 'Not accounted for mitigation bias', 
#     #        transform=ax.transAxes, fontsize=13)  
#     ax.set_xlabel(r'$f_{\rm NL}$ + Mitigation Systematics')
#     g.fig.align_labels()
#     plt.show() 

    
    
    
#     # Triangle plot
#     g = plots.get_single_plotter(width_inch=4*1.5)
#     g.settings.legend_fontsize = 14
#     g.plot_2d([knn1n, ndce, nd, 
#                knn1s, sdce, sd], 'fnl', 'b', filled=True)
#     g.add_legend(['North', 'North [+CALIBZ+HI]', 'North [no dec cut]', 
#                   'South', 'South [+CALIBZ+HI]', 'South [no dec cut]'], colored_text=True, legend_loc='lower left')    
#     g.get_axes().set_ylim(1.25, 1.55)
#     g.get_axes().set_xlim(-60, 130)    
#     g.fig.align_labels()
#     g.fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/mcmc_dr9_cutdec.pdf', bbox_inches='tight')
#     plt.show()
    print_stats(stats)
    print(10*'=')
    print_stats(stats1)
    

    
def plot_model(fnltag='po100'):
    bm = np.load(f'/fs/ess/PHS0336/data/lognormal/v3/mcmc/logmcmc_0_lrg_{fnltag}_desic_256_noweight_steps10k_walkers50.npz')
    #bm['best_fit'][0] /= gratio   
    
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
    print(fnl/gratio, b)
    
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


    fig = plt.figure(figsize=(6, 6), constrained_layout=False)
    gs = GridSpec(3, 1, figure=fig)

    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[2, 0])

    lw = [0.8, 0.8, 3.]
    ls = ['-', '-', '-']
    al = [1., 1., 0.7]
    
    ydata = []
    ydata.append(el)
    ydata.append(cl)
    ydata.append(cl_err)
    
    for i, (n, v) in enumerate(cl_models.items()):
        kw = dict(label=n, lw=lw[i], ls=ls[i], alpha=al[i])
        ax1.plot(v[0], np.log10(v[1]), **kw)
        ax2.plot(el, np.log10(v[1])-cl, **kw)
        
        ydata.append(np.log10(v[1]))
        
    f = np.sqrt(1000.)
    ax1.plot(el, cl, 'C0--', label='Mean of Mocks')
    #ax2.axhline(0.0, color='C0', ls='--')
    ax2.fill_between(el, -cl_err, cl_err, alpha=0.10, color='k')
    ax2.fill_between(el, -f*cl_err, f*cl_err, alpha=0.05, color='k')


    ax1.legend(ncol=1)
    ax1.set(xscale='log', ylabel=r'$\log C_{\ell}$') #, yscale='log')
    ax1.tick_params(labelbottom=False)
    ax2.set(xscale='log', xlabel=r'$\ell$', ylabel=r'$\Delta \log C_{\ell}$', xlim=ax1.get_xlim(), ylim=(-0.025, +0.025))

    fig.subplots_adjust(hspace=0.0, wspace=0.02)
    fig.align_labels()
    np.savetxt('/users/PHS0336/medirz90/github/dimagfnl/figures/fig4_model_mock.txt', np.column_stack(ydata))
    fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/model_mock.pdf', bbox_inches='tight')     
    
    
def plot_dr9cl():
    # read survey geometry
    dt = ft.read(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_desic_256.fits')
    w = np.zeros(12*256*256)
    w[dt['hpix']] = dt['fracgood']
    weight = hp.ud_grade(w, 1024)
    mask = weight > 0.5

    z, b, dNdz = init_sample(kind='lrg')
    model = SurveySpectrum()
    model.add_tracer(z, b, dNdz, p=1.0, s=0.945)
    model.add_kernels(model.el_model)
    model.add_window(weight, mask, np.arange(2048), ngauss=2048)  
    
    fnltag = 'zero'
    cl_ = np.load(f'/fs/ess/PHS0336/data/lognormal/v3/clustering/logclmock_0_lrg_{fnltag}_desic_256_noweight_mean.npz')
    cl_cov_ = np.load(f'/fs/ess/PHS0336/data/lognormal/v3/clustering/logclmock_0_lrg_{fnltag}_desic_256_noweight_cov.npz')

    cl_err = np.diagonal(cl_cov_['clcov']*1000.)**0.5

    mk = ['.', 'o', 'x', '^', 's', '1']
    el_g = np.arange(300)

    plt.figure(figsize=(6, 5))

    ydata1=[]
    ydata2=[]
    
    for i, (n, nm) in enumerate(zip(['noweight', 'linp_known', 'linp_known1', 'linp_all', 'dnnp_known1', 'dnnp_knownp'],
                     ['No Weight', 'Linear Two Maps', 'Linear Three Maps', 'Linear Eight Maps', 'Nonlinear Three Maps', 'Nonlinear Four Maps'])):
        cl_d = np.load(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/clustering/0.57.0/cl_lrg_desic_256_{n}.npy', allow_pickle=True).item()
        cl_b = np.log10(ut.histogram_cell(cl_d['cl_gg']['l'], cl_d['cl_gg']['cl'], bins=ut.ell_edges)[1])

        bestp = np.load(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/logmcmc_lrg_zero_desic_{n}_steps10k_walkers50_elmin0_p1.0_s0.945.npz')
        fnl, b, noise = bestp['best_fit']
        print(nm, fnl/gratio, b)

        cl_bf = np.log10(model(el_g, fnl=fnl, b=b, noise=noise))
        ln = plt.plot(el_g[2:], cl_bf[2:], lw=1)
        
        ydata1.append([el_g[2:], cl_bf[2:]])
        plt.scatter(cl_['el_bin'], cl_b, label=nm, marker=mk[i], color=ln[0].get_color())
        ydata2.append([cl_['el_bin'], cl_b])
        
    ln, = plt.plot(cl_['el_bin'], cl_['cl'], label='Mean of Mocks', ls='-', lw=3, alpha=0.8, color='grey')
    plt.fill_between(cl_['el_bin'], cl_['cl']-cl_err, cl_['cl']+cl_err, alpha=0.1, color=ln.get_color())
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\log C_{\ell}$')
    #plt.ylim(1.2e-6, 2.2e-4)
    plt.xlim(xmin=1.9)
    lgn = plt.legend(ncol=1, loc='upper right')
    for i, txt in enumerate(lgn.get_texts()):
        if i ==0:continue
        txt.set_color('C%d'%(i-1))
        
    np.savetxt('/users/PHS0336/medirz90/github/dimagfnl/figures/fig10_model_dr9_1.txt', np.column_stack(ydata1))
    np.savetxt('/users/PHS0336/medirz90/github/dimagfnl/figures/fig10_model_dr9_2.txt', np.column_stack(ydata2))    
    plt.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/model_dr9.pdf', bbox_inches='tight')         
    

def plot_fnl_lmin():
    
    fig, ax = plt.subplots(ncols=4, figsize=(10, 4), sharey=True)
    fig.subplots_adjust(wspace=0.02)
    
    colors = ['C0', 'C1', 'C2', 'C3']
    names = ['DESI', 'BASS+MzLS', 'DECaLS North', 'DECaLS South']
    markers = ['o', 'x', '^', 's']
    p_ = '/fs/ess/PHS0336/data/rongpu/imaging_sys/clustering/0.57.0/'
    
    for i, r in enumerate(['desic','bmzls', 'ndecalsc', 'sdecalsc']):
        for j, n in enumerate(['noweight', 'dnnp_known1']):
            
            cl_d = np.load(f'{p_}cl_lrg_{r}_256_{n}.npy', allow_pickle=True).item()
            elb, clb = ut.histogram_cell(cl_d['cl_gg']['l'], cl_d['cl_gg']['cl'], bins=ut.ell_edges)
            cl_b = np.log10(clb)
            
            ls = '-' if j==0 else 'none'
            mk = '.' if j==0 else markers[i]
            ln = ax[i].plot(elb, cl_b, marker=mk, ls=ls, color=colors[i], alpha=0.8)
        ax[i].text(0.1, 0.85, names[i], color=colors[i], transform=ax[i].transAxes)
        
        cl_ = np.load(f'/fs/ess/PHS0336/data/lognormal/v3/clustering/logclmock_0_lrg_zero_{r}_256_noweight_mean.npz')
        cl_cov_ = np.load(f'/fs/ess/PHS0336/data/lognormal/v3/clustering/logclmock_0_lrg_zero_{r}_256_noweight_cov.npz')
        cl_err = np.diagonal(cl_cov_['clcov']*1000.)**0.5
        ax[i].plot(cl_['el_bin'], cl_['cl'], label='Mean of Mocks', alpha=0.5, color='k')
        ax[i].fill_between(cl_['el_bin'], cl_['cl']-cl_err, cl_['cl']+cl_err, alpha=0.1, color='k')

    for a in ax:
        a.set(xlim=(1.9, 41), ylim=(-6, -3), 
              xlabel=r'$\ell$', xscale='log')
    ax[0].set_ylabel(r'$\log C_{\ell}$')
    ax[3].text(0.1, 0.1, r'68\% $f_{\rm NL}=0$ mocks', transform=ax[3].transAxes, fontsize=13, alpha=0.8)    
    fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/cldr9_lowell.pdf', bbox_inches='tight')
    
    
    
    
    p = '/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/'
    stg = {'mult_bias_correction_order':0,'smooth_scale_2D':0.15, 
           'smooth_scale_1D':0.3, 'contours': [0.68, 0.95]}
    mc_kw = dict(names=['fnl', 'b', 'n0'], 
                 labels=[r'f_{\rm NL}', 'b', '10^{7}n_{0}'], settings=stg) 

    stats = {}
    elmin = np.arange(10)
    s = {'desic':0.945, 'bmzls':0.951, 'ndecalsc':0.943, 'sdecalsc':0.943}
    
    for r in ['desic', 'bmzls', 'ndecalsc', 'sdecalsc']:
        d = []
        s_ = s[r]
        for elmin_ in elmin:
            d_ = MCMC(f'{p}logmcmc_lrg_zero_{r}_dnnp_known1_steps10k_walkers50_elmin{elmin_}_p1.0_s{s_}.npz', mc_kw=mc_kw)
            d.append(d_.stats)
        stats[r] = np.array(d)    
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ydata = []
    
    names = ['DESI', 'BASS+MzLS', 'DECaLS North', 'DECaLS South',]
    markers = ['o', 'x', '^', 's']
    for i, st in enumerate(['desic', 'bmzls', 'ndecalsc', 'sdecalsc']):
        yp  = stats[st][:, 3]-stats[st][:, 0]
        yn  = stats[st][:, 0]-stats[st][:, 2]
        alpha = 1.0 if st=='desic' else 0.8
        ln = ax.errorbar(elmin+(i*0.1-0.1), stats[st][:, 0], yerr=[yn, yp], 
                         label=names[i], marker=markers[i], capsize=6, ls='none', alpha=alpha)
        
        ydata.append(stats[st][:, 0])
        ydata.append(yn)
        ydata.append(yp)
        
        # --- 95%
        #yp2 = stats[st][:, 5]-stats[st][:, 0]
        #yn2 = stats[st][:, 0]-stats[st][:, 4]    
        #ax.errorbar(elmin+(i*0.1-0.1), stats[st][:, 0], yerr=[yn2, yp2], capsize=5, ls='none', 
        #           color=ln.lines[0].get_color(), alpha=alpha, zorder=-10)   

    lgn = ax.legend(loc='lower left', #bbox_to_anchor=(0, 1.02, 1, 0.4),
                    ncol=2, frameon=False)#mode="expand", borderaxespad=0, 
    for i, li in enumerate(lgn.get_texts()):
        li.set_color('C%d'%i)
    dv.add_locators(ax, xmajor=None, ymajor=50)    

    ax.set_ylim(-55, 155)
    ax.set_xticks(elmin)
    ax.set_xticklabels(ut.ell_edges[:elmin.max()+1])
    ax.axhline(0.0, ls=':', color='grey', lw=1)
    ax.set_xlim(-0.5, 9.5)
    ax.set(xlabel=r'$\ell_{\rm min}$', ylabel=r'$f_{\rm NL}$ + Mitigation Systematics')
    fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/fnl_elmin.pdf', bbox_inches='tight')  
    
    np.savetxt('/users/PHS0336/medirz90/github/dimagfnl/figures/fig14_fnl_elmin.txt',
              np.column_stack(ydata))


def test_nz():
    # read survey geometry
    dt = ft.read(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_desic_256.fits')
    w = np.zeros(12*256*256)
    w[dt['hpix']] = dt['fracgood']
    weight = hp.ud_grade(w, 1024)
    mask = weight > 0.5

    p = '/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/'
    bestp = np.load(f'{p}logmcmc_lrg_zero_desic_dnnp_known1_steps10k_walkers50_elmin0.npz')
    fnl, b0, noise = bestp['best_fit']
    print(fnl, b0, mask.mean())
    
    def call_cell(z, b, dNdz):
        model = SurveySpectrum()
        model.add_tracer(z, b, dNdz, p=1.0)
        model.add_kernels(model.el_model)
        model.add_window(weight, mask, np.arange(2048), ngauss=2048)
        res = []
        for fnl_ in [0, fnl]:
            print(fnl_)
            res.append(model(np.arange(500), fnl_, b0, noise))
        return res    
    
    def treat_noz(z, dNdz, kind=1, zl=23, zh=132):
        nz_low = lambda x:(x/z[zl])**kind*dNdz[zl]
        nz_high = lambda x:np.exp(-1.*(x-z[zh])*kind)*dNdz[zh]
        dNdz_ = dNdz.copy()
        dNdz_[:zl+1] = nz_low(z[:zl+1])
        dNdz_[zh:] = nz_high(z[zh:])
        return dNdz_        

    z, b, dNdz = init_sample('lrg')
    dNdz8 = treat_noz(z, dNdz, kind=8, zl=25, zh=132)
    dNdz4 = treat_noz(z, dNdz, kind=4, zl=24, zh=132)

    cl_fid = call_cell(z, b, dNdz)
    cl_8 = call_cell(z, b, dNdz8)
    cl_4 = call_cell(z, b, dNdz4)
    
    fg, ax = plt.subplots(figsize=(6, 4))
    def plot(ax):
        ax.plot(z, dNdz, label='Fiducial', lw=3, alpha=0.2)
        ax.plot(z, dNdz8, label='k=8', lw=1,)
        ax.plot(z, dNdz4, label='k=4', lw=1, ls='--')

    plot(ax)
    ax.set(xlim=(-3, 5), xlabel='z', ylabel='dN/dz')
    ax.legend(loc=3)
    ax1 = fg.add_axes([0.19, 0.6, 0.23, 0.25])
    plot(ax1)
    ax1.set(xlim=(-0.01, 0.27), ylim=(-1, 6))
    ax2 = fg.add_axes([0.6, 0.6, 0.23, 0.25])
    plot(ax2)
    ax2.set(xlim=(1.3, 1.6), ylim=(-0.2, 1.6))
    fg.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/nztreat.pdf', bbox_inches='tight')    
    plt.show()
    
    ls = ['-', '-', '--']
    lw = [3, 1, 1]
    al = [0.2, 1., 1.]
    for i, (cl_i, ni) in enumerate(zip([cl_fid, cl_8, cl_4],
                                      ['Fiducial', 'K=8', 'K=4'])):
        plt.plot(1.0e5*cl_i[0], color='C%d'%i, ls=ls[i], lw=lw[i], alpha=al[i], label=ni)
        plt.plot(1.0e5*cl_i[1], color='C%d'%i, ls=ls[i], lw=lw[i], alpha=al[i])

    plt.ylabel(r'C$_{\ell}$ [x$10^{-5}$]')
    plt.xlabel(r'$\ell$')
    plt.legend()
    plt.loglog()
    plt.text(2, 2, r'$f_{\rm NL}=28.58$')
    plt.text(3., 0.5, r'$f_{\rm NL}=0$')
    plt.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/cell_nz.pdf', bbox_inches='tight')    
    

def plot_fnlbias():
    truth  = np.array([0.0,    76.92])
    meas0  = np.array([0.36,   77.67])
    meas1  = np.array([-11.64, 54.57])
    meas2  = np.array([-20.14, 38.38])
    meas3  = np.array([-26.91, 6.04])
    measc1 = np.array([-12.12, 54.01])
    measc2 = np.array([-20.97, 37.48])
    measc3 = np.array([-28.13, 4.59])
    
    p = np.load('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logbestfit_0_lrg_po100_desic_256_noweight.npz')
    z = np.load('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logbestfit_0_lrg_zero_desic_256_noweight.npz')
    zk = np.load('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logbestfit_0_lrg_zero_desic_256_dnnp_known1.npz')
    pk = np.load('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logbestfit_0_lrg_po100_desic_256_dnnp_known1.npz')        
    zkp = np.load('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logbestfit_0_lrg_zero_desic_256_dnnp_knownp.npz')
    pkp = np.load('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logbestfit_0_lrg_po100_desic_256_dnnp_knownp.npz')    
    zp = np.load('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logbestfit_0_lrg_zero_desic_256_dnnp_allp.npz')
    pp = np.load('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logbestfit_0_lrg_po100_desic_256_dnnp_allp.npz')    

    colors = [plt.cm.Dark2(i) for i in [0, 2, 3, 1]]        
    
    fig, ax = plt.subplots(figsize=(6, 4), sharex=True, sharey=True)
    
    ax.plot(meas0, truth, label='No weight', color=colors[0], ls='-')    
    i = 0
    for me,ne in zip([meas1, meas2, meas3],
                     ['Nonlinear Three Maps', 'Nonlinear Four Maps', 'Nonlinear Nine Maps']):
        ax.plot(me, truth, label=ne, ls='-', color=colors[i+1])
        i += 1
    lgn = ax.legend()
    for i, lg in enumerate(lgn.get_texts()):
        lg.set_color(colors[i])


    ax.plot(measc1, truth, ls='--', color=colors[1])
    ax.plot(measc2, truth, ls='--', color=colors[2])
    ax.plot(measc3, truth, ls='--', color=colors[3])
     
    ax.scatter(pk['params'][:, 0]/1.3, p['params'][:, 0]/1.3, alpha=0.1, marker='.', color=colors[1], zorder=-10)
    ax.scatter(zk['params'][:, 0]/1.3, z['params'][:, 0]/1.3, alpha=0.1, marker='.', color=colors[1], zorder=-10)    
    ax.scatter(pkp['params'][:, 0]/1.3, p['params'][:, 0]/1.3, alpha=0.1, marker='.', color=colors[2], zorder=-10)
    ax.scatter(zkp['params'][:, 0]/1.3, z['params'][:, 0]/1.3, alpha=0.1, marker='.', color=colors[2], zorder=-10)
    ax.scatter(pp['params'][:, 0]/1.3, p['params'][:, 0]/1.3, alpha=0.1, marker='.', color=colors[3], zorder=-10)
    ax.scatter(zp['params'][:, 0]/1.3, z['params'][:, 0]/1.3, alpha=0.1, marker='.', color=colors[3], zorder=-10)

    
    #plt.plot(1.15*meas1+14, truth, ls=':', color='C1', )
    #plt.plot(1.3*meas2+27, truth, ls=':', color='C2')

    ax.set_xlabel(r'Mitigated $f_{\rm NL}$')
    ax.set_ylabel(r'No mitigation, clean $f_{\rm NL}$')
    
    np.savetxt('/users/PHS0336/medirz90/github/dimagfnl/figures/fig9_fnlbias.txt',
              [p['params'][:, 0]/1.3, pk['params'][:, 0]/1.3, pkp['params'][:, 0]/1.3, pp['params'][:, 0]/1.3,
               z['params'][:, 0]/1.3, zk['params'][:, 0]/1.3, zkp['params'][:, 0]/1.3, zp['params'][:, 0]/1.3])
    fig.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/fnlbias.pdf', bbox_inches='tight')     
    plt.show()
    
    for me,ne in zip([meas0, meas1, meas2, meas3],
                     ['No weight', 'Nonlinear Three Maps', 'Nonlinear Four Maps', 'Nonlinear Nine Maps']):
        plt.plot(me, truth, label=ne, zorder=-10, alpha=0.3, lw=3, marker='o', mfc='w', ls='-')

    plt.plot(1.17*meas1+13.95, truth, ls=':', color='C1', lw=5)
    plt.plot(1.32*meas2+26.97, truth, ls=':', color='C2', lw=4)    
    plt.plot(2.35*meas3+63.50, truth, ls=':', color='C3', lw=6)    

    
    
    
def debias_mcmc():
    #knn1_s.samples[:, 0] = 1.17*knn1_s.samples[:, 0]+13.95
    #dskp_s.samples[:, 0] = 1.32*dskp_s.samples[:, 0]+26.97
    #dsp_s.samples[:, 0]  = 2.35*dsp_s.samples[:, 0]+63.50
    
    def debias(y, a, b):
        return a*y + b

    p = '/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/'
    debias_params = {'known1':[1.17, 13.95*1.3],
                     'knownp':[1.32, 26.97*1.3],
                     'allp':[2.35, 63.50*1.3]}

    for case in ['known1', 'knownp', 'allp']:

        d = np.load(f'{p}logmcmc_lrg_zero_desic_dnnp_{case}_steps10k_walkers50_elmin0_p1.0_s0.945.npz')
        dc = dict(d)
        dc['chain'][:, :, 0] = debias(d['chain'][:, :, 0], *debias_params[case])
        dc['best_fit'][0]  = debias(d['best_fit'][0], *debias_params[case])

        np.savez(f'{p}logmcmc_lrg_zero_desic_dnnp_{case}debiased_steps10k_walkers50_elmin0_p1.0_s0.945.npz', **dc)
        #print(f'{p}logmcmc_lrg_zero_desic_dnnp_{case}debiased_steps10k_walkers50_elmin0.npz')
        #print(d['chain'][5000:, :, 0].mean()/1.3)
        #print(dc['chain'][5000:, :, 0].mean()/1.3)    
        

def debias_mcmc_mocks():
    #knn1_s.samples[:, 0] = 1.17*knn1_s.samples[:, 0]+13.95
    #dskp_s.samples[:, 0] = 1.32*dskp_s.samples[:, 0]+26.97
    #dsp_s.samples[:, 0]  = 2.35*dsp_s.samples[:, 0]+63.50
    
    def debias(y, a, b):
        return a*y + b

    p = '/fs/ess/PHS0336/data/lognormal/v3/mcmc/'
    debias_params = {'known1':[1.17, 13.95*1.3],
                     'knownp':[1.32, 26.97*1.3],
                     'allp':[2.35, 63.50*1.3]}

    for fnl in ['zero', 'po100']:  
        for case in ['known1', 'knownp', 'allp']:

            d = np.load(f'{p}logmcmc_0_lrg_{fnl}_desic_256_dnnp_{case}_steps10k_walkers50.npz')
            dc = dict(d)
            dc['chain'][:, :, 0] = debias(d['chain'][:, :, 0], *debias_params[case])
            dc['best_fit'][0]  = debias(d['best_fit'][0], *debias_params[case])

            np.savez(f'{p}logmcmc_0_lrg_{fnl}_desic_256_dnnp_{case}debiased_steps10k_walkers50.npz', **dc)
            print(f'{p}logmcmc_0_lrg_{fnl}_desic_256_dnnp_{case}debiased_steps10k_walkers50.npz', end=' ')
            print(d['chain'][5000:, :, 0].mean()/1.3, end=' ')
            print(dc['chain'][5000:, :, 0].mean()/1.3, end=' ')                 
            print(fnl, case)
  
    
    

def plot_dr9vsmocks():
    p = '/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/'
    knn1 = np.load(f'{p}logmcmc_lrg_zero_desic_dnnp_known1_steps10k_walkers50.npz')
    bf = np.load('/fs/ess/PHS0336/data/lognormal/v3/mcmc/logbestfit_0_lrg_zero_desic_256_noweight.npz')
    fnl_m = bf['params'][:, 0]/ gratio
    fnl_d = knn1['best_fit'][0] / gratio
    print(fnl_d, knn1['best_fit'][0])
    
    fig, ax = plt.subplots(ncols=2, figsize=(12, 4), sharey=True)
    fig.subplots_adjust(wspace=0.05)

    ax[0].hist(2*bf['neglog']/1000., histtype='step', bins=18)
    ax[0].axvline(2*knn1['best_fit_logprob'], ls='--', color='C1')

    ax[1].hist(fnl_m, histtype='step', bins=18)
    ax[1].axvline(fnl_d, ls='--', color='C1')

    ax[0].set(xlabel=r'min $\chi^{2}$', yticks=[])
    ax[1].set(xlabel=r'$f_{\rm NL}$')

    ax[0].text(0.2, 0.75, 'Mocks', transform=ax[0].transAxes)
    ax[0].text(0.55, 0.75, 'DR9', color='C1', transform=ax[0].transAxes)

    fig.align_xlabels() 
    fig.savefig('/users/PHS0336/medirz90/github/dimagfnl/figures/pdf_dr9vsmocks.pdf', bbox_inches='tight')   
    
    
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

    kw = {'vmin':400, 'vmax':1200, 'cmap':dv.mycolor(), 'in_deg':True}
    dv.mollview(hp_known,  figax=[fig, ax0], **kw)
    dv.mollview(hp_known1, figax=[fig, ax1], **kw)
    dv.mollview(hp_all,    figax=[fig, ax2], **kw)
    dv.mollview(hp_nknown1, figax=[fig, ax3], 
                colorbar=True, galaxy=False, unit=r'Predicted density [deg$^{-2}$]', **kw)

    for ni, axi in zip(['Linear (E[B-V], Depth-z)', 'Linear (E[B-V], Depth-z, PSFsize-r)', 
                        'Linear (All Maps)', 'Nonlinear (E[B-V], Depth-z, PSFsize-r)'], 
                                      [ax0, ax1, ax2, ax3]):
        axi.text(0.15, 0.2, ni, transform=axi.transAxes, alpha=0.8)
    fig.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/npred.pdf', bbox_inches='tight')       
    

def plot_clmocks():    
    fig, ax = plt.subplots(ncols=2, sharey=True, sharex=True, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.05)
    colors = [plt.cm.Dark2(i) for i in [0, 2, 3, 1]]


    for i, fnl in enumerate(['czero', 'cpo100']):

        j = 0
        for meth, name in zip(['noweight', 'dnnp_known1', 'dnnp_knownp', 'dnnp_allp'],
                             ['No Weight', 'Nonlinear Three Maps', 'Nonlinear Four Maps', 'Nonlinear Nine Maps']):
            d = np.load(f'/fs/ess/PHS0336/data/lognormal/v3/clustering/logclmock_0_lrg_{fnl}_desic_256_{meth}_mean.npz')
            ln, = ax[i].plot(d['el_bin'], d['cl'], ls='--', color=colors[j], lw=1)
            fnl_ = fnl[1:]
            print(fnl_)
            d = np.load(f'/fs/ess/PHS0336/data/lognormal/v3/clustering/logclmock_0_lrg_{fnl_}_desic_256_{meth}_mean.npz')
            ax[i].plot(d['el_bin'], d['cl'], ls='-', color=ln.get_color(), label=name)
            j += 1

    for i, fnl in enumerate([r'$f_{\rm NL} = 0$', r'$f_{\rm NL} = 76.9$']):
        ax[i].text(0.5, 0.1, fnl, transform=ax[i].transAxes)

    ax[0].set(xscale='log', xlabel=r'$\ell$', ylabel=r'$\log C_{\ell}$')
    ax[1].set(xlabel=r'$\ell$')

    lgn = ax[1].legend()
    for i, txt in enumerate(lgn.get_texts()):
        txt.set_color(colors[i])

    fig.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/clmocks.pdf', bbox_inches='tight')       
    
    
def fnltime():
    
    fNL = [63, 105, -12, 46]
    fNLerr_lower = [331, 150, 39.0, 25]
    fNLerr_upper = [101, 90,  40.,  30]
    fNL_upper = np.array(fNL)+np.array(fNLerr_upper)
    fNL_lower = np.array(fNL)-np.array(fNLerr_lower)
    measurement = ['SDSS \n photo LRG', 'BOSS \n LRG DR9','eBOSS \n QSO DR16','DESI \n photo LRG\n (This work)']
    year = [2008, 2013, 2021, 2023]
    
    fNL_CMB = [38,32,2.7,-0.9]
    fNLerr_lower_CMB = [96,42,11.6,10.2]
    fNLerr_upper_CMB = [96,42,11.6,10.2]    
    fNL_upper_CMB = np.array(fNL_CMB)+np.array(fNLerr_upper_CMB)
    fNL_lower_CMB = np.array(fNL_CMB)-np.array(fNLerr_lower_CMB)
    measurement_CMB = ['WMAP 1','WMAP 7','Planck 13','Planck 18']
    year_CMB = [2003, 2010, 2013.5, 2018]  
        
    fig, ax = plt.subplots(figsize=(7, 5)) 
    #LSS measurement
    ax.errorbar(year, fNL, yerr=[fNLerr_lower,fNLerr_upper], fmt='o', color='C0', capsize=3, alpha=0.8)
    ax.scatter(year[-1], fNL[-1], s=200, marker='*', color='C0')
    
    #adjust label
    labelshift=[-40, 55, 60, -80]
    xshift=[-0.5, 1, 1, 1]
    for i, txt in enumerate(measurement):
        ax.annotate(txt, (year[i]-xshift[i], fNL[i]-fNLerr_lower[i]-labelshift[i]), color='C0',fontsize=17)   
        
    #CMB measurement    
    ax.errorbar(year_CMB,fNL_CMB,yerr=[fNLerr_lower_CMB,fNLerr_upper_CMB],fmt='x', color='C1', capsize=3, alpha=0.8)
    labelshift=[10, 7, 10, -40]
    xshift = [-2, -1.9, -0.2, -2]
    for i, txt in enumerate(measurement_CMB):
        ax.annotate(txt, (year_CMB[i]+xshift[i], fNL_CMB[i]+fNLerr_upper_CMB[i]+labelshift[i]),color='C1',fontsize=17)

    #lgn = ax.legend(['LSS','CMB'],frameon=False,fontsize=14)
    #txts = lgn.get_texts()
    #txts[0].set_color('C0')
    #txts[1].set_color('C1')
    
    #ax.fill_between(year,fNL_lower,fNL_upper,color='C2',alpha=0.05) 
    #ax.fill_between(year_CMB,fNL_lower_CMB,fNL_upper_CMB,color='C1',alpha=0.05)
    ax.set_ylabel('local PNG $f_\mathrm{NL}$')
    ax.set_xlabel('Publication year')
    ax.set_ylim(-300,220)
    ax.set_xlim(2000.5,2028)
    ax.axhline(0,color='grey',linestyle='dashed')
    plt.tight_layout()
    fig.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/fnl_history.pdf', bbox_inches='tight')       
    
    np.savetxt('/users/PHS0336/medirz90/github/dimagfnl/figures/fig15_fnl_history.txt',
              [year, fNL, fNL_lower, fNL_upper, 
               year_CMB, fNL_CMB, fNL_lower_CMB, fNL_upper_CMB])
    
    
    
def fnl_magbias():
    dirp='/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/'
    stg = {'mult_bias_correction_order':0,'smooth_scale_2D':0.15, 
           'smooth_scale_1D':0.3, 'contours': [0.68, 0.95]}
    mc_kw = dict(names=['fnl', 'b', 'n0'], 
                 labels=[r'f_{\rm NL}', 'b', '10^{7}n_{0}'], settings=stg) 
    stats = {}
    list_p = [0.55, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6]
    for p in list_p:
        d_ = MCMC(f'{dirp}logmcmc_lrg_zero_desic_dnnp_known1_steps10k_walkers50_elmin0_p{p}_s0.945.npz', mc_kw=mc_kw)
        stats[p] = d_.stats

    stats = pd.DataFrame(stats)    
    plt.plot(stats.iloc[0, :], marker='o')
    plt.fill_between(list_p, stats.iloc[4, :], stats.iloc[5, :], alpha=0.05)
    plt.fill_between(list_p, stats.iloc[2, :], stats.iloc[3, :], alpha=0.1, color='C0')
    plt.xlabel('p')
    plt.ylim(ymin=-10)
    plt.ylabel(r'$f_{\rm NL}$ + Mitigation Systematics')   
    plt.savefig(f'/users/PHS0336/medirz90/github/dimagfnl/figures/fnl_magbias.pdf', bbox_inches='tight')       
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