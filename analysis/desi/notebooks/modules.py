import sys
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import fitsio as ft
import healpy as hp
from glob import glob
from time import time
from scipy.optimize import minimize
import pandas as pd

HOME = os.getenv('HOME')
print(f'running on {HOME}')
sys.path.append(f'{HOME}/github/LSSutils')
sys.path.append(f'{HOME}/github/sysnetdev')
import sysnet.sources as src

from lssutils.dataviz import setup_color, add_locators, mollview, mycolor
from lssutils.utils import (histogram_cell, maps_dr9, make_hp,
                            chi2_fn, get_chi2pdf, get_inv, hpix2radec, shiftra, make_overdensity)
from lssutils.io import (read_nbmocks, read_nnbar, read_clx, read_clxmocks, 
                         read_clmocks, read_window, read_window, read_chain)
from lssutils.theory.cell import (dNdz_model, init_sample, SurveySpectrum, Spectrum, bias_model_lrg)
from lssutils.extrn.mcmc import Posterior
from lssutils.extrn import corner
from lssutils.stats.window import WindowSHT
from lssutils.stats.pcc import pcc

import getdist
from getdist import plots, MCSamples

class MCMC(MCSamples):
     def __init__(self, path_to_mcmc, read_kw=dict(), mc_kw=dict()):
            self.stats, chains = read_chain(path_to_mcmc, **read_kw)
            MCSamples.__init__(self, samples=chains, **mc_kw)
            
def plot_ngmoll():
    desi = ft.read('/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_desi_256.fits')
    ng  = make_hp(256, desi['hpix'], desi['label']/(desi['fracgood']*hp.nside2pixarea(256, True)), np.inf)
    print(np.mean(ng[desi['hpix']]))
    mollview(ng, 400, 1200, r'Observed Galaxy Density [deg$^{-2}$]', 
                 cmap='YlOrRd_r', colorbar=True, galaxy=True)
    plt.savefig('figs/nlrg.pdf', bbox_inches='tight')
    
def plot_nz():
    nz = np.loadtxt('/fs/ess/PHS0336/data/rongpu/sv3_lrg_dndz_denali.txt')

    fg, ax = plt.subplots()

    ax.step(nz[:, 0], nz[:, 2], where='pre', lw=1)#, label='dN/dz')
    ax.set(xlim=(-0.05, 1.45), xlabel='z', ylabel='dN/dz')
    ax.text(0.25, 0.7, 'dN/dz', color='C0', transform=ax.transAxes)

    ax1 = ax.twinx()
    z_g = np.linspace(0.1, 1.35)

    ax1.plot(z_g, 1.4*bias_model_lrg(z_g), 'C1--', lw=3, alpha=0.5, zorder=10)#, label='b(z)')
    ax1.text(0.7, 0.48, 'b(z)$\propto$ D$^{-1}$(z)', color='C1', transform=ax1.transAxes)
    ax1.set_ylabel('b(z)')
    ax1.set_ylim((1.3, 3.1))

    #ax1.legend(loc='upper right')
    #ax.legend(loc='upper left')

    fg.savefig('figs/nz_lrg.pdf', bbox_inches='tight')    
    

def plot_model():
    bm = np.load('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_bmzls_noweight_steps10k_walkers50.npz')
    zbdndz = init_sample(kind='lrg')
    # read survey geometry
    dt = ft.read(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/tables/0.57.0/nlrg_features_bmzls_256.fits')
    w = np.zeros(12*256*256)
    w[dt['hpix']] = 1.0
    weight = hp.ud_grade(w, 1024)
    mask = weight > 0.5

    model = Spectrum()
    model.add_tracer(*zbdndz, p=1.0)
    model.add_kernels(np.arange(2000))


    wind = WindowSHT(weight, mask, np.arange(2048), ngauss=2048)
    fnl, b, noise = bm['best_fit']
    el_g = np.arange(2000)
    cl_bf = model(el_g, fnl=fnl, b=b, noise=noise)

    cl_bfw = wind.convolve(el_g, cl_bf)
    lmax = max(el_g)+1
    cl_bfwic = wind.apply_ic(cl_bfw[:lmax])

    cl_ = np.load('/fs/ess/PHS0336/data/lognormal/v2/clustering/clmock_zero_bmzls_mean.npz')
    cl_cov_ = np.load('/fs/ess/PHS0336/data/lognormal/v2/clustering/clmock_zero_bmzls_cov.npz')

    el_edges = cl_['el_edges']
    el = cl_['el_bin']
    cl = cl_['cl']
    cl_err = np.diagonal(cl_cov_['clcov']/1000.)**0.5

    cl_models = {}
    for name, cl_i in zip(['Best Fit Model', '+ Window Convolution', '+ Integral Constraint'],
                          [cl_bf, cl_bfw[:2000], cl_bfwic[:2000]]):

        cl_models[name] = histogram_cell(el_g, cl_i, bins=el_edges)


    fig = plt.figure(figsize=(5, 5), constrained_layout=False)
    gs = GridSpec(3, 1, figure=fig)

    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[2, 0])

    f = 1.0e5
    lw = [0.8, 0.8, 3.]
    ls = ['-', '-', '-']
    al = [1., 1., 0.7]
    for i, (n, v) in enumerate(cl_models.items()):
        kw = dict(label=n, lw=lw[i], ls=ls[i], alpha=al[i])
        ax1.plot(v[0], f*v[1], **kw)
        ax2.plot(el, v[1]/cl, **kw)

    ax1.plot(el, f*cl, 'C0--', label='Mean of Mocks')
    ax2.axhline(1.0, color='C0', ls='--')
    ax2.fill_between(el, 1-cl_err/cl, 1+cl_err/cl, alpha=0.2)


    ax1.legend(ncol=1)
    ax1.set(xscale='log', ylabel=r'$10^{5}C_{\ell}$')
    ax1.tick_params(labelbottom=False)
    ax2.set(xscale='log', xlabel=r'$\ell$', ylabel='Ratio', xlim=ax1.get_xlim(), ylim=(0.95, 1.10))

    fig.subplots_adjust(hspace=0.0, wspace=0.02)
    fig.align_labels()

    fig.savefig('figs/model_window.pdf', bbox_inches='tight')        
    
def plot_mcmc_mocks():
    stg = {'mult_bias_correction_order':0,'smooth_scale_2D':0.15, 'smooth_scale_1D':0.3, 'contours': [0.68, 0.95]}
    mc_kw = dict(names=['fnl', 'b', 'n0'], 
                 labels=['f_{NL}', 'b', '10^{7}n_{0}'], settings=stg) 

    read_kw = dict(ndim=3, iscale=[2])
    mc_kw2 = dict(names=['fnl', 'b', 'n0', 'b2', 'n02'], 
                  labels=['f_{NL}', 'b1', '10^{7}n_{0}', 'b2', '10^{7}n_{0}2'], settings=stg)
    read_kw2 = dict(ndim=5, iscale=[2, 4])
    mc_kw3 = dict(names=['fnl', 'b', 'n0', 'b2', 'n02', 'b3', 'n03'], 
                  labels=['f_{NL}', 'b1', '10^{7}n_{0}', 'b2', '10^{7}n_{0}2', 'b3', '10^{7}n_{0}3'], settings=stg)
    read_kw3 = dict(ndim=7, iscale=[2, 4, 6])
    fs = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_fullsky_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    bm = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_bmzls_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    nd = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_ndecals_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    sd = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_sdecals_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    bn = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_bmzlsndecals_noweight_steps10k_walkers50.npz', mc_kw=mc_kw2, read_kw=read_kw2)
    joint = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_bmzlsndecalssdecals_noweight_steps10k_walkers50.npz', mc_kw=mc_kw3, read_kw=read_kw3)

    stats = {}
    stats['Full Sky'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_fullsky_noweight_steps10k_walkers50.npz', **read_kw)[0]
    stats['F. Sky [BMzLS scaled]'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_fullskyscaled_noweight_steps10k_walkers50.npz', **read_kw)[0]
    stats['BASS/MzLS'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_bmzls_noweight_steps10k_walkers50.npz', **read_kw)[0]
    stats['DECaLS North'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_ndecals_noweight_steps10k_walkers50.npz', **read_kw)[0]
    stats['DECaLS South'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_sdecals_noweight_steps10k_walkers50.npz', **read_kw)[0]

    stats['NGC'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_ngc_noweight_steps10k_walkers50.npz', **read_kw)[0]
    stats['Joint (BMzLS+DECaLS N)'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_bmzlsndecals_noweight_steps10k_walkers50.npz', **read_kw2)[0]

    stats['Joint (DECaLS N+DECaLS S)'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_ndecalssdecals_noweight_steps10k_walkers50.npz', **read_kw2)[0]
    stats['DESI'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_desi_noweight_steps10k_walkers50.npz', **read_kw)[0]
    stats['Joint (BMzLS+DECaLS)'] = read_chain('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_zero_bmzlsndecalssdecals_noweight_steps10k_walkers50.npz', **read_kw3)[0]    

    # Triangle plot
    g = plots.get_single_plotter(width_inch=4*1.5)
    g.settings.legend_fontsize = 14

    g.plot_2d([bm, nd, sd, fs], 'fnl', 'b', filled=False)
    g.add_x_marker(0)
    g.add_y_marker(1.43)
    g.add_legend(['BASS/MzLS', 'DECaLS North', 'DECaLS South', 'Full Sky'], colored_text=True, legend_loc='lower left')
    #prsubplots(g.subplots[0, 0].get_xlim())
    # g.subplots[0, 0].set_xticklabels([])
    g.fig.align_labels()
    g.fig.savefig('figs/fnl2dmocks_area.pdf', bbox_inches='tight')    
    
    # Triangle plot
    g = plots.get_single_plotter(width_inch=4*1.5)
    g.settings.legend_fontsize = 14

    g.plot_1d([bm, bn, joint], 'fnl', filled=False)
    g.add_x_marker(0)
    # g.add_y_marker(1.4262343145500318)
    g.add_legend(['BASS/MzLS', 'BASS/MzLs+DECaLS North', 'BASS/MzLS+DECaLS (North+South)'], colored_text=True)

    g.subplots[0, 0].set_xticks([-100, -75, -50, -25, 0, 25., 50.])
    # g.subplots[0, 0].set_xticklabels([])
    g.fig.align_labels()
    g.fig.savefig('figs/fnl1dmocks_joint.pdf', bbox_inches='tight')    
    
    pstats = pd.DataFrame(stats,
                      index=['MAP [scipy]', 'MAP [chain]', 'Mean [chain]',
                             'Median [chain]', '16th', '84th']).T
    
    bm = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_po100_bmzls_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    nd = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_po100_ndecals_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
    sd = MCMC('/fs/ess/PHS0336/data/lognormal/v2/mcmc/mcmc_lrg_po100_sdecals_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)    

    # Triangle plot
    g = plots.get_single_plotter(width_inch=4*1.5)
    g.settings.legend_fontsize = 14

    g.plot_2d([bm, nd, sd], 'fnl', 'b', filled=False)
    g.add_x_marker(100)
    g.add_y_marker(1.43)
    g.add_legend(['BASS/MzLS', 'DECaLS North', 'DECaLS South'], colored_text=True, legend_loc='lower left')
    #prsubplots(g.subplots[0, 0].get_xlim())
    # g.subplots[0, 0].set_xticklabels([])
    g.fig.align_labels()
    g.fig.savefig('figs/fnl2dmocks_po100.pdf', bbox_inches='tight')    
    return pstats

def add_scale(ax):
    xx = ax.get_xticks()
    dx = xx[1]-xx[0]
    y = ax.get_yticks()[2]
    ax.arrow(xx[-3], y, dx, 0, )
    ax.annotate(f"{dx:.0f}", (xx[-3]+0.25*dx, y))
    ax.set_xticklabels([])
    
    
def plot_mcmc_dr9methods():
    
    titles = {'noweight':'No Correction', 
             'nn_known':'Conservative',
             'nn_all':'Extreme'}

    stg = {'mult_bias_correction_order':0,'smooth_scale_2D':0.15, 'smooth_scale_1D':0.3, 'contours': [0.68, 0.95]}
    mc_kw = dict(names=['fnl', 'b', 'n0'], 
                 labels=['f_{NL}', 'b', '10^{7}n_{0}'], settings=stg) 

    read_kw = dict(ndim=3, iscale=[2])
    mc_kw2 = dict(names=['fnl', 'b', 'n0', 'b2', 'n02'], 
                  labels=['f_{NL}', 'b1', '10^{7}n_{0}', 'b2', '10^{7}n_{0}2'], settings=stg)
    read_kw2 = dict(ndim=5, iscale=[2, 4])
    mc_kw3 = dict(names=['fnl', 'b', 'n0', 'b2', 'n02', 'b3', 'n03'], 
                  labels=['f_{NL}', 'b1', '10^{7}n_{0}', 'b2', '10^{7}n_{0}2', 'b3', '10^{7}n_{0}3'], settings=stg)
    read_kw3 = dict(ndim=7, iscale=[2, 4, 6])

    xlim = None

    with PdfPages('figs/fnl2d_dr9_methods.pdf') as pdf:

        for r in ['noweight', 'nn_known', 'nn_all']:

            noweight = MCMC(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/mcmc_lrg_zero_bmzls_{r}_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
            nnknown = MCMC(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/mcmc_lrg_zero_ndecals_{r}_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
            nnall = MCMC(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/mcmc_lrg_zero_sdecals_{r}_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)


            # Triangle plot
            g = plots.get_single_plotter(width_inch=4*1.5)
            g.settings.legend_fontsize = 14

            g.plot_2d([noweight, nnknown, nnall], 'fnl', 'b', filled=True)
            g.add_legend(['BASS/MzLS', 'DECaLS North', 'DECaLS South'], colored_text=True, title=titles[r])
            g.subplots[0, 0].set_ylim((1.23, 1.6))
            add_scale(g.subplots[0, 0])
            g.fig.align_labels()
            pdf.savefig(bbox_inches='tight')
            
def plot_mcmc_dr9regions():
    titles = {'bmzls':'BASS/MzLS', 
             'ndecals':'DECaLS North',
             'sdecals':'DECaLS South'}

    stg = {'mult_bias_correction_order':0,'smooth_scale_2D':0.15, 'smooth_scale_1D':0.3, 'contours': [0.68, 0.95]}
    mc_kw = dict(names=['fnl', 'b', 'n0'], 
                 labels=['f_{NL}', 'b', '10^{7}n_{0}'], settings=stg) 

    read_kw = dict(ndim=3, iscale=[2])
    mc_kw2 = dict(names=['fnl', 'b', 'n0', 'b2', 'n02'], 
                  labels=['f_{NL}', 'b1', '10^{7}n_{0}', 'b2', '10^{7}n_{0}2'], settings=stg)
    read_kw2 = dict(ndim=5, iscale=[2, 4])
    mc_kw3 = dict(names=['fnl', 'b', 'n0', 'b2', 'n02', 'b3', 'n03'], 
                  labels=['f_{NL}', 'b1', '10^{7}n_{0}', 'b2', '10^{7}n_{0}2', 'b3', '10^{7}n_{0}3'], settings=stg)
    read_kw3 = dict(ndim=7, iscale=[2, 4, 6])

    xlim = None

    with PdfPages('figs/fnl2d_dr9_regions.pdf') as pdf:

        for r in ['bmzls', 'ndecals', 'sdecals']:

            noweight = MCMC(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/mcmc_lrg_zero_{r}_noweight_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
            nnknown = MCMC(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/mcmc_lrg_zero_{r}_nn_known_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
            nnall = MCMC(f'/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/mcmc_lrg_zero_{r}_nn_all_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)


            # Triangle plot
            g = plots.get_single_plotter(width_inch=4*1.5)
            g.settings.legend_fontsize = 14
            g.plot_2d([noweight, nnknown, nnall], 'fnl', 'b', filled=True)
            g.add_legend(['No Correction', 'Conservative', 'Extreme'], colored_text=True, title=titles[r])
            g.subplots[0, 0].set_ylim((1.23, 1.6))
            add_scale(g.subplots[0, 0])
            g.fig.align_labels()
            pdf.savefig(bbox_inches='tight')
            
            
            
def plot_mcmc_dr9joint():
    path_ = '/fs/ess/PHS0336/data/rongpu/imaging_sys/mcmc/0.57.0/'
    titles = {'nn_known':'Conservative', 
             'nn_all':'Extreme'}

    stg = {'mult_bias_correction_order':0,'smooth_scale_2D':0.15, 'smooth_scale_1D':0.3, 'contours': [0.68, 0.95]}
    mc_kw = dict(names=['fnl', 'b', 'n0'], 
                 labels=['f_{NL}', 'b', '10^{7}n_{0}'], settings=stg) 

    read_kw = dict(ndim=3, iscale=[2])
    mc_kw2 = dict(names=['fnl', 'b', 'n0', 'b2', 'n02'], 
                  labels=['f_{NL}', 'b1', '10^{7}n_{0}', 'b2', '10^{7}n_{0}2'], settings=stg)
    read_kw2 = dict(ndim=5, iscale=[2, 4])
    mc_kw3 = dict(names=['fnl', 'b', 'n0', 'b2', 'n02', 'b3', 'n03'], 
                  labels=['f_{NL}', 'b1', '10^{7}n_{0}', 'b2', '10^{7}n_{0}2', 'b3', '10^{7}n_{0}3'], settings=stg)
    read_kw3 = dict(ndim=7, iscale=[2, 4, 6])

    xlim = None

    with PdfPages('figs/fnl2d_dr9_joint.pdf') as pdf:
        for r in ['nn_known', 'nn_all']:
            noweight = MCMC(f'{path_}mcmc_lrg_zero_bmzls_{r}_steps10k_walkers50.npz', mc_kw=mc_kw, read_kw=read_kw)
            nnknown = MCMC(f'{path_}mcmc_lrg_zero_bmzlsndecals_{r}_steps10k_walkers50.npz', mc_kw=mc_kw2, read_kw=read_kw2)
            nnall = MCMC(f'{path_}mcmc_lrg_zero_bmzlsndecalssdecals_{r}_steps10k_walkers50.npz', mc_kw=mc_kw3, read_kw=read_kw3)

            # Triangle plot
            g = plots.get_single_plotter(width_inch=4*1.5)
            g.settings.legend_fontsize = 14

            g.plot_1d([noweight, nnknown, nnall], 'fnl', filled=False)
            g.add_legend(['BASS/MzLS', '+ DECaLS North', '+ DECaLS (North + South)'], 
                         colored_text=True, title=titles[r], legend_loc='lower left')
            add_scale(g.subplots[0, 0])

            g.fig.align_labels()
            pdf.savefig(bbox_inches='tight')