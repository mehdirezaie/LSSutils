import numpy as np
import lssutils.utils as ut
import matplotlib.pyplot as plt

from glob import glob

from lssutils.io import read_nnbar, read_nbmocks


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

def get_chi2_cov(err_tot, covmax):
    nmocks, nbins = err_tot.shape
    hartlapf = (nmocks - 1.) / (nmocks - nbins - 2.) # leave-one-out
    print(f'nmocks: {nmocks}, nbins: {nbins}')
    
    chi2s = []
    for i in range(nmocks):
        
        nbar_ = err_tot[i, :]
        
        invcov_ = np.linalg.inv(covmax*hartlapf)
        
        chi2_ = ut.chi2_fn(nbar_, invcov_)
        chi2s.append(chi2_)       
    return chi2s 


def plot_clx_chi2():

    ell_edges = ut.ell_edges[:10]
    print(f'ell edges: {ell_edges}')
    err_null = read_clxmocks(['/fs/ess/PHS0336/data/lognormal/v3/clustering/clmock_0_%d_lrg_zero_desic_256_noweight.npy'%i for i in range(1, 1000)], ell_edges)
    err_kno1 = read_clxmocks(['/fs/ess/PHS0336/data/lognormal/v3/clustering/clmock_0_%d_lrg_zero_desic_256_dnnp_known1.npy'%i for i in range(1, 1000)], ell_edges)
    err_knop = read_clxmocks(['/fs/ess/PHS0336/data/lognormal/v3/clustering/clmock_0_%d_lrg_zero_desic_256_dnnp_knownp.npy'%i for i in range(1, 1000)], ell_edges)
    err_allp = read_clxmocks(['/fs/ess/PHS0336/data/lognormal/v3/clustering/clmock_0_%d_lrg_zero_desic_256_dnnp_allp.npy'%i for i in range(1, 1000)], ell_edges)

    covmax = np.cov(err_null, rowvar=False)

    chi2_null = ut.get_chi2pdf(err_null)
    chi2_kno1 = get_chi2_cov(err_kno1, covmax)
    chi2_knop = get_chi2_cov(err_knop, covmax)
    chi2_allp = get_chi2_cov(err_allp, covmax)
    
    chi2data = [79.3, 70.9, 49.1]
    cnmean = np.mean(chi2_null)

    plt.figure()
    plt.scatter(chi2_null, chi2_kno1, alpha=0.2, marker='.', label='Three Maps')
    plt.scatter(chi2_null, chi2_knop, alpha=0.2, marker='.', label='Four Maps')
    plt.scatter(chi2_null, chi2_allp, alpha=0.2, marker='.', label='Nine Maps')
    
    for i, chi2_i in enumerate(chi2data):
        plt.scatter(cnmean, chi2_i, color='C%d'%i)

    plt.plot([10, 600], [10, 600])
    plt.xscale('log')
    plt.xlabel('chi2 null')
    plt.ylabel('chi2 null,mitigated')
    plt.legend()
    plt.yscale('log')
    plt.savefig('chi2_mocks.pdf', bbox_inches='tight')



def plot_nbar_chi2():
    err_null = read_nbmocks(['/fs/ess/PHS0336/data/lognormal/v3/clustering/nbarmock_0_%d_lrg_zero_desic_256_noweight.npy'%i for i in range(1, 1000)])
    err_kno1 = read_nbmocks(['/fs/ess/PHS0336/data/lognormal/v3/clustering/nbarmock_0_%d_lrg_zero_desic_256_dnnp_known1.npy'%i for i in range(1, 1000)])
    err_knop = read_nbmocks(['/fs/ess/PHS0336/data/lognormal/v3/clustering/nbarmock_0_%d_lrg_zero_desic_256_dnnp_knownp.npy'%i for i in range(1, 1000)])
    err_allp = read_nbmocks(['/fs/ess/PHS0336/data/lognormal/v3/clustering/nbarmock_0_%d_lrg_zero_desic_256_dnnp_allp.npy'%i for i in range(1, 1000)])

    covmax = np.cov(err_null, rowvar=False)

    chi2_null = ut.get_chi2pdf(err_null)
    chi2_kno1 = get_chi2_cov(err_kno1, covmax)
    chi2_knop = get_chi2_cov(err_knop, covmax)
    chi2_allp = get_chi2_cov(err_allp, covmax)
    
    chi2data = [74.3, 73.2, 39.7]
    cnmean = np.mean(chi2_null)

    plt.figure()
    plt.scatter(chi2_null, chi2_kno1, alpha=0.2, marker='.', label='Three Maps')
    plt.scatter(chi2_null, chi2_knop, alpha=0.2, marker='.', label='Four Maps')
    plt.scatter(chi2_null, chi2_allp, alpha=0.2, marker='.', label='Nine Maps')
    
    for i, chi2_i in enumerate(chi2data):
        plt.scatter(cnmean, chi2_i, color='C%d'%i)


    plt.plot([30, 180], [30, 180])
    plt.xscale('log')
    plt.xlabel('chi2 null')
    plt.ylabel('chi2 null,mitigated')
    plt.legend()
    plt.yscale('log')
    plt.savefig('chi2_mocks_nbar.pdf', bbox_inches='tight')

plot_clx_chi2()
plot_nbar_chi2()



