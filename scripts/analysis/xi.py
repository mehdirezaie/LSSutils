'''
    code that calculates the auto/cross correlation functions
    using healpix based estimator
    
    created 4/3/2018 - Mehdi Rezaie
'''
import numpy as np
from   counter import paircount
from   time    import time
import healpy as hp

def split_jackknife(theta, phi, weight, delta, sysm, njack=20):
    f = weight.sum() // njack
    theta_L = []
    theta_l = []
    phi_L = []
    phi_l = [] 
    frac_L = []
    frac    = 0
    delta_L = []
    delta_l = []
    sysm_L = []
    sysm_l = []
    w_L = []
    w_l = []
    #
    #
    for i in range(theta.size):
        frac += weight[i]            
        theta_l.append(theta[i])
        phi_l.append(phi[i])
        delta_l.append(delta[i])
        w_l.append(weight[i])
        sysm_l.append(sysm[i])
        #
        #
        if frac >= f:
            theta_L.append(theta_l)
            phi_L.append(phi_l)
            frac_L.append(frac)
            delta_L.append(delta_l)
            w_L.append(w_l)
            sysm_L.append(sysm_l)
            frac    = 0
            sysm_l  = []
            w_l     = []
            theta_l = []
            phi_l   = []
            delta_l = []
        elif i == theta.size-1:
            theta_L.append(theta_l)
            phi_L.append(phi_l)
            frac_L.append(frac)
            delta_L.append(delta_l)
            w_L.append(w_l)
            sysm_L.append(sysm_l)            
    return theta_L, phi_L, w_L, delta_L, sysm_L #, frac_L


def hpupgrade(mapin, res_o, res_i):
    """
        same as hp.ud_grade
    """
    #ipix       = np.arange(12*res_o**2)  # 1, 2, .... 12*nside**2 
    #theta, phi = hp.pix2ang(res_o, ipix) # radian
    #ipix_o     = hp.ang2pix(res_i, theta, phi)
    #return mapin[ipix_o]
    return hp.ud_grade(mapin, res_o)

class XI_JACK(object):
    def __init__(self, elgmap, ranmap, select_fun, mask, njack=20):
        # set up the delta
        self.nside = hp.get_nside(elgmap)
        delta = np.zeros(elgmap.size)
        randc = ranmap * select_fun
        sf    = (elgmap[mask].sum() / randc[mask].sum())
        delta[mask] = elgmap[mask] / randc[mask] / sf - 1
        #
        # weight
        w = ranmap[mask]
        theta, phi  = hp.pix2ang(self.nside, np.argwhere(mask).flatten())
        thetal, phil, wl, deltal,_ = split_jackknife(theta, phi, w, delta[mask], delta[mask], njack=njack)
        self.theta  = thetal
        self.phi    = phil
        self.weight = wl
        self.delta  = deltal
        self.mean1  = np.average(delta[mask], weights=w)
        
    def run(self):
        bw    = hp.nside2resol(self.nside)
        bins  = np.arange(bw, np.deg2rad(10), bw)[::-1]
        njack = len(self.theta)
        print('njack ', njack)
        self.result = dict()
        t_i = time()
        for m in range(njack):
            for n in range(m, njack):
                if n == m:
                    auto = 1
                else:
                    auto = 0
                index = str(m)+'-'+str(n)                
                # m
                t1  = self.theta[m].copy()
                p1  = self.phi[m].copy()
                w1  = self.weight[m].copy()
                d1  = self.delta[m].copy()
                # n
                t2  = self.theta[n].copy()
                p2  = self.phi[n].copy()
                w2  = self.weight[n].copy()
                d2  = self.delta[n].copy()
                _, self.result[index] = XI(t1, p1,t2, p2, d1, d2, w1, w2, bins, auto)
        t_f = time()
        print(t_f-t_i, ' secs for computing the auto and cross among regions')
        #t_i = time()
        #for m in range(1, njack):
        #    for n in range(m):
        #        self.result[str(m)+'-'+str(n)] = np.copy(self.result[str(n)+'-'+str(m)])
        #t_f = time()
        #print(t_f-t_i, ' secs for finalizing using the symmetry')
        t_i = time()
        nbins = len(bins)-1
        wa = np.zeros(nbins)
        wb = np.zeros(nbins)
        for key_i in self.result.keys():
            wa += self.result[key_i][0]
            wb += self.result[key_i][1]
        self.xiall  = wa/wb
        #
        #
        t_f = time()
        print(t_f-t_i, ' secs for computing the overall paircount')
        #
        #
        t_i = time()
        xijackl = []
        for i in range(njack):
            wa = np.zeros(nbins)
            wb = np.zeros(nbins)
            for key_i in self.result.keys():
                if str(i) not in key_i.split('-'):
                    wa += self.result[key_i][0]
                    wb += self.result[key_i][1]
            xijackl.append(wa/wb)
        #
        #
        t_f = time()
        print(t_f-t_i, ' secs for computing the overall paircount')
        self.bins = bins
        #
        #
        var = np.zeros(nbins)
        for i in range(njack):
            var += (self.xiall - xijackl[i])**2
        var *= (njack-1)/njack
        self.xi_err = np.sqrt(var)
        self.output = dict(alls=self.result, t=self.bins,
                            njack=njack, w=self.xiall, werr=self.xi_err, 
                            wjacks=xijackl, nside=self.nside, dmean=self.mean1)

class XI_JACK_cross(object):
    def __init__(self, elgmap, ranmap, select_fun, sysm, mask, njack=20):
        self.nside = hp.get_nside(elgmap)
        delta = np.zeros(elgmap.size)
        randc = ranmap * select_fun
        sf    = (elgmap[mask].sum() / randc[mask].sum())
        delta[mask] = elgmap[mask] / randc[mask] / sf - 1
        #
        #
        delta2 = np.zeros(elgmap.size)
        sf2  = (sysm[mask]*ranmap[mask]).sum() / ranmap[mask].sum()
        #delta2[mask] = sysm[mask] / (sf2 * ranmap[mask]) - 1.0
        delta2[mask] = sysm[mask] / (sf2) - 1.0
        #
        #
        w = ranmap[mask]
        theta, phi = hp.pix2ang(self.nside, np.argwhere(mask).flatten())
        thetal, phil, wl, deltal, delta2l = split_jackknife(theta, phi, w, delta[mask], delta2[mask], njack=njack)
        self.theta  = thetal
        self.phi    = phil
        self.weight = wl
        self.delta  = deltal
        self.delta2 = delta2l
        self.mean1  = np.average(delta[mask], weights=w)
        self.mean2  = np.average(delta2[mask], weights=w)
        
    def run(self):
        #bw = 3.*hp.nside2resol(self.nside)*180./3.1416  # 3x resol.
        #bins = np.arange(bw, 10, bw)
        bw    = hp.nside2resol(self.nside)
        bins  = np.arange(bw, np.deg2rad(10), bw)[::-1]
        njack = len(self.theta)
        print('njack ', njack)
        self.result = dict()
        t_i = time()
        auto = 0
        for m in range(njack):
            for n in range(njack):
                index = str(m)+'-'+str(n)                
                # m
                t1  = self.theta[m].copy()
                p1  = self.phi[m].copy()
                w1  = self.weight[m].copy()
                d1  = self.delta[m].copy()
                # n
                t2  = self.theta[n].copy()
                p2  = self.phi[n].copy()
                w2  = self.weight[n].copy()
                d2  = self.delta2[n].copy()
                _, self.result[index] = XI(t1, p1,t2, p2, d1, d2, w1, w2, bins, auto)
        t_f = time()
        print(t_f-t_i, ' secs for computing the auto and cross among regions')
        #t_i = time()
        #for m in range(1, njack):
        #    for n in range(m):
        #        self.result[str(m)+'-'+str(n)] = np.copy(self.result[str(n)+'-'+str(m)])
        #t_f = time()
        #print(t_f-t_i, ' secs for finalizing using the symmetry')
        t_i = time()
        nbins = len(bins)-1
        wa = np.zeros(nbins)
        wb = np.zeros(nbins)
        for key_i in self.result.keys():
            wa += self.result[key_i][0]
            wb += self.result[key_i][1]
        self.xiall  = wa/wb
        t_f = time()
        print(t_f-t_i, ' secs for computing the overall paircount')
        #
        #
        t_i = time()
        xijackl = []
        for i in range(njack):
            wa = np.zeros(nbins)
            wb = np.zeros(nbins)
            for key_i in self.result.keys():
                if str(i) not in key_i.split('-'):
                    wa += self.result[key_i][0]
                    wb += self.result[key_i][1]
            xijackl.append(wa/wb)

        t_f = time()
        print(t_f-t_i, ' secs for computing the overall paircount')
        self.bins = bins
        #
        #
        var = np.zeros(nbins)
        for i in range(njack):
            var += (self.xiall - xijackl[i])**2
        var *= (njack-1)/njack
        self.xi_err = np.sqrt(var)
        self.output = dict(alls=self.result, t=self.bins,
                            njack=njack, w=self.xiall, werr=self.xi_err, 
                            wjacks=xijackl, nside=self.nside, dmean1=self.mean1, dmean2=self.mean2)

class XI_simple_cross(object):
    def __init__(self, elgmap, ranmap, sysmap, select_fun, mask):
        self.nside = hp.get_nside(elgmap)
        # set up the delta
        delta = np.zeros(elgmap.size)
        randc = ranmap * select_fun
        sf    = (elgmap[mask].sum() / randc[mask].sum())
        delta[mask] = elgmap[mask] / randc[mask] / sf - 1
        #
        # set up the delta
        delta2 = np.zeros(elgmap.size)
        sf2  = (sysmap[mask]*ranmap[mask]).sum() / ranmap[mask].sum()
        delta2[mask] = sysmap[mask] / (sf2) - 1.0
        #
        # weight
        w = ranmap[mask]
        theta, phi = hp.pix2ang(self.nside, np.argwhere(mask).flatten())
        self.theta  = theta
        self.phi    = phi
        self.weight = w
        self.delta  = delta[mask]
        self.delta2  = delta2[mask]
        self.mean1  = np.average(delta[mask], weights=w)
        self.mean2  = np.average(delta2[mask], weights=w)
        
    def run(self):
        #bw = 3.*hp.nside2resol(self.nside)*180./3.1416  # 3x resol.
        #bins = np.arange(bw, 10, bw)
        bw    = hp.nside2resol(self.nside)
        bins  = np.arange(bw, np.deg2rad(10), bw)[::-1]
        #ti = time()
        _,xi = XI(self.theta, self.phi, self.theta, self.phi, self.delta, self.delta2, self.weight, self.weight, bins, 0)
        #print('took {}s for the auto correlation '.format(time()-ti))
        self.bins = bins
        self.output = dict(t=self.bins, w=xi, dmean1=self.mean1, dmean2=self.mean2)

class XI_simple(object):
    def __init__(self, elgmap, ranmap, select_fun, mask):
        self.nside = hp.get_nside(elgmap)
        # set up the delta
        delta = np.zeros(elgmap.size)
        randc = ranmap * select_fun
        sf    = (elgmap[mask].sum() / randc[mask].sum())
        delta[mask] = elgmap[mask] / randc[mask] / sf - 1
        #
        # weight
        w = ranmap[mask]
        theta, phi = hp.pix2ang(self.nside, np.argwhere(mask).flatten())
        self.theta  = theta
        self.phi    = phi
        self.weight = w
        self.delta  = delta[mask]
        self.mean1  = np.average(delta[mask], weights=w)
        
    def run(self):
        #bw = 3.*hp.nside2resol(self.nside)*180./3.1416  # 3x resol.
        #bins = np.arange(bw, 10, bw)
        bw    = hp.nside2resol(self.nside)
        bins  = np.arange(bw, np.deg2rad(10), bw)[::-1]

        #ti = time()
        _,xi = XI(self.theta, self.phi, self.theta, self.phi, self.delta, self.delta, self.weight, self.weight, bins, 1)
        #print('took {}s for the auto correlation '.format(time()-ti))
        self.bins = bins
        self.output = dict(t=self.bins, w=xi, dmean=self.mean1)

def XI(theta1, phi1, theta2, phi2, delta1, delta2, weight1, weight2, bins, auto):
    t1 = time()
    cosbins = np.cos(bins)
    w = paircount(theta1, phi1, theta2, phi2, delta1, delta2, weight1, weight2, cosbins, auto)
    print('Finished hp Xi(theta) in %.2f secs'%(time()-t1))
    #binc = 0.5*(bins[1:]+bins[:-1])
    return bins, (w[0], w[1])

def run_XI(ouname, elgmap, ranmap, select_fun, mask, sysm=None, njack=20):
    if njack==0:
        if sysm is None:            
            xijack = XI_simple(elgmap, ranmap, select_fun, mask)
            xijack.run()
        else:
            xijack = XI_simple_cross(elgmap, ranmap, sysm, select_fun, mask)
            xijack.run()
    else:        
        if sysm is not None:
            xijack = XI_JACK_cross(elgmap, ranmap, select_fun, sysm, mask, njack=njack)
            xijack.run()
        else:
            xijack = XI_JACK(elgmap, ranmap, select_fun, mask, njack=njack)
            xijack.run()
    np.save(ouname, xijack.output)
    print('output is save in {}'.format(ouname))
        
def check_nside(map_in, res_out):
    res_in = hp.get_nside(map_in)
    if res_in != res_out:
        print('upgrading the map from {} to {}'.format(res_in, res_out))
        return hp.ud_grade(map_in, res_out)
    else:
        return map_in
    
if __name__ == '__main__':
    import os
    import sys
    import fitsio as ft
    from argparse import ArgumentParser
    #
    # command arguments
    ap = ArgumentParser(description='XI routine: healpix based xi estimator')
    ap.add_argument('--galmap')
    ap.add_argument('--ranmap')
    ap.add_argument('--mask')
    ap.add_argument('--njack',  default=20,  type=int)
    ap.add_argument('--nside',  default=256, type=int)
    ap.add_argument('--oudir',  default='./')
    ap.add_argument('--sysmap', default='none') # for cross-correlation
    ap.add_argument('--ouname', default='xi-eboss-dr5')
    ap.add_argument('--selection', default='none')
    ns = ap.parse_args()
    #
    #
    log = 'running the healpix based paircount XI \n'
    log += 'njack : {}   nside : {}\n'.format(ns.njack, ns.nside) 
    # check if output directory is there
    if not os.path.exists(ns.oudir):
        log  += 'creating the directory {} ...\n'.format(ns.oudir)
        os.makedirs(ns.oudir)
    #
    # check if selection function is given
    if not ns.selection in ['none', 'None', 'NONE']:
        log += 'selection function : {}\n'.format(ns.selection)
        select_fun_i = hp.read_map(ns.selection)
        select_fun   = check_nside(select_fun_i, ns.nside) # check nside
    else:
        log += 'uniform selection function is used!!!\n'
        select_fun = np.ones(12*ns.nside**2)                  # uniform selection mask
    #
    # if a systematic is given, 
    # it computes the cross correlation
    if not ns.sysmap in ['none', 'None', 'NONE']:
        log += 'computing the cross-correlation against {}\n'.format(ns.sysmap)
        sysm_i = hp.read_map(ns.sysmap)
        sysm   = check_nside(sysm_i, ns.nside)
    else:
        log += 'computing the auto correlation\n'
        sysm   = None
    #
    # read galaxy, random and mask maps
    galm_i = hp.read_map(ns.galmap)
    galm   = check_nside(galm_i, ns.nside)
    ranm_i = hp.read_map(ns.ranmap)
    ranm   = check_nside(ranm_i, ns.nside)
    mask_i = hp.read_map(ns.mask)
    mask   = check_nside(mask_i, ns.nside).astype('bool') # should be boolean
    #
    #
    log += 'galaxy hp map : {}\n'.format(ns.galmap)
    log += 'random hp map : {}\n'.format(ns.ranmap)
    log += 'mask   hp map : {}\n'.format(ns.mask)
    #
    # run and save
    path = ns.oudir + ns.ouname + '_nside_' + str(ns.nside) + '_njack_' + str(ns.njack)
    log += 'output under {}'.format(path)
    print(log)
    run_XI(path, galm, ranm, select_fun, mask, sysm=sysm, njack=ns.njack)
