from __future__ import division, print_function
import numpy as np  # legendary numpy package
import healpy as hp # healpy to pixelize sky
import fitsio as ft # fitsio to read/write `fits` tables
from time import time
import sys

def find1neighbor(ipix, nside):
    theta, phi = hp.pix2ang(nside, ipix)
    # first neighbor
    nbrs = hp.get_all_neighbours(nside, theta, phi=phi).T
    return nbrs


def update(nlst, ipix, nside):
    lsl = []
    for ipi in ipix:
        ls = find1neighbor(ipi, nside).tolist()
        for lsi in ls:
            if (lsi not in nlst) & (lsi != -1):
                nlst.append(lsi)
                lsl.append(lsi)
    return nlst, lsl

def sortTP(nlst, nside=4096):
    nn = len(nlst)
    thetaphi = hp.pix2ang(nside, nlst)
    tp = np.zeros(nn, dtype=[('theta','f8'),('phi', 'f8')])
    tp['theta'] = thetaphi[0]
    tp['phi'] = thetaphi[1]
    tp_theta = np.sort(tp,order='theta')
    tp_tp = np.sort(tp_theta, order='phi')
    lnipix = []
    size_tp = tp_tp.size
    for i in range(size_tp):
        nipix = hp.ang2pix(nside, tp_tp['theta'][i], tp_tp['phi'][i])
        lnipix.append(nipix)
    return lnipix


def find_neighbors(ipix, nside=4096, m=1):
    nlst = [ipix]
    lst = [ipix]
    for mi in range(m):
        nlst, lst = update(nlst, lst, nside)
    return sortTP(nlst, nside)


def ipix2neighbors_array(ipix, nside=4096, layer=1):
    nnghbrs = (2*layer+1)**2
    cpix = nnghbrs // 2
    nghbrs_list = []
    npix = len(ipix)
    for i,ip in enumerate(ipix):
        nghbrs = find_neighbors(ip, nside=nside, m=layer)
        nghbrs_list.append(nghbrs)
    nghbrs_array = np.array(nghbrs_list).reshape(npix, nnghbrs)
    return nghbrs_array

def neighbors_mask(nghbrs_array, ipix_total):
    nghbrs_mask = np.in1d(nghbrs_array,ipix_total).reshape(nghbrs_array.shape)
    comp_neighbors = np.where(np.all(nghbrs_mask, axis=1))[0]
    return comp_neighbors


def extract_features(input_data_sample, input_data_all, nside=4096, layer=1):
    #
    #
    input_data_ipix_sample = input_data_sample[:,0].astype('i8')
    input_data_ipix_all = input_data_all[:,0].astype('i8')
    #
    #
    nghbrs_array =  ipix2neighbors_array(input_data_ipix_sample, nside=nside, layer=layer)
    comp_neighbors = neighbors_mask(nghbrs_array, input_data_ipix_all)
    comp_nghbrs_array = nghbrs_array[comp_neighbors]
    #
    #
    pixid_center = input_data_sample[comp_neighbors, 0].astype('i8')
    ngal_center = input_data_sample[comp_neighbors, -1].astype('i8')
    #
    #
    # print ("pixels that have complete neighbors :", comp_neighbors)
    # print ("healpix-id of those pixels :", pixid_center)
    # print ("Ngal in each pixel : ", ngal_center)
    nfeat = 9  # of one pixel
    nsamples, nneigh = comp_nghbrs_array.shape # including the center 3x3
    nfeat_total = nfeat * nneigh
    data_all = np.zeros((12*nside**2, nfeat))
    data_all[input_data_all[:,0].astype('i8'),:] = input_data_all[:,1:-1]
    # print("Num of features and samples :", nfeat_total, nsamples)
    tot_feat = data_all[comp_nghbrs_array[:,:]].reshape((nsamples, nfeat_total))
    return pixid_center, ngal_center, tot_feat

def main(sys):
    try:
        filename = sys.argv[1]
        layers = int(sys.argv[2])
    except:
        filename = 'dr3-ipix-depthrgz-seeingrgz-airmassrg-ebv-ngal.fits.gz'
        layers = 1
        
    s1 = time()
    data_row = ft.read(filename)
    s2 = time()
    print("{} ms for reading the file".format(s2-s1))
    print("total number of pixels : ", data_row.shape)
    s3 = time()
    hpix, ngal, features = extract_features(data_row, data_row, layer=layers)
    s4 = time()
    print("{} ms for extracting the file".format(s4-s3))
    nsamples, nfeat = features.shape
    y = np.zeros(nsamples, dtype=[('ngal','i8'),('hpix', 'i8'),('features','f8',(nfeat,))])
    y['ngal'] = ngal
    y['hpix'] = hpix
    y['features'] = features
    np.savez_compressed('layersall-'+str(layers), y)

if __name__ == '__main__':
    main(sys)
