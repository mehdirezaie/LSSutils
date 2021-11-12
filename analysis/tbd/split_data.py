'''
   code to read the log-normal mock data
   and split it into k-folds
   wo --random it will split data based on RA-DEC
   ie. equi-area regions
   
   Usage:
        mpirun -n 3 python split_data.py --path ../data/mocks/dr5mocks/ --ext 3dbox_nmesh1024_L5274.0_bias1.5_*
'''

try:
    #import healpy as hp
    import fitsio as ft  # moved to loop
except:
    print('warning! fitsio is not installed!')
from sklearn.model_selection import KFold
import numpy as np


def split_jackknife(hpix, weight, label, features, njack=20):
    '''
        split_jackknife(hpix, weight, label, features, njack=20)
        split healpix-format data into k equi-area regions
        hpix: healpix index shape = (N,)
        weight: weight associated to each hpix 
        label: label associated to each hpix
        features: features associate to each pixel shape=(N,M) 
    '''
    f = weight.sum() // njack
    hpix_L = []
    hpix_l = []
    frac_L = []
    frac    = 0
    label_L = []
    label_l = []
    features_L = []
    features_l = []
    w_L = []
    w_l = []
    #
    #
    for i in range(hpix.size):
        frac += weight[i]            
        hpix_l.append(hpix[i])
        label_l.append(label[i])
        w_l.append(weight[i])
        features_l.append(features[i])
        #
        #
        if frac >= f:
            hpix_L.append(hpix_l)
            frac_L.append(frac)
            label_L.append(label_l)
            w_L.append(w_l)
            features_L.append(features_l)
            frac    = 0
            features_l  = []
            w_l     = []
            hpix_l = []
            label_l = []
        elif i == hpix.size-1:
            hpix_L.append(hpix_l)
            frac_L.append(frac)
            label_L.append(label_l)
            w_L.append(w_l)
            features_L.append(features_l)            
    return hpix_L, w_L, label_L, features_L #, frac_L

def concatenate(A, ID):
    # combine A[i] regions for i in ID 
    AA = [A[i] for i in ID]
    return np.concatenate(AA)
    
def combine(hpix, fracgood, label, features, DTYPE, IDS):
    # uses concatenate(A,ID) to combine different attributes
    size = np.sum([len(hpix[i]) for i in IDS])
    zeros = np.zeros(size, dtype=DTYPE)
    zeros['hpix']     = concatenate(hpix, IDS)
    zeros['fracgood'] = concatenate(fracgood, IDS)
    zeros['features'] = concatenate(features, IDS)
    zeros['label']    = concatenate(label, IDS)
    return zeros

    
def split2KfoldsSpatially(data, k=5, shuffle=True, random_seed=123):
    '''
        split data into k contiguous regions
        for training, validation and testing
    '''
    P, W, L, F = split_jackknife(data['hpix'],data['fracgood'],
                                data['label'], data['features'], 
                                 njack=k)
    DTYPE = data.dtype
    np.random.seed(random_seed)
    kfold = KFold(k, shuffle=shuffle, random_state=random_seed)
    index = np.arange(k)
    kfold_data = {'test':{}, 'train':{}, 'validation':{}}
    arrs = P, W, L, F, DTYPE
    for i, (nontestID, testID) in enumerate(kfold.split(index)):
        foldname = 'fold'+str(i)
        validID  = np.random.choice(nontestID, size=testID.size, replace=False)
        trainID  = np.setdiff1d(nontestID, validID)
        kfold_data['test'][foldname]       = combine(*arrs, testID)
        kfold_data['train'][foldname]      = combine(*arrs, trainID)
        kfold_data['validation'][foldname] = combine(*arrs, validID)
    return kfold_data    




def split2Kfolds(data, k=5, shuffle=True, random_seed=123):
    '''
        split data into k randomly chosen regions
        for training, validation and testing
    '''
    np.random.seed(random_seed)
    kfold = KFold(k, shuffle=shuffle, random_state=random_seed)
    index = np.arange(data.size)
    kfold_data = {'test':{}, 'train':{}, 'validation':{}}
    for i, (nontestID, testID) in enumerate(kfold.split(index)):
        #
        #
        foldname = 'fold'+str(i)
        validID  = np.random.choice(nontestID, size=testID.size, replace=False)
        trainID  = np.setdiff1d(nontestID, validID)
        #
        #
        kfold_data['test'][foldname]       = data[testID]
        kfold_data['train'][foldname]      = data[trainID]
        kfold_data['validation'][foldname] = data[validID]
    return kfold_data

def read_write(path2file, path2output, k, random=True):
    ''' 
    read path2file, splits the data either randomly or ra-dec
    then writes the data onto path2output
    '''
    DATA  = ft.read(path2file)
    if random:
        datakfolds = split2Kfolds(DATA, k=k)
    else:
        datakfolds = split2KfoldsSpatially(DATA, k=k)
    np.save(path2output, datakfolds)

    
def loop_filenames(filenames, k, random):
    '''
    loop over filenames and split them into k folds 
    contiguously or randomly
    ''' 
    import fitsio as ft
    for file in filenames:
        fn = file.split('/')[-1]
        inputf  = file
        outputf = file[:-5]+'.'+str(k)+'fold.fits'
        read_write(inputf, outputf, k, random) 
        
if __name__ == '__main__':
    # mpi
    from mpi4py import MPI

    # get the size, and the rank of each mpi task
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        from glob import glob
        from argparse import ArgumentParser
        ap = ArgumentParser(description='Read BigFile mocks and write .dat')
        ap.add_argument('--path',   default='/global/cscratch1/sd/mehdi/mocks/3dbox/')
        ap.add_argument('--ext', default='*') 
        ap.add_argument('--k', default=5, type=int)
        ap.add_argument('--random', action='store_true', default=False) 
        ns = ap.parse_args()
        FILES = glob(ns.path+ns.ext)
        K = ns.k
        random = ns.random
        print('split %d data files, randomly T/F : '%len(FILES), random)
    else:
        FILES = None
        K = None
        random = None

    # bcast FILES
    FILES = comm.bcast(FILES, root=0)
    K = comm.bcast(K, root=0)
    random = comm.bcast(random, root=0)


    #
    # distribute files on different task ids
    # chunksize
    nfiles = len(FILES)

    if np.mod(nfiles, size) == 0:
        chunksize = nfiles // size
    else:
        chunksize = nfiles // size + 1

    my_i      = rank*chunksize
    if rank*chunksize + chunksize > nfiles:
        my_end = None
    else:
        my_end    = rank*chunksize + chunksize
    my_chunk = FILES[my_i:my_end]


    #print('files on rank {} are {}'.format(rank, my_chunk))
    
    # for filei in my_chunk:
    #     print(filei.split('/')[-1])


    #
    # read data and split
    #
    loop_filenames(my_chunk, K, random)
