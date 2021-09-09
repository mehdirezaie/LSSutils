
import numpy as np


def read_nnbar(filename):
    
    d_i = np.load(filename, allow_pickle=True)    
    err_mat = []    
    for i, d_ij in enumerate(d_i):
        err_mat.append(d_ij['nnbar']-1.0)
    
    return np.array(err_mat).flatten()

def read_nbmocks(list_nbars):
    
    err_mat = []    
    for nbar_i in list_nbars:
        
        err_i  = read_nnbar(nbar_i)
        err_mat.append(err_i)
        print('.', end='')

    err_mat = np.array(err_mat)
    print(err_mat.shape)
    return err_mat


def read_clx(fn, cl_ss=None, lmin=0):
    from .utils import histogram_cell
    #--- 
    cl = np.load(fn, allow_pickle=True).item()
    cl_x = []
    
    if cl_ss is None:
        cl_ss = []
        read_clss = True
    else:
        read_clss = False

    nsys = len(cl['cl_sg'])
    #lbins = np.arange(1, len(cl['cl_sg'][0]['cl']), 10)
    lbins = np.arange(1, 101, 10)
    #print(nsys)    
    for i in range(nsys):    
        l_, cl_sg_ = histogram_cell(cl['cl_sg'][i]['cl'], bins=lbins)
        if read_clss:
            _, cl_ss_ = histogram_cell(cl['cl_ss'][i]['cl'], bins=lbins)
            cl_ss.append(cl_ss_)
        else:
            cl_ss_ = cl_ss[i] 
            
        cl_x.append(cl_sg_[lmin:]*cl_sg_[lmin:]/cl_ss_[lmin:])    

    #print(l_[:lmax])
    return l_, np.array(cl_x).flatten(), cl_ss


def read_clxmocks(mocks, cl_ss, lmin=0):    
    #--- mocks
    clx = []
    for mock_i in mocks:
        clx_ = read_clx(mock_i, cl_ss=cl_ss, lmin=lmin)[1]   
        clx.append(clx_) 
        print('.', end='')
        
    err_tot = np.array(clx)
    return err_tot


def readnbodykit(filename):
    with open(filename, 'r') as infile:
        lines = infile.readlines()

        shotnoise = None
        values = []
        for line in lines:
            if '#shotnoise' in line:
                shotnoise = float(line.split(':')[-1])
                #print(f'shotnoise {shotnoise}')
            else:
                if line.startswith('#'):
                    continue
                else:
                    strings = line.split(' ')
                    values.append([float(s) for s in strings])
        if values != []:
            values = np.array(values)
        else:
            raise RuntimeError('reading failed')
    return values, shotnoise