
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