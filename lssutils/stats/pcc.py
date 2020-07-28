import numpy as np
from scipy.stats import spearmanr, pearsonr

def PCC(xc, yc, kind='spearman'):
    if not kind in ['pearson', 'spearman']:
        raise ValueError(f'{kind} not defined')
    elif kind == 'pearson':
        func = pearsonr
    elif kind == 'spearman':
        func = spearmanr
        
    pcc = []
    for j in range(xc.shape[1]):
        pcc.append(func(xc[:,j], yc)[0])
    return pcc

def BTPCC(xc, yc, num=100, verbose=False):
    np.random.seed(123456)
    pcc = []
    for _ in range(num):
        pcc.append(PCC(xc, np.random.permutation(yc)))
        if verbose:
            print('.',end='')
    return pcc