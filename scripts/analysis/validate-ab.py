'''
    run the validation procedure on a dataset
    run it with
    mpirun -np 5 python validate.py 
'''
import os
import numpy as np
from time import time
import sys
HOME = os.getenv('HOME')
sys.path.append(HOME + '/github/DESILSS')
import NN
from scipy.stats import pearsonr

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def getcf(d):
    # lbl = ['ebv', 'nstar'] + [''.join((s, b)) for s in ['depth', 'seeing', 'airmass', 'skymag', 'exptime'] for b in 'rgz']
    cflist = []
    indices = []
    for i in range(d['train']['fold0']['features'].shape[1]):
        for j in range(5):
            fold = ''.join(['fold', str(j)])
            cf = pearsonr(d['train'][fold]['label'], d['train'][fold]['features'][:,i])[0]
            if np.abs(cf) >= 0.02:
                #print('{:s} : sys_i: {} : cf : {:.4f}'.format(fold, lbl[i], cf))
                indices.append(i)
                cflist.append(cf)
    if len(indices) > 0:
        indices = np.unique(np.array(indices))
        return indices
    else:
        print('no significant features')
        return None
#     cf = []
#     indices = []
#     for i in range(features.shape[1]):
#         cf.append(pearsonr(label, features[:,i]))
#         if np.abs(cf) > 0.0
   
def get_all(ablationlog):
    d = np.load(ablationlog).item()
    indices = None
    for il, l in enumerate(d['validmin']):
       m = (np.array(l) - d['RMSEall']) > 0.0
       #print(np.any(m), np.all(m))
       if np.all(m):
         #print(il, d['indices'][il])
         #print(il, [lbs[m] for m in d['indices'][il]])
         #break
         indices = d['indices'][il]
         break
       if (il == len(d['validmin'])-1) & (np.any(m)):
          indices = [d['indices'][il][-1]]       
    # return either None or indices
    return indices
 
   
    
if rank == 0:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rc('font', family='serif')
    #
    #
    import os 
    import fitsio as ft
    import healpy as hp
    from argparse import ArgumentParser
    ap = ArgumentParser(description='Neural Net regression')
    ap.add_argument('--input',  default='test_train_eboss_dr5-masked.npy')
    ap.add_argument('--output', default='test_train_eboss_dr5-maskednn.npy')
    ap.add_argument('--nside', default=256, type=int)
    ap.add_argument('--axfit', nargs='*', type=int,\
                                   default=[i for i in range(17)])
    ap.add_argument('--ablationlog', default='None')
    ns = ap.parse_args()
    NSIDE = ns.nside
    config = dict(nchain=10, batchsize=1024, nepoch=100, Units=[0],
              tol=0.0, scale=0.0, learning_rate=0.001)
    # scales : 1000, 100, 50, 20, 10, 1, 5
    #
    log  = '! ===== Regression with Neural Net ======\n'
    log += 'reading input : {} with nside : {} \n'.format(ns.input, NSIDE)
    log += 'the fiducial config is {}\n'.format(config)
    data   = np.load(ns.input).item()
    if ns.ablationlog == 'cf':
       axfit  = getcf(data)
    elif os.path.isfile(ns.ablationlog):
       axfit  = get_all(ns.ablationlog)
    elif ns.ablationlog == 'None':
       axfit  = ns.axfit
    else:
       RaiseError("axis is not set correctly!")
    oupath = ns.output
    if not os.path.exists(ns.output):
       os.makedirs(ns.output)
else:
    oupath = None
    data   = None
    config = None
    axfit  = None

data   = comm.bcast(data,   root=0)
oupath = comm.bcast(oupath, root=0)
config = comm.bcast(config, root=0)
axfit  = comm.bcast(axfit,  root=0)

if rank == 0:
    if axfit is None:
        print('no correction is required')
        print('will make a constant map ... ')
        oumap  = np.ones(12*NSIDE*NSIDE)
        hp.write_map(oupath+'nn-weights.hp'+str(NSIDE)+'.fits', oumap, fits_IDL=False, overwrite=True)
        sys.exit()
    else:
        print('will carry on with {} features '.format(axfit))
else:
    if axfit is None:
        sys.exit()
    else:
        print('will carry on ...')


assert size == 5, 'required number of mpi tasks should be equal to number of folds'
train_i = data['train']['fold'+str(rank)]
test_i  = data['test']['fold'+str(rank)]
valid_i = data['validation']['fold'+str(rank)]




if rank==0:log+='rank %d has train %d validation %d and test %d \n'%(rank, train_i.size, test_i.size, valid_i.size)
comm.Barrier()


#
# train Num of Layers
#
valid_min  = []
nlayers = [[0], [40], [20,20], [20, 10, 10], [10, 10, 10, 10]] #  
if rank==0:
     log+='start the hyper-parameter training with the initial config {}\n'.format(config)
     log+='training num of layers {} \n'.format(nlayers)
for nl_i in nlayers:
     t1 = time()
     config.update(Units=nl_i)
     net = NN.Netregression(train_i, valid_i, test_i, axfit)
     net.train_evaluate(**config)
     rmse = []
     for a in net.epoch_MSEs:
         rmse.append(np.sqrt(a[2][:,2]))
     RMSE  = np.column_stack(rmse)
     RMSEm = np.mean(RMSE, axis=1)
     #RMSEe = np.std(RMSE, axis=1)/np.sqrt(RMSE.shape[1])
     baseline = np.sqrt(net.optionsdic['baselineMSE'][1])
     #valid_rmse[str(nl_i)] = [net.epoch_MSEs[0][2][:,0], RMSEm/baseline, RMSEe/baseline]
     valid_min.append(np.min(RMSEm)/baseline)
     if rank==0:log+='rank{} finished {} in {} s\n'.format(rank, nl_i, time()-t1)
     #plt.plot(np.arange(RMSEm.size), RMSEm/baseline, label='{}'.format(nl_i))
comm.Barrier()
# #if rank ==0:
# #  plt.legend()
# #  plt.ylim(0.95, 1.05)
# #  plt.show() 
# #sys.exit()
# # valid_rmse = comm.gather(valid_rmse, root=0)
valid_min  = comm.gather(valid_min, root=0)

if rank == 0:
     VMIN = np.array(valid_min)
     Mean = VMIN.mean(axis=0)
     argbest = np.argmin(Mean)
     x = np.arange(VMIN.shape[1])
     plt.figure(figsize=(4,3))
     for i in range(VMIN.shape[0]):
         plt.plot(x, VMIN[i,:], color='grey', ls=':', alpha=0.5)
     plt.plot(x, Mean, color='red', ls='-', label='average across the folds')
     plt.axvline(x[argbest], ls='--', color='k', label='nlayers CV')
     plt.xticks(x, [str(l) for l in nlayers], rotation=45)
     plt.ylabel(r'$RMSE_{NN}/RMSE_{baseline}$')
     plt.legend()
     nlbest = nlayers[argbest]
     config.update(Units=nlbest)
     log += 'best nlayers is :: {}\n'.format(nlbest)
     log += 'the updated config is {}\n'.format(config)
     plt.savefig(oupath+'nlayers_validation.png', bbox_inches='tight', dpi=300)
else:
     config = None    
config = comm.bcast(config, root=0)
if rank == 0:print(config)
    

#
# train learning rate scale
#
valid_min  = []
lrates = np.array([0.0001, 0.001, 0.01])
if rank==0:log+='training learning rate {} \n'.format(lrates)
for lrate in lrates:
    t1 = time()
    config.update(learning_rate=lrate)
    net = NN.Netregression(train_i, valid_i, test_i, axfit)
    net.train_evaluate(**config)
    rmse = []
    for a in net.epoch_MSEs:
        rmse.append(np.sqrt(a[2][:,2]))
    RMSE  = np.column_stack(rmse)
    RMSEm = np.mean(RMSE, axis=1)
    #RMSEe = np.std(RMSE, axis=1)/np.sqrt(RMSE.shape[1])
    baseline = np.sqrt(net.optionsdic['baselineMSE'][1])
    valid_min.append(np.min(RMSEm)/baseline)
    if rank==0:log+='rank{} finished {} in {} s\n'.format(rank, lrate, time()-t1)
comm.Barrier()
valid_min  = comm.gather(valid_min, root=0)
if rank == 0:
    VMIN = np.array(valid_min)
    Mean = VMIN.mean(axis=0)
    argbest = np.argmin(Mean)
    plt.figure(figsize=(4,3))
    for i in range(VMIN.shape[0]):
        plt.plot(-np.log10(lrates), VMIN[i,:], color='grey', ls=':', alpha=0.5)
    #
    plt.plot(-np.log10(lrates), Mean, color='red', ls='-', label='average across the folds')
    plt.axvline(-np.log10(lrates[argbest]), ls='--', color='k', label=r'Learning Rate  CV')
    plt.ylabel(r'$RMSE_{NN}/RMSE_{baseline}$')
    plt.xlabel(r'$-\log(Learning Rate)$')
    plt.legend()    
    lrbest = lrates[argbest]
    config.update(learning_rate=lrbest)
    log += 'best learning rate is :: {}\n'.format(lrbest)
    log += 'the updated config is {}\n'.format(config)
    plt.savefig(oupath+'learningrate_validation.png', bbox_inches='tight', dpi=300)
else:
    config = None    
config = comm.bcast(config, root=0)    
if rank == 0:print(config)
#
# train batchsize
#
valid_min  = []
bsizes = np.array([512, 1024, 2048, 4096])
if rank==0:log+='training batchsize {} \n'.format(bsizes)
for bsize in bsizes:
     t1 = time()
     config.update(batchsize=bsize)
     net = NN.Netregression(train_i, valid_i, test_i, axfit)
     net.train_evaluate(**config)
     rmse = []
     for a in net.epoch_MSEs:
         rmse.append(np.sqrt(a[2][:,2]))
     RMSE  = np.column_stack(rmse)
     RMSEm = np.mean(RMSE, axis=1)
     #RMSEe = np.std(RMSE, axis=1)/np.sqrt(RMSE.shape[1])
     baseline = np.sqrt(net.optionsdic['baselineMSE'][1])
     valid_min.append(np.min(RMSEm)/baseline)
     if rank==0:log+='rank{} finished {} in {} s\n'.format(rank, bsize, time()-t1)
comm.Barrier()
valid_min  = comm.gather(valid_min, root=0)
if rank == 0:
     VMIN = np.array(valid_min)
     Mean = VMIN.mean(axis=0)
     argbest = np.argmin(Mean)
     plt.figure(figsize=(4,3))
     for i in range(VMIN.shape[0]):
         plt.plot(np.log10(bsizes), VMIN[i,:], color='grey', ls=':', alpha=0.5)

     plt.plot(np.log10(bsizes), Mean, color='red', ls='-', label='average across the folds')
     plt.axvline(np.log10(bsizes[argbest]), ls='--', color='k', label=r'Batch Size  CV')
     plt.ylabel(r'$RMSE_{NN}/RMSE_{baseline}$')
     plt.xlabel(r'$\log(Batchsize)$')
     plt.legend()    
     bsbest = bsizes[argbest]
     config.update(batchsize=bsbest)
     log +='best Batchsize is :: {}\n'.format(bsbest)
     log += 'the updated config is {}\n'.format(config)
     plt.savefig(oupath+'batchsize_validation.png', bbox_inches='tight', dpi=300)
else:
     config = None     
config = comm.bcast(config, root=0)        
if rank == 0:print(config)    

#
# train regularization scale
#
valid_min  = []
 # scales = np.exp([-50, -30, -20, -10, -5, 0])
scales = np.exp([-50, -20.0, -10.0, -5.0, -3.0, 1.0, 0.0, 1., 2., 5., 10., 15.])
# scales = np.exp([])
if rank==0:log+='training reg. scale {} \n'.format(scales)
for scale_i in scales:
     t1 = time()
     config.update(scale=scale_i)
     net = NN.Netregression(train_i, valid_i, test_i, axfit)
     net.train_evaluate(**config)
     rmse = []
     for a in net.epoch_MSEs:
         rmse.append(np.sqrt(a[2][:,2]))
     RMSE  = np.column_stack(rmse)
     RMSEm = np.mean(RMSE, axis=1)
     #RMSEe = np.std(RMSE, axis=1)/np.sqrt(RMSE.shape[1])
     baseline = np.sqrt(net.optionsdic['baselineMSE'][1])
     valid_min.append(np.min(RMSEm)/baseline)
     if rank==0: log+='rank{} finished {} in {} s\n'.format(rank, scale_i, time()-t1)

comm.Barrier()
valid_min  = comm.gather(valid_min, root=0)
if rank == 0:
     VMIN = np.array(valid_min)
     Mean = VMIN.mean(axis=0)
     argbest = np.argmin(Mean)
     plt.figure(figsize=(4,3))
     for i in range(VMIN.shape[0]):
         plt.plot(-np.log(scales), VMIN[i,:], color='grey', ls=':', alpha=0.5)
     #
     plt.plot(-np.log(scales), Mean, color='red', ls='-', label='average across the folds')
     plt.axvline(-np.log(scales[argbest]), ls='--', color='k', label=r'$\alpha$ CV')
     plt.ylabel(r'$RMSE_{NN}/RMSE_{baseline}$')
     plt.xlabel(r'$-\ln(\alpha)$')
     plt.legend()
     sclbest = scales[argbest]
     config.update(scale=sclbest)
     log +='best scale is :: {}\n'.format(sclbest)
     log += 'the updated config is {}\n'.format(config)
     plt.savefig(oupath + 'scales_validation.png', bbox_inches='tight', dpi=300)
else:
     config = None
config = comm.bcast(config, root=0)

if rank == 0:print(config)
#    
# train number of epochs
#
valid_min  = []
nepoch_max = 400
nepochs = np.arange(nepoch_max + 1)
config.update(nepoch=nepoch_max)
if rank==0:log+='training num of epochs up to {} \n'.format(nepoch_max)
t1 = time()
net = NN.Netregression(train_i, valid_i, test_i, axfit)
net.train_evaluate(**config)
rmse = []
for a in net.epoch_MSEs:
    rmse.append(np.sqrt(a[2][:,2]))
RMSE  = np.column_stack(rmse)
RMSEm = np.mean(RMSE, axis=1)
#RMSEe = np.std(RMSE, axis=1)/np.sqrt(RMSE.shape[1])
baseline = np.sqrt(net.optionsdic['baselineMSE'][1])
valid_min = RMSEm/baseline
if rank==0: log+='rank{} finished {} in {} s\n'.format(rank, nepochs[-1], time()-t1)

comm.Barrier()
valid_min  = comm.gather(valid_min, root=0)
if rank == 0:
     VMIN = np.array(valid_min)
     Mean = VMIN.mean(axis=0)
     argbest = np.argmin(Mean)
     plt.figure(figsize=(4,3))
     for i in range(VMIN.shape[0]):
         plt.plot(nepochs, VMIN[i,:], color='grey', ls=':', alpha=0.5)
     #
     plt.plot(nepochs, Mean, color='red', ls='-', label='average across the folds')
     plt.axvline(nepochs[argbest], ls='--', color='k', label=r'Number of Epochs  CV')
     plt.ylabel(r'$RMSE_{NN}/RMSE_{baseline}$')
     plt.xlabel(r'NumEpoch')
     plt.legend()    
     nebest = nepochs[argbest]
     config.update(nepoch=nebest)
     log +='best nepoch is :: {}\n'.format(nebest)
     log += 'the updated config is {}\n'.format(config)
     plt.savefig(oupath+'nepochs_validation.png', bbox_inches='tight', dpi=300)
else:
     config = None    
config = comm.bcast(config, root=0)
if rank == 0:print(config)

if rank==0:
   log+='final run for the best hyper-parameters\n'
   log+='BHPS: {}\n'.format(config)
t1    = time()
net   = NN.Netregression(train_i, valid_i, test_i, axfit)
net.train_evaluate(**config)
rmse  = []
for a in net.epoch_MSEs:
    rmse.append(np.sqrt(a[2][:,2]))
RMSE  = np.column_stack(rmse)
RMSEm = np.mean(RMSE, axis=1)
RMSEe = np.std(RMSE, axis=1)/np.sqrt(RMSE.shape[1])
baseline = np.sqrt(net.optionsdic['baselineMSE'][1])
rmselist = [net.epoch_MSEs[0][2][:,0], RMSEm/baseline, RMSEe/baseline]
meanY, stdY = net.optionsdic['stats']['ystat']
predP = net.test.P
#predY = stdY*net.test.Y.squeeze()+meanY # true input label
y_avg = []
for yi in net.chain_y:
    y_avg.append(yi[1].squeeze().tolist())    
predY = stdY*np.mean(np.array(y_avg), axis=0) + meanY
 


if rank==0: log+='finished final run for rank {} in {} s \n'.format(rank, time()-t1)

comm.Barrier()
rmselist = comm.gather(rmselist, root=0)
predY    = comm.gather(predY, root=0)
predP    = comm.gather(predP, root=0)
if rank ==0:
    hpix = np.concatenate(predP)
    ngal = np.concatenate(predY)
    oudata = np.zeros(ngal.size, dtype=[('hpix', 'i8'), ('weight','f8')])
    oudata['hpix']   = hpix
    oudata['weight'] = ngal
    oumap  = np.zeros(12*NSIDE*NSIDE)
    oumap[hpix] = ngal
    np.save(oupath+'nn-rmse', rmselist)
    ft.write(oupath+'nn-weights'+str(NSIDE)+'.fits', oudata)
    hp.write_map(oupath+'nn-weights.hp'+str(NSIDE)+'.fits', oumap, fits_IDL=False, overwrite=True)
    log += 'write rmses in {}\n'.format(oupath+'nn-rmse')
    log += 'write hpix.weight in {}\n'.format(oupath+'nn-weights'+str(NSIDE)+'.fits')
    log += 'write weight map in {}\n'.format(oupath+'nn-weights.hp'+str(NSIDE)+'.fits')
    #
    #
    plt.figure(figsize=(4,3))
    for l in range(len(rmselist)):
        x  = rmselist[l][0]
        y  = rmselist[l][1]
        ye = rmselist[l][2]
        plt.fill_between(x, y-ye, y+ye, color='grey', alpha=0.1)
        plt.plot(x, y, color='grey')
    plt.xlabel('Training Epoch')
    plt.ylabel(r'$RMSE_{NN}/RMSE_{baseline}$')
    plt.ylim(0.995, 1.005)
    plt.savefig(oupath+'nn-rmse.png', bbox_inches='tight', dpi=300)
    log += 'write the rmse plot in {}'.format(oupath+'nn-rmse.png')
    logfile = open(oupath+'nn-log.txt', 'w')
    logfile.write(log)
    #
    #
'''
t1 = time()
net = NN.Netregression(data['train']['fold0'], data['validation']['fold0'], data['test']['fold0'])
net.train_evaluate(nchain=1, batchsize=1000, nepoch=300, Units=[4, 4], learning_rate=1.e-3)
print('done in {} s'.format(time()-t1))

plt.plot(net.epoch_MSEs[0][2][:,0], net.epoch_MSEs[0][2][:,1], 'k-')
plt.plot(net.epoch_MSEs[0][2][:,0], net.epoch_MSEs[0][2][:,2], 'k--')
plt.ylabel('MSE')
plt.xlabel('Training epoch')
plt.ylim(1.0,1.5)


print('RMSE for baseline : %.3f NN : %.3f'%(np.sqrt(net.optionsdic['baselineMSE'][2]), np.sqrt(net.epoch_MSEs[0][1])))
'''
