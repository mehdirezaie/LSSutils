#
import os
home=os.getenv('HOME')
scratch = home
print('scratch : %s'%scratch)


import sys
sys.path.append(home + '/github/LSSutils')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import healpy as hp

from sklearn.feature_selection import f_regression, mutual_info_regression
from LSSutils.catalogs.combinefits import EBOSSCAT
from LSSutils.catalogs.datarelease import cols_eboss_v6_qso_simp as attrs_names

from sklearn import linear_model
from sklearn.model_selection import train_test_split

def lasso_regression(ZCUT, cap, verbose=False):
    sysmaps = pd.read_hdf(scratch + '/data/eboss/sysmaps/SDSS_HI_imageprop_nside256.h5')
    qso = EBOSSCAT([scratch + '/data/eboss/v6/eBOSS_QSO_clustering_'+cap+'_v6.dat.fits'],
                  ['weight_noz', 'weight_cp', 'weight_fkp']) # 
    qso.apply_zcut(zcuts=ZCUT)
    qso.project2hp(nside=256)

    ran = EBOSSCAT([scratch + '/data/eboss/v6/eBOSS_QSO_clustering_'+cap+'_v6.ran.fits'], 
                   weights=['weight_noz', 'weight_cp', 'weight_systot', 'weight_fkp'])
    ran.apply_zcut(zcuts=ZCUT)
    ran.project2hp(nside=256)
    
    x    = sysmaps.values
    m    = ran.galm != 0.0
    y    = np.zeros_like(qso.galm)*np.nan
    y[m] = qso.galm[m] / (ran.galm[m] * (qso.galm[m].sum()/ran.galm[m].sum()))


    mydata = pd.DataFrame(np.column_stack([x, y]),
                          columns=attrs_names + ['nqso'])
    mydata.dropna(inplace=True)
    # sns.heatmap(mydata.corr(), cmap=plt.cm.seismic, center=0.0, cbar_kws={'label':'PCC'})
    #
    #  scale the imaging attrs.
    #
    #print(mydata.shape)
    X = mydata.values[:, :-1]
    Y = mydata.values[:, 17]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.33, 
                                                        random_state=42)
    
    meanX, stdX = np.mean(X_train, axis=0), np.std(X_train, axis=0, ddof=1)
    meanY, stdY = np.mean(Y_train), np.std(Y_train, ddof=1)
    
    X_train = (X_train - meanX)/stdX
    X_test  = (X_test  - meanX)/stdX
    Y_train = (Y_train - meanY)/stdY
    Y_test  = (Y_test - meanY)/stdY    
    
    #alphas    = [0.0001, 0.001, 0.01, 0.1, 1.]
    mse_base  = np.mean((Y_test - meanY)**2)
    
    alphas    = np.logspace(-3, -1, num=12, endpoint=True)
    coef_list = []
    mse_list  = []
    for alpha_i in alphas:#, 0.001, 0.01, 0.1, 1.]:
        reg = linear_model.Lasso(alpha=alpha_i)
        reg.fit(X_train, Y_train)  
        Y_pred = reg.predict(X_test)
        mse = np.mean((Y_pred - Y_test)**2)
        mse_list.append(mse)
        coef_list.append(reg.coef_)
        if verbose:print('%.4f %.4f %.4f'%(alpha_i, mse, mse_base))
    return dict(alphas=alphas, coefs=np.column_stack(coef_list), mses=mse_list)

def elastic_regression(ZCUT, cap, verbose=False, nside=256):
    sysmaps = pd.read_hdf(scratch + '/data/eboss/sysmaps/SDSS_HI_imageprop_nside256.h5')
    qso = EBOSSCAT([scratch + '/data/eboss/v6_elnet/eBOSS_QSO_clustering_'+cap+'_v6.dat.fits'],
                  ['weight_noz', 'weight_cp'])
    qso.apply_zcut(zcuts=ZCUT)
    qso.project2hp(nside=256)

    ran = EBOSSCAT([scratch + '/data/eboss/v6_elnet/eBOSS_QSO_clustering_'+cap+'_v6.ran.fits'], 
                   weights=['weight_noz', 'weight_cp', 'weight_systot'])
    ran.apply_zcut(zcuts=[0.8, 2.2])
    ran.project2hp(nside=256)
    
    x    = sysmaps.values
    m    = ran.galm != 0.0
    y    = np.zeros_like(qso.galm)*np.nan
    y[m] = qso.galm[m] / (ran.galm[m] * (qso.galm[m].sum()/ran.galm[m].sum()))


    mydata = pd.DataFrame(np.column_stack([x, y]),
                          columns=attrs_names + ['nqso'])
    mydata.dropna(inplace=True)
    # sns.heatmap(mydata.corr(), cmap=plt.cm.seismic, center=0.0, cbar_kws={'label':'PCC'})
    #
    #  scale the imaging attrs.
    #
    #print(mydata.shape)
    hpix = mydata.index
    
    X = mydata.values[:, :-1]
    Y = mydata.values[:, 17]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.33, 
                                                        random_state=42)
    
    meanX, stdX = np.mean(X_train, axis=0), np.std(X_train, axis=0, ddof=1)
    meanY, stdY = np.mean(Y_train), np.std(Y_train, ddof=1)
    
    X_train = (X_train - meanX)/stdX
    X_test  = (X_test  - meanX)/stdX
    #Y_train = (Y_train - meanY)/stdY
    #Y_test  = (Y_test - meanY)/stdY    
    
    #alphas    = [0.0001, 0.001, 0.01, 0.1, 1.]
    mse_base  = np.mean((Y_test - meanY)**2)
    
    alphas    = np.logspace(-3, -1, num=12, endpoint=True)
    coef_list = []
    mse_list  = []
    for alpha_i in alphas:#, 0.001, 0.01, 0.1, 1.]:
        reg = linear_model.ElasticNet(alpha=alpha_i)
        reg.fit(X_train, Y_train)  
        Y_pred = reg.predict(X_test)
        mse = np.mean((Y_pred - Y_test)**2)
        mse_list.append(mse)
        coef_list.append(reg.coef_)
        if verbose:print('%.4f %.4f %.4f'%(alpha_i, mse, mse_base))
    # run with the best
    alpha_best = alphas[np.argmin(mse_list)]
    reg = linear_model.ElasticNet(alpha=alpha_best)
    reg.fit(X_train, Y_train)  
    Y_pred = reg.predict(X_test)
    mse = np.mean((Y_pred - Y_test)**2)    
    
    # make the weight maps 
    Xtot    = (X - meanX)/stdX
    wmap    = np.zeros(12*nside*nside)*np.nan
    wmap[mydata.index] = reg.predict(Xtot)
    results = dict(alphas=alphas, coefs=np.column_stack(coef_list), mses=mse_list,
                   mse_best=mse, mse_base=mse_base, coef_best=reg.coef_, alpha_best=alpha_best,
                   wmap=wmap)
    return results



def run(savemodel=None, savemaps=None, savefig=None, title='NGC', regression='elastic'):
    if regression == 'lasso':        
        results1 = lasso_regression([0.8, 1.1],   title, verbose=False)
        results2 = lasso_regression([1.1, 2.2],   title, verbose=False)    
    elif regression == 'elastic':
        results1 = elastic_regression([0.8, 1.1], title, verbose=False)
        results2 = elastic_regression([1.1, 2.2], title, verbose=False)    
        
    # plot the parameters
    colors = plt.cm.jet
    fig, ax = plt.subplots(nrows=2, figsize=(8, 10), sharex=True)
    plt.subplots_adjust(hspace=0.0)
    ax[0].set_title('%s-%s solid (0.8<z<1.1) dashed (1.5<z<2.2)'%(title, regression))
    for i in range(17):
        ax[0].plot(-np.log(results1['alphas']), results1['coefs'][i,:], 
                   label=attrs_names[i], c=colors(i/17), linestyle='-')
        ax[0].plot(-np.log(results2['alphas']), results2['coefs'][i,:], 
                   linestyle='--', c=colors(i/17))
    ax[0].legend(bbox_to_anchor=(1., 1.))
    ax[0].set_ylabel('parameters')

    ax[1].plot(-np.log(results1['alphas']), 1.e3*(results1['mses']/np.min(results1['mses'])-1), 'k-', label='0.8<z<1.1')
    ax[1].plot(-np.log(results2['alphas']), 1.e3*(results2['mses']/np.min(results2['mses'])-1), 'k--', label='1.1<z<2.2')
    ax[1].set_xlabel('- log(alpha)')
    ax[1].set_ylabel('1.e4 [ Validation MSE - 1]')
    ax[1].legend()
    if savefig is not None:plt.savefig(savefig, bbox_inches='tight')
    if savemodel is not None:
        results = {'low':{'z':[0.8, 1.1], 'model':results1}, 
                   'high':{'z':[1.1, 2.2], 'model':results2}}
        np.save(savemodel, results)
    if savemaps is not None:
        hp.write_map(savemaps['low'],  results1['wmap'], fits_IDL=False, overwrite=True)
        hp.write_map(savemaps['high'], results2['wmap'], fits_IDL=False, overwrite=True)       
        
def run_eboss_qso(title='NGC'):
    snside=str(256)
    path      = scratch + '/data/eboss/v6_elnet/'
    if not os.path.exists(path):
        raise RuntimeError('path not correct')
        print('will create %s'%path)        
        os.makedirs(path)
    savemodel = path + 'elnet_models_'+title+'.npy'
    savemaps  = {'low':path + 'nn-weights_'+title+'low.hp'+snside+'.fits',
                 'high':path + 'nn-weights_'+title+'high.hp'+snside+'.fits'}
    savefig   = path + 'params_valMSE'+title+'.png'
    run(savemodel=savemodel, savemaps=savemaps, savefig=savefig, title=title, regression='elastic')
        
if __name__ == '__main__':
    #
    title = sys.argv[1]
    if title not in ['NGC', 'SGC']:raise RuntimeError('%s not recognized'%title)
    print(title)
    run_eboss_qso(title)