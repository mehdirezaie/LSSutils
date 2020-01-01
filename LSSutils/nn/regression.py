# --- modules
import matplotlib.pyplot as plt
import tensorflow as tf                    
import numpy
import time
import os
import logging

    


class NetRegression(object):
    '''
    Perform Regression with Feed Forward Neural Networks
    
    parameters
    ----------
    train : class Data 
    has x, y, w, and p attributes
    valid : class Data
    test  : class Data
    
    
    
    
    Example:
    
        data  = np.load('/Users/mehdi/Downloads/trunk/qso.ngc.all.hp.256.r.npy', allow_pickle=True).item()
        train = regression.Data(data['train']['fold0'])
        valid = regression.Data(data['validation']['fold0'])
        test  = regression.Data(data['test']['fold0'])



        t_i = time.time()
        Net = regression.NetRegression(train, valid, test)
        #Net.fit_w_hparam_training()
        Net.fit(predict=True, min_delta=1.e-8,
                batch_size=1024, units=[10, 10],
                learning_rate=0.1)
        Net._descale() # descale
        Net.make_plots()
        t_f = time.time()
        print(f'took {t_f-t_i} secs')
    '''        
    
    logger = logging.getLogger('NetRegression')
    
    def __init__(self, train, valid, test):         
        self.train   = train
        self.valid   = valid
        self.test    = test
                
        for data in [train, valid, test]:
            for attr in ['x', 'y', 'w', 'p']:
                if not hasattr(data, attr):
                    raise AttributeError('Inputs must have a ``%s`` attribute'%attr)
                    
        self._scale()    
        
    def fit_w_hparam_training(self, options=None):
        
        if options is None:            
            '''
                only trains the number of layers,
                L2 regularization,
                and mini-batch size
                
            '''            
            options = {
                       'units':[[40], [20,20], [20, 10, 10], [10, 10, 10, 10]],
                       'scale':[0.0001, 0.001, 0.01, 0.1, 1., 10, 100, 1000],
                       'batch_size':[128, 256, 512, 1024, 2048, 4096]
                       }


        self.fit()
        hyperparams_dict    = []
        hyperparams_dict.append({key:getattr(self, key).copy()\
                                 for  key in ['attrs', 'total_val_loss']})
        attrs        = getattr(self, 'attrs')
        val_loss_min = getattr(self, 'total_val_loss')
        self.logger.info(f'start hyperparameter training with {attrs}')
        
        for key, values in options.items():
            parameter_best = attrs[key] # hold the current `best`

            for value in values:
                self.logger.info(f'training with {key} : {value}')
                attrs.update({key:value})

                # --- perform training
                self.fit(**attrs)
                if self.total_val_loss < val_loss_min:
                    val_loss_min    = self.total_val_loss
                    parameter_best  = value
                hyperparams_dict.append({key:getattr(self, key)\
                                         for  key in ['attrs', 'total_val_loss']})

            # --- determine the best parameter
            attrs.update({key:parameter_best})
            self.logger.info(f'select {key} : {parameter_best} as the best')
        
        # --- final set of hyper-parameters
        self.logger.info(f'final set of hyperparameters : {attrs}')        
        self.hyperparams_dict = hyperparams_dict
        
        
        # --- final run
        attrs.update(nchain=5)
        attrs.update(predict=True)
        self.logger.info(f'final training with {attrs}')
        self.fit(**attrs)
        
        
    def fit(self, 
           learning_rate=0.01,
           batch_size=256, 
           nepochs=500,
           nchain=1,
           units=[20, 20], 
           min_delta=1.e-5, 
           scale=0.0,
           patience=10, 
           verbose=0,
           global_seed=123567,
           model_dir='model/',
           save_model=False,
           verbose_early=0,
           verbose_reduce_lr=0,
           predict=False):
        
        assert self.scaled
        self.attrs = {}
        self.attrs['learning_rate'] = learning_rate
        self.attrs['batch_size']    = batch_size
        self.attrs['nepochs']       = nepochs
        self.attrs['nchain']        = nchain
        self.attrs['units']         = units
        self.attrs['scale']         = scale
        self.attrs['patience']      = patience
        self.attrs['min_delta']     = min_delta        
        self.attrs['global_seed']   = global_seed
        self.attrs['model_dir']     = model_dir
        self.attrs['save_model']    = save_model
        
        if not os.path.isdir(model_dir) and save_model:
            self.logger.info('create {}'.format(model_dir))
            os.makedirs(model_dir)
        
        #--- early stopping 
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                      patience=patience, 
                                                      min_delta=min_delta,
                                                      verbose=verbose_early)
        
        # reduce learning rate
        reduce_lr  = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                          factor=0.2,
                                                          patience=5, 
                                                          min_lr=0.0001,
                                                          verbose=verbose_reduce_lr)
        tf.random.set_seed(global_seed)
        numpy.random.seed(global_seed)
        seeds = numpy.random.randint(0, 4294967295, size=nchain)

        self.history = []
        if predict:self.ypreds  = []

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        for ii in range(nchain):
            self.logger.info('chain-{} with seed : {}'.format(ii, seeds[ii]))
            
            kwargs    = dict(kernel_regularizer=tf.keras.regularizers.l2(scale),
                              kernel_initializer=tf.keras.initializers.he_uniform(seed=seeds[ii]))                        
            model     = get_model(self.train.x.shape[1], units, **kwargs)            
            model.compile(loss='mse', 
                          optimizer=optimizer,
                          metrics=['mae', 'mse'])  
            
            t0 = time.time()            
            self.history.append(model.fit(
                               self.train.x, self.train.y,
                               epochs=nepochs,
                               sample_weight=self.train.w,
                               validation_data=(self.valid.x, self.valid.y, self.valid.w),
                               verbose=verbose,
                               callbacks=[early_stop, PrintDot(), reduce_lr],
                               batch_size=batch_size))            
            self.logger.info('done in {:.1f} secs'.format(time.time()-t0))              
            
            if predict:                
                #--- perform on the test set                  
                ypred = self._evaluate(model)
                self.ypreds.append(ypred.flatten())  # flatten the target
            
            if save_model:
                model_name = model_dir + f'model_{ii}.h5'
                model.save(model_name)  
                self.logger.info('save model at {}'.format(model_name))
                

        self.total_val_loss = sum([min(history_i.history['val_loss'])\
                                  for history_i in self.history])
        self.logger.info(f'Total val loss : {self.total_val_loss}')


        
    def _scale(self):
        ''' Z-score normalization 
        
        '''
        self.scaled  = True        
        self.logger.info('Scale features and label')
        
        self.meanx  = numpy.mean(self.train.x, axis=0)
        self.stdx   = numpy.std(self.train.x,  axis=0)
        self.meany  = numpy.mean(self.train.y, axis=0)
        self.stdy   = numpy.std(self.train.y,  axis=0)

        assert numpy.all(self.stdx != 0.0)
        assert (self.stdy != 0.0)        

        
        self.train.x = (self.train.x - self.meanx) / self.stdx
        self.train.y = (self.train.y - self.meany) / self.stdy
        
        self.test.x  = (self.test.x - self.meanx) / self.stdx
        self.test.y  = (self.test.y - self.meany) / self.stdy
        
        self.valid.x = (self.valid.x - self.meanx) / self.stdx
        self.valid.y = (self.valid.y - self.meany) / self.stdy   
                
    def _descale(self):
        ''' Undo Z-score normalization
        
        '''
        self.scaled  = False
        self.train.x = self.train.x*self.stdx + self.meanx
        self.train.y = self.train.y*self.stdy + self.meany
        
        self.test.x  = self.test.x*self.stdx + self.meanx
        self.test.y  = self.test.y*self.stdy + self.meany
        
        self.valid.x = self.valid.x*self.stdx + self.meanx
        self.valid.y = self.valid.y*self.stdy + self.meany
        
        if hasattr(self, 'ypreds'):
            self.ypreds = numpy.array(self.ypreds)
            self.ypreds = (self.ypreds*self.stdy + self.meany)
            
    def _evaluate(self, model, verbose=0):
        assert self.scaled
        ypred = model.predict(self.test.x)    
        loss, mae, mse = model.evaluate(self.test.x, self.test.y, 
                                        sample_weight=self.test.w,
                                        verbose=verbose)    
        # baseline
        assert numpy.mean(self.train.y) < 1.e-8
        mse_base    = numpy.mean(self.test.y*self.test.y)
        self.logger.info('Test LOSS : {0:.3f} MAE : {1:.3f} MSE : {2:.3f}'\
                    .format(loss, mae, mse))
        self.logger.info('Baseline test MSE : {:.3f}'.format(mse_base))    # variance of the test.y (if mean (test.y) = 0)
        self.logger.info('Variance of the test label : {:.3f}'.format(numpy.var(self.test.y)))
        return ypred
        
    def load(self, models):
        self.models = []
        for model in models:
            self.models.append(tf.keras.models.load_model(model))
            
        
    def predict(self, verbose=0):  
        assert self.scaled        
        if not hasattr(self, 'ypreds'):
            self.ypreds = []
        if not hasattr(self, 'models'):
            raise ValueError(f'{self} should have `models`')
            
        for model in self.models:
            ypred = model.predict(self.test.x)    
            loss, mae, mse = model.evaluate(self.test.x, self.test.y, 
                                            sample_weight=self.test.w,
                                            verbose=verbose)    
            # baseline
            assert numpy.mean(self.train.y) < 1.e-8
            mse_base    = numpy.mean(self.test.y*self.test.y)
            self.ypreds.append(ypred.flatten())  # flatten the target
            self.logger.info('Test LOSS : {0:.3f} MAE : {1:.3f} MSE : {2:.3f}'\
                        .format(loss, mae, mse))
            self.logger.info('Baseline test MSE : {:.3f}'.format(mse_base))
            self.logger.info('Variance of the test label : {:.3f}'.format(numpy.var(self.test.y)))

    def make_plots(self):
        self.plot_histograms()
        self.plot_metrics()
        self.plot_deltaY()
        self.plot_weights(0)
        
    def plot_histograms(self, cf_min=0.05, bins=6):
        from scipy.stats import spearmanr, binned_statistic
        #cf_min  = 0.05 
        indices = []
        for j in range(self.test.x.shape[1]):
            cf =  spearmanr(self.test.x[:, j], self.test.y)[0]
            if abs(cf) > cf_min:
                indices.append(j)
        
        # --- plot
        
        nrows = len(indices)//2 if len(indices)%2 ==0 else len(indices)//2 + 1
        fig, ax = plt.subplots(ncols=2, nrows=nrows, sharey=True, 
                               figsize=(6*2, nrows*4))
        fig.subplots_adjust(hspace=0.25, wspace=0.15, top=0.95)
        ax = ax.flatten()
        
        npanels = len(ax)
        j = -1
        while len(indices) < npanels:
            fig.delaxes(ax[j])
            j -= 1
            npanels -= 1
            

        fig.suptitle('Mean Y vs Important Features')
        for i,index in enumerate(indices):    
            
            bin_means, bin_edges, _  = binned_statistic(self.test.x[:, index], 
                                                        self.test.y, 
                                                        statistic='mean',
                                                        bins=bins)
            
            bin_means2, bin_edges, _ = binned_statistic(self.test.x[:, index],  
                                                        numpy.mean(self.ypreds, 0),
                                                        statistic='mean', 
                                                        bins=bin_edges)
            
            ax[i].scatter(bin_edges[:-1], bin_means,  color='k', alpha=0.8, marker='s')
            ax[i].scatter(bin_edges[:-1], bin_means2, color='r', alpha=0.8, marker='o')
            ax[i].set_xlabel(f'Feature-{index}')      
            ax[i].grid(True, linestyle='--', color='grey')
            
    def plot_metrics(self, **kwargs):
        # yscale='linear', xscale='linear'
        plot_history(self.history, **kwargs)
        
    def plot_deltaY(self, **kwargs):
        # color='k', bins=40, alpha=0.8
        sf = self.stdy if self.scaled else 1
        plt.hist(sf*(self.test.y-numpy.mean(self.ypreds, 0)), **kwargs)
        plt.yscale('log')
        plt.xlabel('Ytrue - Ypred')
    
    def plot_weights(self, chain, **kwargs):
        weights = self.history[chain].model.get_weights()
        ncols   = len(weights) //2
        fig, ax = plt.subplots(ncols=ncols, figsize=(4*ncols, 5))
        fig.subplots_adjust(wspace=0.0, top=0.9)
        fig.suptitle('Parameters')

        for j in range(ncols):
            ax[j].set_title(f'Layer - {j}')
            extend = [0, 20, 0, 20]
            map1 = ax[j].imshow(numpy.row_stack([weights[2*j], weights[2*j+1]]), 
                         cmap=plt.cm.seismic, vmin=-2., vmax=2.)#, extent=extend)
            #plt.setp(ax[j].get_xticklabels(), visible=False)
            #plt.setp(ax[j].get_yticklabels(), visible=False)
            ax[j].tick_params(
                    axis='both',        # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    right=False,
                    left=False,
                    labelbottom=False) # labels along the bottom edge are off
            ax[j].set(yticks=[], xticks=[])        
            if j==0:ax[j].set(ylabel='Input Features (including bias)')

        cax = plt.axes([.92, 0.2, 0.01, 0.6])
        #cax = plt.axes([.25, 0.05, 0.5, 0.05]) ## goes w orientation='horizontal'
        ## colorbar label='parameters'
        fig.colorbar(map1, cax=cax, 
                     shrink=0.7, ticks=[-2, 0, 2], 
                     extend='both')
        
    
class Data(object):
    '''
        Class to facilitate reading data 
        
        parameters
        ----------
        data : `numpy` structured array
        it should have `features`, `label`, `fracgood`, and `hpind`
        axes : list, optional
        a list of column indices that  will be used for training
    '''
    
    def __init__(self, data, cachex=False):
        
        self.x = data['features']
        self.y = data['label']
        self.w = data['fracgood']
        self.p = data['hpind']        
            
        if cachex:self.xc = self.x.copy()
            
    def __call__(self, axes=None):
        if not hasattr(self, 'xc'):
            self.xc = self.x.copy()
        
        if not axes is None:
            self.x = self.xc[:, axes]
        if len(self.x.shape) == 1:
            self.x = self.x[:, numpy.newaxis]
            

class PrintDot(tf.keras.callbacks.Callback):
    '''
        Displays training progress 
        by printing a single dot 
        for each completed epoch
    '''
    def on_epoch_end(self, epoch, logs):
        #if epoch % 100 == 0:print('')
        print('.', end='')
               
'''

    Models with different number of hidden layers

'''                
def model0(nfeature, units=[0], **kwargs):
    assert (len(units)==1) & (units[0]==0)
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=[nfeature], **kwargs)
            ])
    return model

def model1(nfeature, units=[40], **kwargs):
    assert (len(units)==1) & (units[0]!=0)
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(units[0], activation='relu', 
                         input_shape=[nfeature], **kwargs),
            tf.keras.layers.Dense(1, **kwargs)
            ])
    return model       

def model2(nfeature, units=[20, 20], **kwargs):
    assert (len(units)==2) & (units[0]!=0) & (units[1]!=0) 
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(units[0], activation='relu', 
                         input_shape=[nfeature], **kwargs),
            tf.keras.layers.Dense(units[1], activation='relu', **kwargs),
            tf.keras.layers.Dense(1, **kwargs)
            ])
    return model

def model3(nfeature, units=[20, 20, 20], **kwargs):
    assert (len(units)==3) & (units[0]!=0)\
            & (units[1]!=0) & (units[2]!=0) 
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(units[0], activation='relu', 
                         input_shape=[nfeature], **kwargs),
            tf.keras.layers.Dense(units[1], activation='relu', **kwargs),
            tf.keras.layers.Dense(units[2], activation='relu', **kwargs),
            tf.keras.layers.Dense(1, **kwargs)
            ])
    return model

def model4(nfeature, units=[20, 20, 20, 20], **kwargs):
    assert (len(units)==4) & (units[0]!=0) \
            & (units[1]!=0) & (units[2]!=0) & (units[3]!=0) 
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(units[0], activation='relu', 
                         input_shape=[nfeature], **kwargs),
            tf.keras.layers.Dense(units[1], activation='relu', **kwargs),
            tf.keras.layers.Dense(units[2], activation='relu', **kwargs),
            tf.keras.layers.Dense(units[3], activation='relu', **kwargs),        
            tf.keras.layers.Dense(1, **kwargs)
            ])
    return model


def get_model(nfeatures, units, **kwargs):
    if (len(units)==1) & (units[0]==0):
        #logger.info('run linear model')
        model = model0(nfeatures, units, **kwargs)
    elif (len(units)==1) & (units[0]!=0):
        #logger.info('run with one hidden layer')
        model = model1(nfeatures, units, **kwargs)
    elif (len(units)==2) & (units[0]!=0) & (units[1]!=0):
        #logger.info('run with two hidden layers')
        model = model2(nfeatures, units, **kwargs)
    elif (len(units)==3) & (units[0]!=0) & (units[1]!=0)\
        & (units[2]!=0):
        #logger.info('run with three hidden layers')
        model = model3(nfeatures, units, **kwargs)  
    elif (len(units)==4) & (units[0]!=0) & (units[1]!=0)\
        & (units[2]!=0) & (units[3]!=0):
        #logger.info('run with four hidden layers')
        model = model4(nfeatures, units, **kwargs)                  
    else:
        #logger.info('`units`={} is not implemented'.format(units))
        raise ValueError('Units should be either None, [M], [M,N] ...')
    return model


def plot_history(history_list, labels=None, yscale='linear', xscale='linear'):
    '''
        Plots MAE and MSE vs. epoch
        
    '''

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 8))
    plt.subplots_adjust(hspace=0.0)
    # fig 0
    ax[0].set_ylabel('Mean Abs Error')
    
    n = len(history_list)
    c = plt.cm.jet    
    for i,history in enumerate(history_list):
        #
        hist = history.history#pd.DataFrame(history.history)
        hist['epoch'] = history.epoch    
        #
        ax[0].plot(hist['epoch'], hist['mae'], color=c(i/n))
        ax[0].plot(hist['epoch'], hist['val_mae'], ls='--', color=c(i/n))
        ax[1].plot(hist['epoch'], hist['mse'], color=c(i/n))
        ax[1].plot(hist['epoch'], hist['val_mse'], ls='--', color=c(i/n))
        
    
    # fig 1 
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Mean Square Error')
    if labels is None:
        labels = ['model-'+str(i) for i in range(n)]    
    for i, label_i in enumerate(labels):
        ax[0].text(0.8, 0.9-0.1*i, label_i, \
                   color=c(i/n), transform=ax[0].transAxes)
    #ax[0].legend()
    for ax_i in ax:
        ax_i.grid(True, linestyle='--', color='grey')
        ax_i.set(yscale=yscale, xscale=xscale)
    plt.show() 

        