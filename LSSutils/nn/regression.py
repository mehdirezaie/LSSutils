'''
    This is part of a bigger project,
    LSSutils (utilities to analyze Large Scale Structure)
    The goal is to repackage the Neural Network based mitigation software
    https://github.com/mehdirezaie/SYSNet

    credit: Mehdi Rezaie, mr095415@ohio.edu


    This file hosts all the functionality related to the regression.
    If used as a code, eg.

    $> python regression.py

    It will run a test to model the function sin(x)
    and make two scatter plots of y_true vs y_pred, etc.

'''




# --- modules
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy
import time
import os
import logging

from copy import copy



__all__ = ['LinearRegression', 'NetRegression', 'Data']


class LinearRegression(object):
    """
    example:
    --------
    df = dataloader(300)
    LR = LinearRegression(verbose=True)
    LR.fit(df.loc[:, ['x1', 'x2']].values, df['class'].values)
    """
    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, X, Y):
        '''
        inputs
        --------
        X: features, [N, M]
        Y: label [N, ]
        returns None
        '''
        intersect = numpy.ones(X.shape[0])
        X = numpy.column_stack([intersect, X])
        XX = X.T.dot(X)
        invXX = numpy.linalg.inv(XX)
        self.Beta = invXX.dot(X.T.dot(Y))

        if self.verbose:
            print(f'{self.Beta} with RSS: {self._rss(X, Y):.2f}')

    def predict(self, X):
        '''
        inputs
        ------
        X: featues, [N, M]
        returns X.T.Beta
        '''
        intersect = numpy.ones(X.shape[0])
        X = numpy.column_stack([intersect, X])
        return X.dot(self.Beta)

    def _rss(self, X, Y):
        '''
        inputs
        --------
        X: features, [N, M]
        Y: label [N, ]
        returns RSS
        '''
        return ((Y - X.dot(self.Beta))**2).sum()

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

        data  = np.load('/Users/mehdi/Downloads/trunk/qso.ngc.all.hp.256.r.npy',\
                        allow_pickle=True).item()
        train = regression.Data(data['train']['fold0'])
        valid = regression.Data(data['validation']['fold0'])
        test  = regression.Data(data['test']['fold0'])



        t_i = time.time()
        Net = regression.NetRegression(train, valid, test)
        #Net.fit(hyperparams=True) # train with hyperparameters
        Net.fit(predict=True, min_delta=1.e-8,
                batch_size=1024, units=[10, 10],
                learning_rate=0.1)
        Net._descale() # descale
        Net.make_plots()
        t_f = time.time()
        print(f'took {t_f-t_i} secs')
    '''

    logger = logging.getLogger('NetRegression')

    def __init__(self, train, valid, test, norm_output=None):
        self.train   = copy(train)
        self.valid   = copy(valid)
        self.test    = copy(test)
        self.norm_output = norm_output

        for data in [train, valid, test]:
            for attr in ['x', 'y', 'w', 'p']:
                if not hasattr(data, attr):
                    raise AttributeError('Inputs must have a ``%s`` attribute'%attr)

        self._scale()

    def fit(self, hyperparams=False, options=None, **kwargs):
        '''
        performs training, validation, and testing

        parameters
        -----------
        hyperparams : boolean (default=False), it will perform a greedy grid
                      search  of the best hyperparameters

        options : dictionary, the hyper-parameters and their search space

        **kwargs: optional parameters for the ._fit method

        '''
        if hyperparams:
            self._fit_w_hyperparams(options=options, **kwargs)
        else:
            self._fit(**kwargs)


    def _fit_w_hyperparams(self, options=None, model_dir=None, **kwargs):

        if options is None:
            '''
                only trains the number of layers,
                L2 regularization,
                and mini-batch size

            '''
            options = {
                       'units':[[40], [20,20], [20, 10, 10], [10, 10, 10, 10]],
                       'scale':[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.],
                       'batch_size':[128, 256, 512, 1024, 2048, 4096]
                       }


        self._fit()  # run the default
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
                self._fit(**attrs)
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
        model_dir = kwargs.pop('model_dir', None)
        attrs.update(model_dir=model_dir)
        attrs.update(nchain=1)
        attrs.update(predict=True)
        self.logger.info(f'final training with {attrs}')
        self._fit(**attrs)


    def _fit(self,
           learning_rate=0.01,
           batch_size=512,
           nepochs=500,
           nchain=1,
           units=[20, 20],
           min_delta=1.e-5,
           scale=0.0,
           patience=10,
           verbose=0,
           global_seed=123567,
           model_dir=None,
           verbose_early=0,
           verbose_reduce_lr=0,
           predict=False):
        '''
           learning_rate: float, the learning rate default = 0.01
           batch_size: integer, the size  of the batch, default= 256
           nepochs: integer, the size of the maximum epoch, default=500
           nchain: integer,  the number of chains, each chain with different initialization,  default=1
           units: list of ints, the number of  units in the neural network,  default=[20, 20]
           min_delta:float, the minimum tolerance for early stopping default=1.e-5
           scale:float, L2 regularization strength default=0.0
           patience:int, the number of times early stopper waits, default=10
           verbose:int, 0:verbosity off, 1: verbosity on  default=0,
           global_seed: int, global seed to make the result  reproducible, default= 123567,
           model_dir:str, path  to save the model, default='model/',
           verbose_early:int, verbosity for the early stopping , default (off) 0
           verbose_reduce_lr:int, verbosity for  the learning rate  reducer, default=0 (off),
           predict: boolean, apply the trained model on the test set, default=False
        '''
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

        if nchain > 1:
            self.logger.warning('FIXME: save model fails')
        
        '''
        FIXME: 
        if model_dir is not None:
            model_path = ''.join(model_dir.split('/')[:-1])

            if not os.path.isdir(model_path): # /path/to/file -> /path/to/
                self.logger.info('create {}'.format(model_path))
                os.makedirs(model_path)
        '''
        
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

            if model_dir is not None:
                model_name = ''.join([model_dir, '_chain', str(ii), '.h5'])
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
        
        #
        if self.norm_output is not None:
            
            kwargs = {
                'meanx':self.meanx,
                'stdx':self.stdx,
                'meany':self.meany,
                'stdy':self.stdy
                        }
            numpy.savez(self.norm_output, **kwargs)

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
        # calls, model.predict
        # performs the trained model on the test set
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

        # variance of the test.y (if mean (test.y) = 0)
        self.logger.info('Baseline test MSE : {:.3f}'.format(mse_base))
        self.logger.info('Variance of the test label : {:.3f}'\
                         .format(numpy.var(self.test.y)))
        return ypred
    
    def export(self, path, aggregate='mean'):
        self.logger.info(f'aggregate with {aggregate}')
        
        assert hasattr(self, 'ypreds')        
        if self.scaled:
            self.logger.info('descale the values before export')
            self._descale()
        
        if aggregate == 'mean':
            y_ = numpy.mean(self.ypreds, axis=0)
            
        elif aggregate == 'median':
            y_ = numpy.median(self.ypreds, axis=0)
        else:
            raise RuntimeError(f'{aggregate} must be mean or median!')
            
        self.attrs['aggregation'] = aggregate
        with open(path, 'w') as file:
            
            for kw in self.attrs:
                file.write(f'# {kw}: {self.attrs[kw]}\n')
                
            npix = len(self.test.p)
            file.write('# hpix - y_pred - y_true\n')
            for i in range(npix):
                file.write('{:d} {:.8e} {:.8e}\n'.format(self.test.p[i], y_[i], self.test.y[i]))
        
        

    def load(self, models):
        # load the model
        self.models = []
        for model in models:
            self.models.append(tf.keras.models.load_model(model))


    def predict(self, verbose=0):
        # apply the trained model on the test set
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
            self.logger.info('Variance of the test label : {:.3f}'\
                             .format(numpy.var(self.test.y)))

    def make_plots(self):
        # make all the visualizations
        self.plot_histograms()  # histogram of the label vs features
        self.plot_metrics()     # MSE vs epochs
        self.plot_deltaY()      # pdf of Y_true  - Y_predicted
        self.plot_weights(0)    # colorcoded plot of the weights and biases

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

            ax[i].scatter(bin_edges[:-1], bin_means,
                          color='k', alpha=0.8, marker='s')
            ax[i].scatter(bin_edges[:-1], bin_means2,
                          color='r', alpha=0.8, marker='o')

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

    def __init__(self, data, cachex=False, isnumpy=True):

        if isnumpy:
            self.x = data['features']
            self.y = data['label']
            self.w = data['fracgood']
            self.p = data['hpind']
        else:
            self.x = data[0]
            self.y = data[1]
            self.w = data[2]
            self.p = data[3]

        if cachex:self.xc = self.x.copy()
            
    def copy(self):
        
        from copy import copy
        
        return copy(self)

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

    # nfeatures: number of input features
    # units: list of ints,
    # eg. [10, 10] a nn with two hidden layers,
    # and 10 units on each layer

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


def test_sin():
    '''
        This test relies on split2Kfolds from
        https://github.com/mehdirezaie/LSSutils/blob/master/LSSutils/utils.py
        Numpy, Matplotlib
    '''
    #import sys
    #sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    # append the path to PYTHONPATH environment variable
    from LSSutils.utils import split2Kfolds

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


    def TABLE(n=512): # create mock data
        ''' n: number of data points, default = 512
        '''
        numpy.random.seed(1234567)
        x = numpy.linspace(0., 2.*numpy.pi, n)
        z = numpy.random.uniform(0, 2*numpy.pi, size=n)
        numpy.random.shuffle(x) # inplace
        y = numpy.sin(x) #+ 0.2*z
        #x = x[:, np.newaxis]
        x = numpy.column_stack([x, z])

        n,m = x.shape
        d = numpy.empty(n, dtype=[('label', 'f8'),
                                ('features', ('f8', m)),
                                ('fracgood', 'f8'),
                                ('hpind', 'i8')])
        d['label'] = y
        if m==1:
            d['features']=x.squeeze()
        else:
            d['features']=x

        d['hpind']=1.
        d['fracgood']=1.0
        return d



    # make table [label, features, fracgood, hpind]
    Table  = TABLE()          # make table
    Data5f = split2Kfolds(Table, k=5)     # split

    # take one fold for example
    fold   = 'fold0'
    train  = Data(Data5f['train'][fold])
    test   = Data(Data5f['test'][fold])
    valid  = Data(Data5f['validation'][fold])

    # run the neural network,
    # default setting
    t_i = time.time()
    Net = NetRegression(train, valid, test)
    Net.fit(hyperparams=True)
    Net._descale() # descale
    t_f = time.time()

    # show sin(x), test and prediction
    plt.figure()
    plt.scatter(test.x[:,0], test.y)
    plt.scatter(test.x[:,0], Net.ypreds[0])
    plt.show()

    plt.figure()
    plt.scatter(test.y, Net.ypreds[0])
    plt.savefig('sinx.png')
    #plt.show()
    print(f'took {t_f-t_i} secs')



if __name__ == '__main__':
    # if run as a code, eg. $> python regression.py,
    # it will perform a regression on mock data based on sin(x)
    test_sin()
