
import tensorflow as tf
from   tensorflow import keras
from   tensorflow.keras import layers
import time
import numpy as np


# set logger
import logging
logger = tf.get_logger()
logger.setLevel(logging.INFO)
logger.info('TensorFlow version: {}'.format(tf.__version__))



# kernel parameters
kwargs = dict(kernel_regularizer=keras.regularizers.l2(0.0),
              kernel_initializer=keras.initializers.he_uniform(seed=123456))
#kwargs = dict(kernel_regularizer=keras.regularizers.l2(0.0),
#              kernel_initializer=keras.initializers.glorot_uniform(seed=123456))
 

class PrintDot(keras.callbacks.Callback):
    '''
        Displays training progress 
        by printing a single dot 
        for each completed epoch
    '''
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:print('')
        print('.', end='')


        
def model0(nfeature, units=[0], **kwargs):
    assert (len(units)==1) & (units[0]==0)
    tf.keras.backend.clear_session()
    model = keras.Sequential([
            layers.Dense(1, input_shape=[nfeature], **kwargs)
            ])
    return model

def model1(nfeature, units=[40], **kwargs):
    assert (len(units)==1) & (units[0]!=0)
    tf.keras.backend.clear_session()
    model = keras.Sequential([
            layers.Dense(units[0], activation='relu', 
                         input_shape=[nfeature], **kwargs),
            layers.Dense(1, **kwargs)
            ])
    return model       

def model2(nfeature, units=[20, 20], **kwargs):
    assert (len(units)==2) & (units[0]!=0) & (units[1]!=0) 
    tf.keras.backend.clear_session()
    model = keras.Sequential([
            layers.Dense(units[0], activation='relu', 
                         input_shape=[nfeature], **kwargs),
            layers.Dense(units[1], activation='relu', **kwargs),
            layers.Dense(1, **kwargs)
            ])
    return model


class FFNN(object):
    def __init__(self, train, valid, test,
                  units=[0], monitor='val_loss',
                  patience=10, min_delta=1.e-7, 
                  learning_rate=0.01,
                  loss='mse', metrics=['mae', 'mse'],
                  **kwargs):
        '''
        
            TF 2.X compatible FFNN
        '''
        try:
            nfeature  = train.x.shape[1]
        except:
            nfeature  = 1

        self.early_stop = keras.callbacks.EarlyStopping(monitor=monitor, 
                                                        patience=patience, 
                                                        min_delta=min_delta)
        if (len(units)==1) & (units[0]==0):
            logger.info('run linear model')
            model = model0(nfeature, units, **kwargs)
        elif (len(units)==1) & (units[0]!=0):
            logger.info('run with one hidden layer')
            model = model1(nfeature, units, **kwargs)
        elif (len(units)==2) & (units[0]!=0) & (units[1]!=0):
            logger.info('run with two hidden layer')
            model = model2(nfeature, units, **kwargs)
            
        #optimizer = tf.keras.optimizers.RMSprop(0.001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)

        self.model = model
        self.train = train
        self.test  = test
        self.valid = valid
        self.units = units
        self.num   = nfeature

        self.patience  = patience
        self.min_delta = min_delta

    def run(self, batch_size=256, nepochs=500, verbose=0):
        t0 = time.time()
        history = self.model.fit(
                               self.train.x, self.train.y,
                               epochs=nepochs, 
                               validation_data=(self.valid.x, self.valid.y),
                               verbose=verbose,
                               callbacks=[self.early_stop, PrintDot()],
                               batch_size=batch_size)
        logger.info('done in {:.1f} secs'.format(time.time()-t0))

        # perform on test
        Ypred = self.model.predict(self.test.x)    
        loss, mae, mse = self.model.evaluate(self.test.x, self.test.y, 
                                             verbose=verbose)    
        self.result =  {'history':history, 
                        'eval':{'loss':loss, 'mae':mae, 'mse':mse}}
        self.Ypred  = Ypred.flatten()  # flatten the target
        logger.info('Test LOSS : {:.3f} MAE : {:.3f} MSE : {:.3f}'\
                    .format(loss, mae, mse))

    def scale(self):
        #
        # Z-score
        self.meanX   = np.mean(self.train.x)
        self.stdX    =   np.std(self.train.x)

        logger.info('scale features')
        self.train.x = (self.train.x - self.meanX)/self.stdX
        self.test.x  = (self.test.x  - self.meanX)/self.stdX
        self.valid.x = (self.valid.x - self.meanX)/self.stdX

        self.meanY   = np.mean(self.train.y)
        self.stdY    = np.std(self.train.y)

        logger.info('scale label')
        self.train.y = (self.train.y - self.meanY)/self.stdY
        self.test.y  = (self.test.y  - self.meanY)/self.stdY
        self.valid.y = (self.valid.y - self.meanY)/self.stdY

    def descale(self):
        logger.info('de-scale features')
        self.train.x = self.train.x*self.stdX + self.meanX
        self.test.x  = self.test.x*self.stdX  + self.meanX
        self.valid.x = self.valid.x*self.stdX + self.meanX

        logger.info('descale label')
        self.Ypred   = self.Ypred*self.stdY   + self.meanY
        self.train.y = self.train.y*self.stdY + self.meanY
        self.test.y  = self.test.y*self.stdY  + self.meanY
        self.valid.y = self.valid.y*self.stdY + self.meanY

    def make_plots(self):
        import matplotlib.pyplot as plt
        import LSSutils.nn.nnutils as utils
        # plots
        utils.plot_history([self.result['history']]) # plot MSE and MAE
        plt.show()
        utils.plot_prederr(self.test.y, self.Ypred)              # plot prediction error
        plt.show()


class DATA(object):
    def __init__(self, data):
        self.x = data['features']
        self.y = data['label']
        self.w = data['fracgood']
        self.p = data['hpind']

def TABLE(): # create mock
    # create data
    n = 512
    x = np.linspace(0., 2.*np.pi, n)
    np.random.shuffle(x) # inplace 
    y = np.sin(x)
    x = x[:, np.newaxis]

    n,m = x.shape
    d = np.empty(n, dtype=[('label', 'f8'), 
                            ('features', ('f8', m)),
                            ('fracgood', 'f8'),
                            ('hpind', 'i8')])
    d['label'] = y
    if m==1:
        d['features']=x.squeeze()
    else:
        d['features']=x

    d['hpind']=1.
    d['fracgood']=1.
    return d

def test():
    '''
        Test 

        generates `N' points between 0 and 2pi
        trains a network to model Sin(x)
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    sys.path.append('/Users/mehdi/github/LSSutils')
    from LSSutils.utils import split2Kfolds 



    logger.info('Data created')
    # make table [label, features, fracgood, hpind]
    Table  = TABLE()          # make table
    Data5f = split2Kfolds(Table, k=5)     # split

    # take one fold for example
    fold   = 'fold0'
    train  = DATA(Data5f['train'][fold])
    test   = DATA(Data5f['test'][fold])
    valid  = DATA(Data5f['validation'][fold])

    # run the FFNN
    myffnn = FFNN(train, valid, test, units=[100, 100], **kwargs)
    myffnn.scale()
    myffnn.run()
    myffnn.descale()
    myffnn.make_plots()

    plt.scatter(myffnn.train.x, myffnn.train.y, 2., alpha=0.5)   # plot data
    plt.scatter(myffnn.test.x,  myffnn.Ypred, 2)
    plt.show()
     
    print(myffnn.model.get_config())  # print config

if __name__ == '__main__':
    test()
