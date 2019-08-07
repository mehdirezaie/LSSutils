import tensorflow as tf
from   tensorflow import keras
from   tensorflow.keras import layers
import time

class PrintDot(keras.callbacks.Callback):
    '''
        Displays training progress 
        by printing a single dot 
        for each completed epoch
    '''
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:print('')
        print('.', end='')


        
def model0(nfeature, units=[0]):
    assert (len(units)==1) & (units[0]==0)
    tf.keras.backend.clear_session()
    kwargs = dict(kernel_regularizer=keras.regularizers.l2(0.0),
                  kernel_initializer=keras.initializers.he_normal(seed=123456))
    model = keras.Sequential([
        layers.Dense(1, input_shape=[nfeature], **kwargs)
        ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

def model1(nfeature, units=[40]):
    assert (len(units)==1) & (units[0]!=0)
    tf.keras.backend.clear_session()
    kwargs = dict(kernel_regularizer=keras.regularizers.l2(0.0),
                  kernel_initializer=keras.initializers.he_normal(seed=123456))
    model = keras.Sequential([
        layers.Dense(units[0], activation='relu', input_shape=[nfeature], **kwargs),
        layers.Dense(1, **kwargs)
        ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model       
        
def model2(nfeature, units=[20, 20]):
    assert (len(units)==2) & (units[0]!=0) & (units[1]!=0) 
    tf.keras.backend.clear_session()
    kwargs = dict(kernel_regularizer=keras.regularizers.l2(0.0),
                  kernel_initializer=keras.initializers.he_normal(seed=123456))
    model = keras.Sequential([
        layers.Dense(units[0], activation='relu', input_shape=[nfeature], **kwargs),
        layers.Dense(units[1], activation='relu', **kwargs),
        layers.Dense(1, **kwargs)
        ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

def run_model(nfeature, train, valid, test,
              batch_size=256, nepochs=100, units=[0], verbose=0):
    '''
        
    '''
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                               patience=10, min_delta=1.e-6)
    if (len(units)==1) & (units[0]==0):
        print('run linear model')
        model = model0(nfeature, units)
    elif (len(units)==1) & (units[0]!=0):
        print('run with one hidden layer')
        model = model1(nfeature, units)
    elif (len(units)==2) & (units[0]!=0) & (units[1]!=0):
        print('run with two hidden layer')
        model = model2(nfeature, units)
        
    # run
    t0 = time.time()
    history = model.fit(
                    train.X, train.Y,
                    epochs=nepochs, validation_data=(valid.X, valid.Y), verbose=verbose,
                    callbacks=[early_stop, PrintDot()], batch_size=batch_size)
    print('done in {:.1f} secs'.format(time.time()-t0))
    
    # perform on test
    Ypred = model.predict(test.X)    
    loss, mae, mse = model.evaluate(test.X, test.Y, verbose=verbose)    
    return {'history':history, 'eval':{'loss':loss, 'mae':mae, 'mse':mse}, 'Ypred':Ypred}


