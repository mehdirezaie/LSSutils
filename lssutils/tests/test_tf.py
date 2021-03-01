# title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import matplotlib as mpl
#mpl.use('Agg')
mpl.use('TKAgg')
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print('testing tf :', tf.__version__)



if os.path.isfile('/Users/mehdi/.keras/datasets/auto-mpg.data'):
   dataset_path = '/Users/mehdi/.keras/datasets/auto-mpg.data'
else:
   dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu"\
                                   +"/ml/machine-learning-databases/"\
                                   +"auto-mpg/auto-mpg.data") 

# read using pandas
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()

# drop nan
dataset = dataset.dropna()

# origin 
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0

# split 
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset  = dataset.drop(train_dataset.index)

# get training data stats
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

# get label
train_labels = train_dataset.pop('MPG')
test_labels  = test_dataset.pop('MPG')

# scale
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data  = norm(test_dataset)

# define the model
def build_model():
    tf.keras.backend.clear_session()
    kwargs = dict(kernel_regularizer=keras.regularizers.l2(0.001),
                  kernel_initializer=keras.initializers.he_normal(seed=123456))
    nfeature = len(train_dataset.keys())
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=[nfeature], **kwargs),
        layers.Dense(128, activation='relu', **kwargs),
        layers.Dense(128, activation='relu', **kwargs),
        layers.Dense(1,  **kwargs)
        ])

    #optimizer = tf.keras.optimizers.RMSprop(0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig, ax = plt.subplots(nrows=2)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Mean Abs Error [MPG]')
    ax[1].plot(hist['epoch'], hist['mae'],
           label='Train Error')
    ax[1].plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
    ax[1].set_ylim([0,5])
    ax[1].legend()

    #plt.figure()
    #ax[0].xlabel('Epoch')
    ax[0].set_ylabel('Mean Square Error [$MPG^2$]')
    ax[0].plot(hist['epoch'], hist['mse'],
           label='Train Error')
    ax[0].plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
    ax[0].set_ylim([0,20])
    #ax[0].legend()
    plt.show()

def plot_prederr(test_labels, test_predictions):
    error = test_predictions - test_labels

    fig, ax = plt.subplots(ncols=2)
    ax[0].scatter(test_labels, test_predictions)
    ax[0].set_xlabel('True Values [MPG]')
    ax[0].set_ylabel('Predictions [MPG]')
    #plt.axis('equal')
    #plt.axis('square')
    #plt.xlim([0,plt.xlim()[1]])
    #plt.ylim([0,plt.ylim()[1]])
    ax[0].plot([-100, 100], [-100, 100])
    ax[1].hist(error, bins = 25)
    ax[1].set_xlabel("Prediction Error [MPG]")
    ax[1].set_ylabel("Count")
    plt.show()


# Display training progress 
# by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 1000
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history    = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                       validation_split = 0.2, verbose=0, 
                       callbacks=[early_stop, PrintDot()])

plot_history(history)

print('')
print('apply on test')
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))



test_predictions = model.predict(normed_test_data).flatten()
plot_prederr(test_labels, test_predictions)

