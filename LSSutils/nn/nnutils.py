import matplotlib 
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd




def plot_prederr(test_labels, test_predictions):
    error = test_predictions - test_labels

    fig, ax = plt.subplots(ncols=2)
    ax[0].scatter(test_labels, test_predictions)
    ax[0].set_xlabel('True Values')
    ax[0].set_ylabel('Predictions')
    #plt.axis('equal')
    #plt.axis('square')
    #plt.xlim([0,plt.xlim()[1]])
    #plt.ylim([0,plt.ylim()[1]])
    #ax[0].plot([-100, 100], [-100, 100])
    ax[1].hist(error, bins = 25)
    ax[1].set_xlabel("Prediction Error")
    ax[1].set_ylabel("Count")
    plt.show()

              
        
def plot_history(history_list, labels=None):
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
        hist = pd.DataFrame(history.history)
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
        ax[0].text(0.8, 0.9-0.1*i, label_i, color=c(i/n), transform=ax[0].transAxes)
    #ax[0].legend()
    for ax_i in ax:ax_i.grid(True)
    plt.show()        
