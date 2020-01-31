import logging
import regression
import numpy

class FeatureElimination:
    '''
    Perform Recursive Feature Elimination
    
    
    Ex. 
    data  = np.load('/Users/mehdi/Downloads/trunk/qso.ngc.all.hp.256.r.npy', allow_pickle=True).item()
    #-- start from all indices
    train = regression.Data(data['train']['fold0'],      cachex=True)
    valid = regression.Data(data['validation']['fold0'], cachex=True)
    test  = regression.Data(data['test']['fold0'],       cachex=True)

    FE = FeatureElimination(train, valid, test)
    FE([0, 1, 2]) # only work with 3 maps
    
    '''
    logger = logging.getLogger('FeatureElimination')
    
    def __init__(self, train, validation, test):
        self.train   = train
        self.test    = test
        self.valid   = validation
        self.results = {'validmin':[], 'importance':[], 'indices':[]}
        self.kwargs  = dict(predict=False, min_delta=1.e-8,
                            batch_size=1024, units=[0],
                            learning_rate=0.1)
        
    def __call__(self, indices):

        if len(indices) == 1:
            # this means that we are left with one map
            # append this map to the importance, and return
            self.results['importance'].append(indices[1])
            return self.results

        vloss = []
        self.logger.info(f'Perform Feature Selection with {indices}')
        for index in indices:
            in_indices = indices.copy()
            in_indices.remove(index)

            #print(index_out, in_indices)
            self.train(axes=in_indices)
            self.valid(axes=in_indices)
            self.test(axes=in_indices)


            Net = regression.NetRegression(self.train, 
                                           self.valid,
                                           self.test)    
            #Net.fit_w_hparam_training()
            Net.fit(**self.kwargs)
            vloss.append(Net.total_val_loss)

        arg = numpy.argmin(vloss)
        self.logger.info(f'attribute index-{arg} with {vloss[arg]}')
        self.results['validmin'].append(vloss)
        self.results['indices'].append(indices.copy())
        self.results['importance'].append(indices.pop(arg))
        
        self(indices)