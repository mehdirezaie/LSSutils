import numpy as np
from LSSutils.utils import corrmat

def main():
    x = np.random.multivariate_normal([1, -1], 
                                      [[1., 0.9], [0.9, 1.]], 
                                      size=1000)
                                  
    corr = corrmat(x, estimator='pearsonr')
    assert np.allclose(corr, [[1., 0.9], [0.9, 1.]], rtol=1.0e-2)
    print('corrmatrix:', corr)
    
    
if __name__ == '__main__':
    main()