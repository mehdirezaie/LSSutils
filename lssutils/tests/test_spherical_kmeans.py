""" Test Spherical K-Means

"""
import numpy as np
from lssutils.utils import SphericalKMeans

def generate_data():
    
    np.random.seed(42)
    ndata = 1000
    
    x0 = np.random.normal(100, 10, size=ndata)
    y0 = np.random.normal(10, 10, size=ndata)
    c0 = np.zeros_like(x0)

    x1 = np.random.normal(10, 10, size=ndata)
    y1 = np.random.normal(-45, 10, size=ndata)
    c1 = np.ones_like(x1)

    ra = np.concatenate([x0, x1])
    dec = np.concatenate([y0, y1])
    z = np.concatenate([c0, c1])
    
    return ra, dec, z


def main():
    """
    tests Spherical K-means on ra,dec: (0,-40) and (100, 20)
    """
    ra, dec, z = generate_data()
    km = SphericalKMeans(n_clusters=2)
    km.fit_radec(ra, dec)
    assert np.allclose(km.predict_radec([10, 100], [-45, 10]), [1, 0])
    print('cluster centers (ra, dec):', km.centers_radec)
    
if __name__ == '__main__':
    main()
    
