from .version import __version__
from mpi4py import MPI
import dask
import warnings




try:
    dask.config.set(scheduler='synchronous')
except:
    dask.set_options(get=dask.get)

_global_options = {}
_global_options['global_cache_size'] = 1e8
_global_options['dask_chunk_size']   = 10000
_global_options['paint_chunk_size']  = 1024*1024*4

def initiate_logging(level='info'):
    #
    # (c) nbodykit Nick Hand, Yu Feng 2017
    #
    import logging
    import time

    levels = {'info':logging.INFO, 'debug':logging.DEBUG, 
              'warning':logging.WARNING, 'error':logging.ERROR}

    # create logger
    logger = logging.getLogger()

    t0 = time.time()
    rank = MPI.COMM_WORLD.rank

    class Formatter(logging.Formatter):
        def format(self, record):
            s1 = ('[ %09.2f ] % 3d: ' % (time.time() - t0, rank))
            return s1 + logging.Formatter.format(self, record)

    fmt = Formatter(fmt='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M ')

    logging_handler = logging.StreamHandler()
    logger.addHandler(logging_handler)
    logging_handler.setFormatter(fmt)
    logger.setLevel(levels[level])
