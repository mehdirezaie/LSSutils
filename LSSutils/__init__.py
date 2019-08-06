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

def setup_logging()
