from .version import __version__
from mpi4py import MPI
#import dask
import warnings




#try:
#    dask.config.set(scheduler='synchronous')
#except:
#    dask.set_options(get=dask.get)

#_global_options = {}
#_global_options['global_cache_size'] = 1e8
#_global_options['dask_chunk_size']   = 10000
#_global_options['paint_chunk_size']  = 1024*1024*4

from contextlib import contextmanager
import logging
# see https://github.com/tensorflow/tensorflow/issues/26691
# & https://github.com/abseil/abseil-py/issues/99
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False


class CurrentMPIComm(object):
    """
    (c) Nbodykit, Nick Hand, Yu Feng
    
    
    A class to faciliate getting and setting the current MPI communicator.
    """
    _stack = [MPI.COMM_WORLD]
    logger = logging.getLogger("CurrentMPIComm")

    @staticmethod
    def enable(func):
        """
        Decorator to attach the current MPI communicator to the input
        keyword arguments of ``func``, via the ``comm`` keyword.
        """
        import functools
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            kwargs.setdefault('comm', None)
            if kwargs['comm'] is None:
                kwargs['comm'] = CurrentMPIComm.get()
            return func(*args, **kwargs)
        return wrapped

    @classmethod
    @contextmanager
    def enter(cls, comm):
        """
        Enters a context where the current default MPI communicator is modified to the
        argument `comm`. After leaving the context manager the communicator is restored.

        Example:

        .. code:: python

            with CurrentMPIComm.enter(comm):
                cat = UniformCatalog(...)

        is identical to 

        .. code:: python

            cat = UniformCatalog(..., comm=comm)

        """
        cls.push(comm)

        yield

        cls.pop()

    @classmethod
    def push(cls, comm):
        """ Switch to a new current default MPI communicator """
        cls._stack.append(comm)
        if comm.rank == 0:
            cls.logger.info("Entering a current communicator of size %d" % comm.size)
        cls._stack[-1].barrier()
    @classmethod
    def pop(cls):
        """ Restore to the previous current default MPI communicator """
        comm = cls._stack[-1]
        if comm.rank == 0:
            cls.logger.info("Leaving current communicator of size %d" % comm.size)
        cls._stack[-1].barrier()
        cls._stack.pop()
        comm = cls._stack[-1]
        if comm.rank == 0:
            cls.logger.info("Restored current communicator to size %d" % comm.size)

    @classmethod
    def get(cls):
        """
        Get the default current MPI communicator. The initial value is ``MPI.COMM_WORLD``.
        """
        return cls._stack[-1]

    @classmethod
    def set(cls, comm):
        """
        Set the current MPI communicator to the input value.
        """

        warnings.warn("CurrentMPIComm.set is deprecated. Use `with CurrentMPIComm.enter(comm):` instead")
        cls._stack[-1].barrier()
        cls._stack[-1] = comm
        cls._stack[-1].barrier()



    
_logging_handler = None
def setup_logging(log_level="info"):
    """
    (c) Nbodykit, Nick Hand, Yu Feng
    
    Turn on logging, with the specified level.

    Parameters
    ----------
    log_level : 'info', 'debug', 'warning'
        the logging level to set; logging below this level is ignored
    """

    # This gives:
    #
    # [ 000000.43 ]   0: 06-28 14:49  measurestats    INFO     Nproc = [2, 1, 1]
    # [ 000000.43 ]   0: 06-28 14:49  measurestats    INFO     Rmax = 120
    import logging

    levels = {
            "info" : logging.INFO,
            "debug" : logging.DEBUG,
            "warning" : logging.WARNING,
            }

    import time
    logger = logging.getLogger();
    t0 = time.time()

    rank = MPI.COMM_WORLD.rank

    class Formatter(logging.Formatter):
        def format(self, record):
            s1 = ('[ %09.2f ] % 3d: ' % (time.time() - t0, rank))
            return s1 + logging.Formatter.format(self, record)

    fmt = Formatter(fmt='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M ')

    global _logging_handler
    if _logging_handler is None:
        _logging_handler = logging.StreamHandler()
        logger.addHandler(_logging_handler)

    _logging_handler.setFormatter(fmt)
    logger.setLevel(levels[log_level])