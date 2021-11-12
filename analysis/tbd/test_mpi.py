

from lssutils import setup_logging, CurrentMPIComm


setup_logging('info')
comm = CurrentMPIComm.get()

if comm.rank == 0:
   print('Hi from rank %d'%comm.rank)
else:
   print('Hey from rank %d'%comm.rank)


