
from LSSutils.lab  import CurrentMPIComm,  run_ConvolvedFFTPower
from LSSutils import setup_logging

setup_logging("info") # turn on logging to screen
comm = CurrentMPIComm.get()


if comm.rank ==0:

    from argparse import ArgumentParser
    ap = ArgumentParser(description='Power Spectrum (NBODYKIT)')
    ap.add_argument('-g', '--galaxy_path', required=True)
    ap.add_argument('-r', '--random_path', required=True)
    ap.add_argument('-o', '--output_path', required=True)
    ap.add_argument('-n', '--nmesh', type=int, default=256)
    ap.add_argument('--dk', type=float, default=0.002)
    ap.add_argument('-b', '--boxsize', type=float, default=None)
    ap.add_argument('-z', '--zlim', nargs='*', type=float, default=[0.8, 2.2])    
    ap.add_argument('--use_systot', action='store_true')
    ns = ap.parse_args()

    for (key, value) in ns.__dict__.items():
        print(f'{key:15s} : {value}') 

else:
    ns = None

ns = comm.bcast(ns, root=0)


run_ConvolvedFFTPower(ns.galaxy_path, ns.random_path, ns.output_path, 
                      use_systot=ns.use_systot, zmin=ns.zlim[0], zmax=ns.zlim[1],
                      dk=ns.dk, nmesh=ns.nmesh, boxsize=ns.boxsize)