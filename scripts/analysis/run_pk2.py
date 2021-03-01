
from lssutils.lab  import CurrentMPIComm,  run_ConvolvedFFTPower
from lssutils import setup_logging

setup_logging("info") # turn on logging to screen
comm = CurrentMPIComm.get()


if comm.rank ==0:

    from argparse import ArgumentParser
    ap = ArgumentParser(description='Power Spectrum (NBODYKIT)')
    ap.add_argument('-g', '--galaxy_path', required=True)
    ap.add_argument('-r', '--random_path', required=True)
    ap.add_argument('-o', '--output_path', required=True)
    ap.add_argument('-n', '--nmesh', nargs='*', type=int, default=512)
    ap.add_argument('--dk', type=float, default=0.001903995548) # 4pi/boxsize
    ap.add_argument('--kmax', type=float, default=None)
    ap.add_argument('-z', '--zlim', nargs='*', type=float, default=[0.8, 2.2]) 
    ap.add_argument('--cosmo', type=str, default=None)
    ap.add_argument('-p', '--poles', nargs='*', type=int, default=[0, 2, 4]) 
    ap.add_argument('--use_systot', action='store_true')
    ns = ap.parse_args()

    for (key, value) in ns.__dict__.items():
        print(f'{key:15s} : {value}') 

else:
    ns = None

ns = comm.bcast(ns, root=0)


run_ConvolvedFFTPower(ns.galaxy_path, ns.random_path, ns.output_path, 
                      use_systot=ns.use_systot, zmin=ns.zlim[0], zmax=ns.zlim[1],
                      dk=ns.dk, kmax=ns.kmax, nmesh=ns.nmesh,
                      poles=ns.poles, cosmology=ns.cosmo)
