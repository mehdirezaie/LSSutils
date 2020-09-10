'''
    Code to read all imaging maps 
    and write them onto a hdf file
'''

from lssutils.catalogs.combinefits import Readfits

from argparse import ArgumentParser
ap = ArgumentParser(description='systematic maps combining routine')
ap.add_argument('--paths',  type=str,  nargs='*')
ap.add_argument('--nside',  type=int,  default=256)
ap.add_argument('--tohdf',  type=str,  default='/Volumes/TimeMachine/data/DR8/dr8_combined256.h5')
ap.add_argument('--figs',   type=str,  default='/Volumes/TimeMachine/data/DR8/dr8_combined256.png')
ap.add_argument('--mkwy',   action='store_true')
ap.add_argument('--mkwytemp', type=str, nargs='*')
ap.add_argument('--debug',  action='store_true')
ns = ap.parse_args()



if ns.debug:
    args = ns.__dict__
    for (a,b) in zip(args.keys(), args.values()):
        print('{:6s}{:15s} : {}\n'.format('', a, b))
else:
    dr8data = Readfits(ns.paths, res_out=ns.nside)
    dr8data.run(add_foreground=ns.mkwy, mkwytemp=ns.mkwytemp)
    dr8data.make_plot(ns.figs)
    dr8data.save(ns.tohdf)



