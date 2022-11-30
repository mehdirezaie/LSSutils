import sys
from lssutils.utils import combine_nn


fnl = sys.argv[1]
mock = sys.argv[2]
method = sys.argv[3] #'dnnp_known1'

print(f'fnl: {fnl}')

maps = [f'/fs/ess/PHS0336/data/lognormal/v3/regression/fnl_{fnl}/{mock}/{method}/bmzls/nn-weights.fits',
        f'/fs/ess/PHS0336/data/lognormal/v3/regression/fnl_{fnl}/{mock}/{method}/ndecalsc/nn-weights.fits',
        f'/fs/ess/PHS0336/data/lognormal/v3/regression/fnl_{fnl}/{mock}/{method}/sdecalsc/nn-weights.fits']

combine_nn(maps, f'/fs/ess/PHS0336/data/lognormal/v3/regression/fnl_{fnl}/{mock}/{method}_lrg_desic.hp256.fits')
