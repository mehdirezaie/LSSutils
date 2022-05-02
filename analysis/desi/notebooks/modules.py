import sys
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import fitsio as ft
import healpy as hp
from glob import glob
from time import time
from scipy.optimize import minimize
import pandas as pd

HOME = os.getenv('HOME')
print(f'running on {HOME}')
sys.path.append(f'{HOME}/github/LSSutils')
sys.path.append(f'{HOME}/github/sysnetdev')
import sysnet.sources as src

from lssutils.dataviz import setup_color, add_locators, mollview, mycolor
from lssutils.utils import (histogram_cell, maps_dr9, make_hp,
                            chi2_fn, get_chi2pdf, get_inv, hpix2radec, shiftra, make_overdensity)
from lssutils.io import (read_nbmocks, read_nnbar, read_clx, read_clxmocks, 
                         read_clmocks, read_window, read_window, read_chain)
from lssutils.theory.cell import (dNdz_model, init_sample, SurveySpectrum, Spectrum, bias_model_lrg)
from lssutils.extrn.mcmc import Posterior
from lssutils.extrn import corner
from lssutils.stats.window import WindowSHT
from lssutils.stats.pcc import pcc

import getdist
from getdist import plots, MCSamples

class MCMC(MCSamples):
     def __init__(self, path_to_mcmc, read_kw=dict(), mc_kw=dict()):
            self.stats, chains = read_chain(path_to_mcmc, **read_kw)
            MCSamples.__init__(self, samples=chains, **mc_kw)